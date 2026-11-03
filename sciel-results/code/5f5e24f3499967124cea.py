"""Matrix multiplication 2x3x5 rank minimization."""
import time
import numpy as np
from fractions import Fraction

def construct():
    start = time.time()
    n, m, p = 2, 3, 5
    N, M, P = n * m, m * p, p * n  # 6, 15, 10
    
    # Build the target tensor T of shape (N, M, P) = (6, 15, 10)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    # This encodes: coefficient of A[i0,j0]*B[j1,k1] in C[i,k] must be delta(i0,i)*delta(j0,j1)*delta(k1,k)
    
    T = np.zeros((N, M, P), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Precompute flattened T for faster access
    T_flat_a = T.reshape(N, M * P) # (6, 150)
    T_flat_b = T.transpose(1, 0, 2).reshape(M, N * P) # (15, 60)
    T_flat_c = T.transpose(2, 0, 1).reshape(P, N * M) # (10, 90)

    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        R = np.zeros((N, M, P), dtype=float)
        for c_idx in range(r):
            R += np.einsum('a,b,c->abc', U[:, c_idx], V[:, c_idx], W[:, c_idx])
        return np.allclose(R, T, atol=tol)
    
    # Schoolbook solution: rank 30
    rank_30 = n * m * p
    U30 = np.zeros((N, rank_30), dtype=float)
    V30 = np.zeros((M, rank_30), dtype=float)
    W30 = np.zeros((P, rank_30), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U30[i*m+j, col] = 1.0
                V30[j*p+k, col] = 1.0
                W30[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_30
    best_U, best_V, best_W = U30.copy(), V30.copy(), W30.copy()
    
    def try_rank(r, max_iters=100, seed=42, deadline=None):
        """Try to find a rank-r decomposition using iterative refinement."""
        np.random.seed(seed)
        
        for attempt in range(3): # Reduced attempts
            U = np.random.randn(N, r)
            V = np.random.randn(M, r)
            W = np.random.randn(P, r)
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return None, None, None, False
                
                # Update U
                # Y shape: (M*P, r). Y[b*P+c, :] = V[b, :] * W[c, :]
                # Vectorized construction:
                # V is (M, r), W is (P, r)
                # We need outer product of V and W in a specific way?
                # Y[b, c, k] = V[b, k] * W[c, k]
                # Reshape to (M*P, r)
                Y_U = np.einsum('bk,ck->bck', V, W).reshape(M * P, r)
                # Solve U[a, :] such that U[a, :] @ Y_U.T = T_flat_a[a, :]
                # Y_U.T is (r, M*P). We want U_row (1, r) such that U_row @ Y_U.T = T_row (1, M*P)
                # This is equivalent to Y_U @ U_row.T = T_row.T
                # So we solve Y_U @ x = T_flat_a[a, :].T for x (r, 1)
                for a in range(N):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(Y_U, T_flat_a[a, :].T, rcond=None)
                        U[a, :] = sol.flatten()
                    except np.linalg.LinAlgError:
                        pass

                # Update V
                Y_V = np.einsum('ak,ck->ack', U, W).reshape(N * P, r)
                for b in range(M):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(Y_V, T_flat_b[b, :].T, rcond=None)
                        V[b, :] = sol.flatten()
                    except np.linalg.LinAlgError:
                        pass

                # Update W
                Y_W = np.einsum('ak,bk->abk', U, V).reshape(N * M, r)
                for c in range(P):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(Y_W, T_flat_c[c, :].T, rcond=None)
                        W[c, :] = sol.flatten()
                    except np.linalg.LinAlgError:
                        pass
                
                # Check convergence periodically
                if it % 10 == 0:
                    R = np.zeros((N, M, P), dtype=float)
                    for c_idx in range(r):
                        R += np.einsum('a,b,c->abc', U[:, c_idx], V[:, c_idx], W[:, c_idx])
                    err = np.linalg.norm(R - T)
                    if err < 1e-6:
                        # Try rounding
                        for scale in [1, 2]:
                            U_r = np.round(U * scale) / scale
                            V_r = np.round(V * scale) / scale
                            W_r = np.round(W * scale) / scale
                            if verify(U_r, V_r, W_r, tol=1e-5):
                                return U_r, V_r, W_r, True
                        # If floating point solution is good enough, keep it? 
                        # Task requires integer/half-integer. 
                        # If we can't round, this attempt fails.
                        return None, None, None, False

        return None, None, None, False

    # Try ranks from 20 upward (30 is baseline)
    # We know 30 works. Let's try 20 and 25.
    deadline = start + 140 # Safety margin
    
    # Try Rank 20
    U_r, V_r, W_r, success = try_rank(20, max_iters=50, seed=1, deadline=deadline)
    if success:
        best_rank = 20
        best_U, best_V, best_W = U_r, V_r, W_r
    else:
        # Try Rank 25
        U_r, V_r, W_r, success = try_rank(25, max_iters=50, seed=2, deadline=deadline)
        if success:
            best_rank = 25
            best_U, best_V, best_W = U_r, V_r, W_r

    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
