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
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        # U: (N, r), V: (M, r), W: (P, r)
        r = U.shape[1]
        # Compute the reconstructed tensor
        # R[a, b, c] = sum_{c_idx} U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
        # Efficient way: for each c_idx, outer product
        R = np.zeros((N, M, P), dtype=float)
        for c_idx in range(r):
            R += np.einsum('a,b,c->abc', U[:, c_idx], V[:, c_idx], W[:, c_idx])
        return np.allclose(R, T, atol=tol)
    
    def verify_exact(U, V, W):
        """Check with exact arithmetic using Fractions."""
        r = U.shape[1]
        # Check a sample of entries for speed, or all if small enough
        # Total entries: 6*15*10 = 900, times r operations each
        # For r <= 30, that's 27000 fraction ops - should be fine
        for a in range(N):
            for b in range(M):
                for c in range(P):
                    expected = T[a, b, c]
                    actual = 0
                    for c_idx in range(r):
                        actual += U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
                    if abs(float(actual) - expected) > 1e-10:
                        return False
        return True
    
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
    
    # Try to find lower rank solutions
    # Strategy: use the structure of the problem
    # The tensor T has a special structure. Let's try various ranks.
    
    def try_rank(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using iterative refinement."""
        np.random.seed(seed)
        
        for attempt in range(5):
            # Random initialization
            U = np.random.randn(N, r)
            V = np.random.randn(M, r)
            W = np.random.randn(P, r)
            
            # Iterative refinement: minimize ||R - T||_F
            for it in range(max_iters):
                # Update U: for each a, U[a,:] should minimize ||sum_c U[a,c]V[b,c]W[c_idx,c] - T[a,b,c_idx]||
                # Fix V and W, solve for U
                # R[a,b,c] = sum_c U[a,c] * V[b,c] * W[c,c]
                # For fixed b,c: R[a,b,c] = sum_c_idx U[a,c_idx] * (V[b,c_idx]*W[c,c_idx])
                # So U[a,:] @ (V[b,:] * W[c,:]) = T[a,b,c] for all b,c
                # This is a linear system: U[a,:] @ X[b,c] = T[a,b,c]
                # where X[b,c_idx] = V[b,c_idx]*W[c,c_idx]
                # Stack over (b,c): U[a,:] @ X_flat = T[a,:,:].flatten()
                # X_flat is (M*P, r), T_flat is (M*P,)
                
                # Update U
                for a in range(N):
                    # X[b,c_idx] = V[b,c_idx] * W[c,c_idx]
                    X = np.outer(V, W)  # (M, P) x (P, r) -- no, this isn't right
                    # Actually: for each (b, c), the coefficient vector is V[b,:] * W[c,:] (element-wise)
                    # So the system is: U[a,:] * (V[b,:] * W[c,:])^T = T[a,b,c]
                    # Stack: let Y be (M*P, r) where Y[b*P+c, c_idx] = V[b,c_idx] * W[c,c_idx]
                    Y = np.zeros((M*P, r))
                    for b in range(M):
                        for c in range(P):
                            Y[b*P+c, :] = V[b, :] * W[c, :]
                    rhs = T[a, :, :].flatten()
                    # Solve U[a,:] @ Y^T = rhs
                    # Y^T is (r, M*P)
                    # Use least squares
                    sol, _, _, _ = np.linalg.lstsq(Y.T, rhs, rcond=None)
                    U[a, :] = sol
                
                # Update V similarly
                for b in range(M):
                    Y = np.zeros((N*P, r))
                    for a in range(N):
                        for c in range(P):
                            Y[a*P+c, :] = U[a, :] * W[c, :]
                    rhs = T[:, b, :].flatten()
                    sol, _, _, _ = np.linalg.lstsq(Y.T, rhs, rcond=None)
                    V[b, :] = sol
                
                # Update W similarly
                for c in range(P):
                    Y = np.zeros((N*M, r))
                    for a in range(N):
                        for b in range(M):
                            Y[a*M+b, :] = U[a, :] * V[b, :]
                    rhs = T[:, :, c].flatten()
                    sol, _, _, _ = np.linalg.lstsq(Y.T, rhs, rcond=None)
                    W[c, :] = sol
                
                # Check convergence
                R = np.zeros((N, M, P), dtype=float)
                for c_idx in range(r):
                    R += np.einsum('a,b,c->abc', U[:, c_idx], V[:, c_idx], W[:, c_idx])
                err = np.linalg.norm(R - T)
                if err < 1e-8:
                    # Found a solution! Try to round to rationals
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-8):
                        return U_rounded, V_rounded, W_rounded, True
                    # Try rounding to integers
                    U_rounded = np.round(U)
                    V_rounded = np.round(V)
                    W_rounded = np.round(W)
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-8):
                        return U_rounded, V_rounded, W_rounded, True
                    # Return the floating point solution (may not be valid if entries aren't rational)
                    return U, V, W, False
        
        return None, None, None, False
    
    # Try ranks from 10 upward
    deadline = start + 120
    for r in range(10, 30):
        if time.time() > deadline - 10:
            break
        U_r, V_r, W_r, success = try_rank(r, max_iters=200)
        if success and verify(U_r, V_r, W_r, tol=1e-8):
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    # Convert to lists for return
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
