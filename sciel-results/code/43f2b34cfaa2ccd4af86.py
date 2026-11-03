import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer for 120s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 45)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    # Target Tensor T of shape (n, m, p)
    # T[i, j, k] = 1 for all i, j, k
    # Wait, the tensor definition for matrix multiplication C = A * B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) and value 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Since every A[i,j] and B[j,k] pair contributes to exactly one C[i,k] (specifically C[i,k]),
    # the tensor T is simply all ones.
    T = np.ones((n, m, p), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist(),
        "rank": rank_schoolbook
    }
    
    # 2. Attempt to find lower rank decompositions
    # We try ranks from 44 downwards.
    # We use a randomized local search with gradient descent.
    
    def verify_decomp(U, V, W, T):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reconstruct tensor
        # T_recon[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        r = U.shape[1]
        # Efficient computation:
        # For each c, U[:,c] is vector of len n*m, V[:,c] len m*p, W[:,c] len p*n.
        # This is a sum of rank-1 tensors.
        T_recon = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            u_vec = U[:, c].reshape((n, m))
            v_vec = V[:, c].reshape((m, p))
            w_vec = W[:, c].reshape((p, n))
            # The term is u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            # This is an outer product contraction.
            # T_c[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # We can compute this by iterating or using einsum
            # np.einsum('ij,jk,ki->ijk', u_vec, v_vec, w_vec)
            T_recon += np.einsum('ij,jk,ki->ijk', u_vec, v_vec, w_vec)
            
        # Check if T_recon matches T
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        # Check if entries are integer or half-integer
        # Multiply by 2, round, check if close to integer
        U_int = np.round(U * 2)
        V_int = np.round(V * 2)
        W_int = np.round(W * 2)
        if not np.allclose(U * 2, U_int, atol=1e-6): return False
        if not np.allclose(V * 2, V_int, atol=1e-6): return False
        if not np.allclose(W * 2, W_int, atol=1e-6): return False
        
        # Convert to list of lists of floats (integers or half-integers)
        U_out = (U_int / 2.0).tolist()
        V_out = (V_int / 2.0).tolist()
        W_out = (W_int / 2.0).tolist()
        return True, U_out, V_out, W_out

    # Search loop
    # We try a few ranks. 44 is the most likely to be found if it exists.
    # 43, 42...
    
    # To improve success rate, we can try multiple initializations for each rank.
    # However, time is limited.
    
    for r in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Try 5 random initializations for this rank
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            # Scale is important. Schoolbook uses 1s.
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*r].reshape((n*m, r))
                V = x[n*m*r : n*m*r + m*p*r].reshape((m*p, r))
                W = x[n*m*r + m*p*r :].reshape((p*n, r))
                
                T_recon = np.zeros((n, m, p), dtype=float)
                for c in range(r):
                    u_vec = U[:, c].reshape((n, m))
                    v_vec = V[:, c].reshape((m, p))
                    w_vec = W[:, c].reshape((p, n))
                    T_recon += np.einsum('ij,jk,ki->ijk', u_vec, v_vec, w_vec)
                
                return np.linalg.norm(T - T_recon)
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='BFGS', options={'maxiter': 50, 'gtol': 1e-8})
                if res.success or res.fun < 1e-6:
                    x_opt = res.x
                    U_opt = x_opt[:n*m*r].reshape((n*m, r))
                    V_opt = x_opt[n*m*r : n*m*r + m*p*r].reshape((m*p, r))
                    W_opt = x_opt[n*m*r + m*p*r :].reshape((p*n, r))
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2.0
                    V_snap = np.round(V_opt * 2) / 2.0
                    W_snap = np.round(W_opt * 2) / 2.0
                    
                    # Verify
                    res_ver = verify_decomp(U_snap, V_snap, W_snap, T)
                    if res_ver[0]:
                        return {
                            "U": res_ver[1],
                            "V": res_ver[2],
                            "W": res_ver[3]
                        }
            except Exception as e:
                pass
                
    return best_result
