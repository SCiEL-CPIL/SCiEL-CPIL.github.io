import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    n, m, p = 2, 3, 9
    
    # Build the rank-50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0
    
    # Flatten T to vector for verification
    T_flat = T.flatten()
    
    def verify_decomposition(U_new, V_new, W_new, T_flat):
        """Verify decomposition using efficient matrix operations"""
        # For each rank-1 term r, compute outer product
        # and check if sum equals T
        residual = 0.0
        for r in range(U_new.shape[1]):
            # Compute the rank-1 tensor contribution
            # T_r[i,j,k] = U_new[i,r] * V_new[j,r] * W_new[k,r]
            # This is equivalent to outer product of U[:,r], V[:,r], W[:,r]
            term = np.einsum('ir,jr,kr->ijk', U_new[:, r], V_new[:, r], W_new[:, r])
            residual += np.sum((term - T_flat.reshape(n*m, m*p, p*n))**2)
        return residual
    
    # Check for linear dependencies using SVD
    try:
        # Build the coefficient matrix more efficiently
        # For each rank term r, the coefficient vector is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We can build this as a 2916 x rank matrix
        n_entries = n * m * m * p * p * n
        mat = np.zeros((n_entries, rank))
        
        # More efficient construction
        for r in range(rank):
            # Kronecker product of the three vectors
            u_r = U_np[:, r]
            v_r = V_np[:, r]
            w_r = W_np[:, r]
            # Build the full vector
            idx = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        mat[idx, r] = u_r[i] * v_r[j] * w_r[k]
                        idx += 1
        
        # Compute SVD
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        
        # Find the numerical rank
        tol = s[0] * rank * 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # Try to find the best column to drop
            # Use multiple null space vectors
            best_residual = float('inf')
            best_drop_col = None
            
            null_dim = rank - num_rank
            for null_idx in range(null_dim):
                null_vec = Vt[-(null_idx+1)]
                # Try dropping each column with significant weight in null space
                abs_null = np.abs(null_vec)
                # Only consider columns with large weight
                threshold = np.max(abs_null) * 0.1
                candidates = np.where(abs_null > threshold)[0]
                
                for drop_col in candidates:
                    if time.time() > deadline:
                        break
                    
                    # Remove that column
                    mask = np.ones(rank, dtype=bool)
                    mask[drop_col] = False
                    U_new = U_np[:, mask]
                    V_new = V_np[:, mask]
                    W_new = W_np[:, mask]
                    
                    # Verify
                    residual = verify_decomposition(U_new, V_new, W_new, T_flat)
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_drop_col = drop_col
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                    
                    if residual < 1e-8:
                        break
                
                if best_residual < 1e-8:
                    break
            
            if best_residual < 1e-8 and best_drop_col is not None:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    except Exception as e:
        pass
    
    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
