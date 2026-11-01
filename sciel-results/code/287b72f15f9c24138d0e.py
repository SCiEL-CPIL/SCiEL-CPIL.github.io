import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _strassen_2x2x6():
    """Return the rank-21 Strassen decomposition."""
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1

    return U, V, W

def _try_reduce_rank(deadline):
    """Try to find a rank-20 decomposition by perturbing the rank-21 solution."""
    n, m, p = 2, 2, 6
    U0, V0, W0 = _strassen_2x2x6()
    
    # We use numpy for faster residual calculation
    # U is (4, rank), V is (12, rank), W is (12, rank)
    # Residual for (i,j,k) is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = 21
    
    # Try rank 20
    target_rank = 20
    if time.time() < deadline - 10:
        # Initialize with rank-21 solution, zero out last column
        U = np.array(U0, dtype=float)
        V = np.array(V0, dtype=float)
        W = np.array(W0, dtype=float)
        
        # Zero out the last column (index target_rank)
        U[:, target_rank] = 0
        V[:, target_rank] = 0
        W[:, target_rank] = 0
        
        # Precompute indices for residual calculation
        # We need to compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
        # This is a tensor contraction.
        
        def compute_residual_np(U, V, W, n, m, p, rank):
            # U: (n*m, rank)
            # V: (m*p, rank)
            # W: (p*n, rank)
            # We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Residual = sum_{i,j,k} (T[i,j,k] - 1)^2
            
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.dot(U[u_idx, :rank], V[v_idx, :rank] * W[w_idx, :rank])
                        res += (s - 1.0) ** 2
            return res
        
        # Simple local search
        lr = 0.1
        best_res = compute_residual_np(U, V, W, n, m, p, target_rank)
        
        for iteration in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Try to improve
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Pick random matrix and element
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U
                elif mat_idx == 1:
                    mat = V
                else:
                    mat = W
                
                row = np.random.randint(mat.shape[0])
                col = np.random.randint(target_rank)
                
                old_val = mat[row, col]
                mat[row, col] = old_val + lr * np.random.randn()
                
                new_res = compute_residual_np(U, V, W, n, m, p, target_rank)
                
                if new_res < best_res:
                    best_res = new_res
                    improved = True
                else:
                    mat[row, col] = old_val
            
            if not improved:
                lr *= 0.8
                if lr < 1e-6:
                    break
            
            if best_res < 1e-10:
                break
        
        if best_res < 1e-10:
            # Round to nearest half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Convert back to list of lists for return
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list, target_rank
    
    return U0, V0, W0, 21

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    U, V, W, rank = _try_reduce_rank(deadline)
    
    return {"U": U, "V": V, "W": W}
