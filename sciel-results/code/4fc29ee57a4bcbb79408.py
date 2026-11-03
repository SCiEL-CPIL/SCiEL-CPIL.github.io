import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    # Target: 2x3x2 block
    # U shape: (2*3, R) = (6, R)
    # V shape: (3*2, R) = (6, R)
    # W shape: (2*2, R) = (4, R)
    
    dim_u = n * m # 6
    dim_v = m * 2 # 6
    dim_w = 2 * n # 4
    
    # Construct target tensor T[i, j, k] = 1 if valid index
    # i in 0..1, j in 0..2, k in 0..1
    T_block = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(2):
                alpha = i * m + j
                beta = j * 2 + k
                gamma = k * n + i
                T_block[alpha, beta, gamma] = 1.0

    best_rank = None
    best_U_block = None
    best_V_block = None
    best_W_block = None
    
    # Try ranks from 7 up to 12
    for R in range(7, 13):
        if time.time() > deadline - 30:
            break
            
        found = False
        # Multiple random restarts for each rank
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + attempt * 13 + R)
            # Initialize with small random values
            U_b = np.random.randn(dim_u, R) * 0.1
            V_b = np.random.randn(dim_v, R) * 0.1
            W_b = np.random.randn(dim_w, R) * 0.1
            
            # ALS Optimization
            max_iter = 200
            tol = 1e-12
            prev_err = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 30:
                    break
                
                # Update U
                # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # For fixed i, minimize || U[i, :] M - T[i, :, :] ||_F
                # where M[j, k, r] = V[j, r] W[k, r]
                # Reshape M to (dim_v * dim_w, R)
                # M_flat = (dim_v * dim_w, R)
                # M_flat[j * dim_w + k, r] = V[j, r] * W[k, r]
                
                # Vectorized construction of M_flat
                # V_b is (dim_v, R), W_b is (dim_w, R)
                # We want outer product of V and W along the R dimension? No.
                # M_flat[j*dim_w+k, r] = V_b[j, r] * W_b[k, r]
                # This is element-wise product of V_b (broadcast to dim_w) and W_b (broadcast to dim_v)
                # Let's use einsum: 'jr,kr->jk r' -> reshape to (j*k, r)
                M_u = np.einsum('jr,kr->jkr', V_b, W_b).reshape(dim_v * dim_w, R)
                
                # Solve for each i: U_b[i, :] = argmin || U_b[i, :] M_u^T - T_block[i, :, :].flatten() ||
                # This is a linear least squares problem: M_u.T @ U_b[i, :] = T_block[i, :, :].flatten()
                # Actually, we want U_b[i, :] such that U_b[i, :] @ M_u.T = T_flat
                # So M_u.T @ U_b[i, :].T = T_flat.T
                # Or (M_u.T @ M_u) @ U_b[i, :].T = M_u.T @ T_flat.T
                # Since R is small, we can just use lstsq for each i.
                
                T_flat_all = T_block.reshape(n, dim_v * dim_w)
                # Solve for all i at once?
                # M_u.T is (R, dim_v*dim_w)
                # T_flat_all is (n, dim_v*dim_w)
                # We want U_b (n, R) such that U_b @ M_u.T = T_flat_all
                # This is equivalent to M_u @ U_b.T = T_flat_all.T
                # Let's use lstsq on the whole system?
                # A = M_u (dim_v*dim_w, R)
                # B = T_flat_all.T (dim_v*dim_w, n)
                # We want X (R, n) such that A @ X = B
                # Then U_b = X.T
                
                try:
                    X, residuals, rank, sv = np.linalg.lstsq(M_u, T_flat_all.T, rcond=None)
                    U_b = X.T
                except:
                    pass
                
                # Update V
                # M_v[i, k, r] = U[i, r] W[k, r]
                M_v = np.einsum('ir,kr->ikr', U_b, W_b).reshape(dim_u * dim_w, R)
                # T_block[j, i, k] = 1. We need T_block reshaped for j.
                # T_block is (i, j, k). We want to solve for V_b[j, :]
                # Equation: V_b[j, :] @ M_v.T = T_block[:, j, :].flatten()
                # M_v.T is (R, dim_u*dim_w)
                # T_block[:, j, :].flatten() is (dim_u*dim_w,)
                # Let's group by j.
                # T_flat_for_v = T_block.transpose(1, 0, 2).reshape(m, dim_u * dim_w)
                T_flat_v = T_block.transpose(1, 0, 2).reshape(m, dim_u * dim_w)
                # A = M_v (dim_u*dim_w, R)
                # B = T_flat_v.T (dim_u*dim_w, m)
                # X = lstsq(A, B) -> (R, m)
                # V_b = X.T
                
                try:
                    X, residuals, rank, sv = np.linalg.lstsq(M_v, T_flat_v.T, rcond=None)
                    V_b = X.T
                except:
                    pass
                
                # Update W
                # M_w[i, j, r] = U[i, r] V[j, r]
                M_w = np.einsum('ir,jr->ijr', U_b, V_b).reshape(dim_u * dim_v, R)
                # T_block[i, j, k] = 1. Solve for W_b[k, :]
                # Equation: W_b[k, :] @ M_w.T = T_block[:, :, k].flatten()
                # T_flat_w = T_block.transpose(2, 0, 1).reshape(2, dim_u * dim_v)
                T_flat_w = T_block.transpose(2, 0, 1).reshape(2, dim_u * dim_v)
                # A = M_w (dim_u*dim_v, R)
                # B = T_flat_w.T (dim_u*dim_v, 2)
                # X = lstsq(A, B) -> (R, 2)
                # W_b = X.T
                
                try:
                    X, residuals, rank, sv = np.linalg.lstsq(M_w, T_flat_w.T, rcond=None)
                    W_b = X.T
                except:
                    pass
                
                # Calculate error
                T_rec = np.einsum('ir,jr,kr->ijk', U_b, V_b, W_b)
                err = np.linalg.norm(T_rec - T_block)
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
                
            if prev_err < 1e-9:
                best_rank = R
                best_U_block = U_b
                best_V_block = V_b
                best_W_block = W_b
                found = True
                break # Found minimal rank for this R, break to next R? No, we want lowest R.
        
        if found:
            break
    
    # Construct the full decomposition
    if best_rank is not None:
        R_full = best_rank * 5
        U_full = np.zeros((n * m, R_full))
        V_full = np.zeros((m * p, R_full))
        W_full = np.zeros((p * n, R_full))
        
        for b in range(5):
            U_full[:, b*best_rank:(b+1)*best_rank] = best_U_block
            # V_full: rows j*p + k. For block b, k is 2b, 2b+1.
            for j in range(m):
                for k in range(2):
                    V_full[j * p + 2 * b + k, b * best_rank : (b+1) * best_rank] = best_V_block[j, :]
            
            # W_full: rows k*n + i. For block b, k is 2b, 2b+1.
            for k in range(2):
                for i in range(n):
                    W_full[(2 * b + k) * n + i, b * best_rank : (b+1) * best_rank] = best_W_block[k, :]
        
        # Verify full decomposition
        T_full = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_rec_full = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
        
        if np.allclose(T_rec_full, T_full, atol=1e-9):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback: Return the original rank-55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1
        
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
