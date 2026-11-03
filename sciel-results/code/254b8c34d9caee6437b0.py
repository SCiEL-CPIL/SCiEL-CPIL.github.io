"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # We suspect the rank can be reduced by exploiting the block structure.
    # Split p=10 into 5 blocks of size 2.
    # Each block is a 2x3 * 3x2 multiplication.
    # We search for a low-rank decomposition of the 2x3x2 tensor.
    
    # Target tensor for 2x3x2 multiplication
    # Indices: i in 0..1 (rows of A), j in 0..2 (cols of A / rows of B), k in 0..1 (cols of B)
    # U shape: (2*3, R) = (6, R)
    # V shape: (3*2, R) = (6, R)
    # W shape: (2*2, R) = (4, R)
    
    dim_u_block = n * 2  # 4? No, A is 2x3. U corresponds to A entries.
    # Wait, the standard mapping is:
    # U index: i*m + j. For 2x3, i in 0..1, j in 0..2. Total 6.
    # V index: j*p_block + k. For 3x2, j in 0..2, k in 0..1. Total 6.
    # W index: k*n_block + i. For 2x2, k in 0..1, i in 0..1. Total 4.
    
    dim_u = n * m # 6
    dim_v_block = m * 2 # 6
    dim_w_block = 2 * n # 4
    
    T_block = np.zeros((dim_u, dim_v_block, dim_w_block))
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
    
    # Try ranks from 8 up to 11 (11 is naive 2*3*2/2? No, 2*3*2=12. Strassen+1 is 8? No.
    # Naive is 12.
    # Let's try 8, 9, 10, 11, 12.
    
    for R in range(8, 13):
        if time.time() > deadline - 20:
            break
            
        # Initialize with random or structured values
        # Use a structured initialization based on Strassen for the 2x2x2 part and naive for the rest?
        # Or just random.
        
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(42 + attempt * 100 + R)
            U_b = np.random.randn(dim_u, R) * 0.1
            V_b = np.random.randn(dim_v_block, R) * 0.1
            W_b = np.random.randn(dim_w_block, R) * 0.1
            
            # ALS Optimization
            max_iter = 500
            tol = 1e-12
            prev_err = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                # T_block[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # For fixed i, we want to minimize sum_{j,k} (T[i,j,k] - sum_r U[i,r] V[j,r] W[k,r])^2
                # This is a linear system in U[i, :]
                # Let M_jk_r = V[j,r] W[k,r]
                # We want U[i, :] such that U[i, :] dot M_jk ≈ T[i,j,k]
                # Reshape M to (R, dim_v*dim_w) and T[i] to (dim_v*dim_w)
                # U[i, :] = argmin || U[i, :] M^T - T[i, :] ||
                
                # Build the matrix A_u of shape (dim_v * dim_w, R)
                # A_u[j*dim_w + k, r] = V_b[j, r] * W_b[k, r]
                A_u = np.zeros((dim_v_block * dim_w_block, R))
                for j in range(m):
                    for k in range(2):
                        for r in range(R):
                            A_u[j * 2 + k, r] = V_b[j, r] * W_b[k, r]
                
                # Solve for each i
                for i in range(n):
                    b_u = T_block[i, :, :].flatten()
                    # Solve A_u.T @ U[i, :] = b_u
                    try:
                        U_b[i, :], res, rank, sv = np.linalg.lstsq(A_u.T, b_u, rcond=None)
                    except:
                        pass
                
                # Update V
                A_v = np.zeros((dim_u * dim_w_block, R))
                for i in range(n):
                    for k in range(2):
                        for r in range(R):
                            A_v[i * 2 + k, r] = U_b[i, r] * W_b[k, r]
                
                for j in range(m):
                    b_v = np.zeros(dim_u * dim_w_block)
                    for i in range(n):
                        for k in range(2):
                            b_v[i * 2 + k] = T_block[i, j, k]
                    try:
                        V_b[j, :], res, rank, sv = np.linalg.lstsq(A_v.T, b_v, rcond=None)
                    except:
                        pass

                # Update W
                A_w = np.zeros((dim_u * dim_v_block, R))
                for i in range(n):
                    for j in range(m):
                        for r in range(R):
                            A_w[i * m + j, r] = U_b[i, r] * V_b[j, r]
                
                for k in range(2):
                    b_w = np.zeros(dim_u * dim_v_block)
                    for i in range(n):
                        for j in range(m):
                            b_w[i * m + j] = T_block[i, j, k]
                    try:
                        W_b[k, :], res, rank, sv = np.linalg.lstsq(A_w.T, b_w, rcond=None)
                    except:
                        pass
                
                # Calculate error
                T_rec = np.zeros((dim_u, dim_v_block, dim_w_block))
                for r in range(R):
                    T_rec += np.einsum('i,j,k->ijk', U_b[:, r], V_b[:, r], W_b[:, r])
                err = np.linalg.norm(T_rec - T_block)
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
                
            if prev_err < 1e-9:
                best_rank = R
                best_U_block = U_b
                best_V_block = V_b
                best_W_block = W_b
                break # Found minimal rank for this R, break to next R? No, we want lowest R.
                # Since we iterate R from 8 upwards, the first success is the best.
        
        if best_rank is not None:
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
            # V_block has rows for j=0..2, k=0..1.
            # We need to map V_block[j, r] to V_full[j*p + 2b + k, b*R + r]
            for j in range(m):
                for k in range(2):
                    V_full[j * p + 2 * b + k, b * best_rank : (b+1) * best_rank] = best_V_block[j, :]
            
            # W_full: rows k*n + i. For block b, k is 2b, 2b+1.
            # W_block has rows for k=0..1, i=0..1.
            # We need to map W_block[k, r] to W_full[(2b+k)*n + i, b*R + r]
            for k in range(2):
                for i in range(n):
                    W_full[(2 * b + k) * n + i, b * best_rank : (b+1) * best_rank] = best_W_block[k, :]
        
        # Verify full decomposition
        T_full = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_rec_full = np.zeros((n * m, m * p, p * n))
        for r in range(R_full):
            T_rec_full += np.einsum('i,j,k->ijk', U_full[:, r], V_full[:, r], W_full[:, r])
        
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
