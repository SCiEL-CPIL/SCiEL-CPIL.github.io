import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build target tensor T indices
    # T[a, b, w] = 1 if a=i*m+j, b=j*p+k, w=k*n+i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    
    # Precompute the full tensor for verification
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.float64)
    for a, b, w in indices:
        T[a, b, w] = 1.0
    
    start_time = time.time()
    deadline = 145.0
    
    def get_fallback():
        """Strassen-based rank-42 decomposition"""
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        
        # Block 1: k=0 (b cols 0-5), i=0 (w rows 0,3,6,9) -> w indices 0, 2, 1, 3
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        # Block 2: k=1 (b cols 6-11), i=0 (w rows 1,4,7,10) -> w indices 4, 6, 5, 7
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        # Block 3: k=2 (b cols 12-17), i=0 (w rows 2,5,8,11) -> w indices 8, 10, 9, 11
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W
    
    U_fb, V_fb, W_fb = get_fallback()
    
    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the tensor"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        
        full_hat = hat_T.copy()
        full_hat[idx_a, idx_b, idx_w] = 0
        max_err = np.max(np.abs(full_hat))
        return max_err < 1e-9
    
    # Verify fallback
    U_fb_arr = np.array(U_fb, dtype=np.float64)
    V_fb_arr = np.array(V_fb, dtype=np.float64)
    W_fb_arr = np.array(W_fb, dtype=np.float64)
    
    if not verify_solution(U_fb_arr, V_fb_arr, W_fb_arr):
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 42
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # WFTA-based approach using Z_2 x Z_3 structure
    # The key insight: M_{2,4,6} can be decomposed using the tensor product structure
    # 2 = 2, 4 = 2*2, 6 = 2*3
    # This gives a decomposition into blocks that can be handled more efficiently
    
    # Let's try a structured approach:
    # View A as 2x(2x2), B as (2x2)x(2x3), C as (2x3)x2
    # The tensor is M_{2, 2x2, 2x3}
    
    # Using the Kronecker product structure:
    # M_{2,4,6} = M_{2,2,2} ⊗ M_{1,2,3} (approximately, in tensor rank terms)
    # Rank(M_{2,2,2}) = 7 (Strassen)
    # Rank(M_{1,2,3}) = 6 (schoolbook) or possibly less
    
    # Actually, let's think more carefully.
    # A is 2x4 = 2x(2x2)
    # B is 4x6 = (2x2)x(2x3)  
    # C is 6x2 = (2x3)x2
    
    # The multiplication C = A*B can be written as:
    # C[i, (k1,k2)] = sum_j A[i,j] * B[j, (k1,k2)]
    # where j = (j1, j2) with j1 in {0,1}, j2 in {0,1}
    # and k = (k1, k2) with k1 in {0,1}, k2 in {0,1,2}
    
    # So C[i, (k1,k2)] = sum_{j1,j2} A[i, (j1,j2)] * B[(j1,j2), (k1,k2)]
    
    # This is a sum over j1,j2 of products of A[i,(j1,j2)] and B[(j1,j2),(k1,k2)]
    
    # We can group by j1:
    # C[i, (k1,k2)] = sum_{j1} [sum_{j2} A[i,(j1,j2)] * B[(j1,j2),(k1,k2)]]
    
    # For each j1, we have a 2x2 by 2x3 multiplication
    # Rank of M_{2,2,3} = 6 (schoolbook) or possibly 5?
    
    # Let's try to find a rank-5 decomposition for M_{2,2,3}
    # If we can, then total rank = 2 * 5 = 10, which is much better than 42!
    
    # Actually, let's reconsider. The structure is:
    # A has shape 2x4, which we can view as 2 x (2x2)
    # B has shape 4x6, which we can view as (2x2) x (2x3)
    # C has shape 6x2, which we can view as (2x3) x 2
    
    # The tensor T[a, b, w] where:
    # a = i * 4 + j, i in {0,1}, j in {0,1,2,3}
    # b = j * 6 + k, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # w = k * 2 + i, k in {0,1,2,3,4,5}, i in {0,1}
    
    # Let's reindex:
    # j = 2*j1 + j2, j1 in {0,1}, j2 in {0,1}
    # k = 3*k1 + k2, k1 in {0,1}, k2 in {0,1,2}
    
    # Then:
    # a = i * 4 + 2*j1 + j2
    # b = (2*j1 + j2) * 6 + 3*k1 + k2 = 12*j1 + 6*j2 + 3*k1 + k2
    # w = (3*k1 + k2) * 2 + i = 6*k1 + 2*k2 + i
    
    # So T[a, b, w] = 1 when:
    # a = 4*i + 2*j1 + j2
    # b = 12*j1 + 6*j2 + 3*k1 + k2
    # w = 6*k1 + 2*k2 + i
    
    # for i in {0,1}, j1 in {0,1}, j2 in {0,1}, k1 in {0,1}, k2 in {0,1,2}
    
    # This is a tensor with 2*2*2*2*3 = 48 non-zero entries
    
    # Let's try to find a low-rank decomposition by exploiting this structure
    
    # For a fixed (j1, j2), we have a sub-tensor:
    # T_sub[i, k1, k2] = 1 for all i, k1, k2
    # This is just the all-ones tensor of shape 2x2x3, which has rank 1!
    
    # Wait, that's not right. Let me reconsider.
    
    # For fixed (j1, j2):
    # a = 4*i + 2*j1 + j2 (varies with i)
    # b = 12*j1 + 6*j2 + 3*k1 + k2 (varies with k1, k2)
    # w = 6*k1 + 2*k2 + i (varies with k1, k2, i)
    
    # So for fixed (j1, j2), we have a tensor T_{j1,j2}[i, (k1,k2), i'] where:
    # T_{j1,j2}[i, (k1,k2), i'] = 1 if i' = i (from w = 6*k1 + 2*k2 + i)
    
    # Wait, w = 6*k1 + 2*k2 + i, and w also determines i' via w = 6*k1 + 2*k2 + i'
    # So i' = i. This means for fixed (j1, j2), the tensor is:
    # T_{j1,j2}[i, (k1,k2), i'] = 1 if i = i'
    
    # This is a diagonal tensor in the i dimension, with all ones in the (k1,k2) dimension.
    # Rank of this tensor = 2 (one for each i value)
    
    # So total rank = 4 (for j1,j2 values) * 2 = 8?
    
    # Let's verify this. For fixed (j1, j2), the sub-tensor has:
    # - i in {0,1}
    # - (k1,k2) in {0,1} x {0,1,2} (6 values)
    # - i' in {0,1}
    
    # T_{j1,j2}[i, (k1,k2), i'] = 1 if i = i'
    
    # This can be written as:
    # T_{j1,j2} = sum_{i=0}^{1} e_i ⊗ J ⊗ e_i
    # where J is the all-ones vector of length 6
    
    # So rank = 2 for each (j1,j2), total rank = 4 * 2 = 8
    
    # But wait, this can't be right because the lower bound is max(8, 24, 12) = 24
    
    # Let me re-examine. The issue is that b depends on both j and k, not just k.
    
    # Let me try a different approach: direct optimization with the structured initialization
    
    def try_rank_optimization(r, num_restarts=3):
        """Try to find a rank-r decomposition using structured initialization"""
        for attempt in range(num_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + attempt)
            
            # Structured initialization based on the tensor's algebraic structure
            U = np.zeros((dim_A, r))
            V = np.zeros((dim_B, r))
            W = np.zeros((dim_C, r))
            
            # Initialize with random sparse vectors
            for t in range(r):
                idx_u = np.random.choice(dim_A, size=3, replace=False)
                idx_v = np.random.choice(dim_B, size=3, replace=False)
                idx_w = np.random.choice(dim_C, size=3, replace=False)
                
                U[idx_u, t] = np.random.randn(3)
                V[idx_v, t] = np.random.randn(3)
                W[idx_w, t] = np.random.randn(3)
            
            # ALS Iterations
            T_mat_u = T.reshape(dim_A, dim_B * dim_C)
            T_mat_v = T.reshape(dim_B, dim_A * dim_C)
            T_mat_w = T.reshape(dim_C, dim_A * dim_B)
            
            for it in range(50):
                if time.time() > deadline:
                    return None
                
                # Solve for U
                N = np.zeros((dim_B * dim_C, r))
                for b in range(dim_B):
                    for w in range(dim_C):
                        N[b * dim_C + w, :] = V[b, :] * W[w, :]
                
                try:
                    U, _, _, _ = np.linalg.lstsq(N.T, T_mat_u.T, rcond=None)
                    U = U.T
                except:
                    pass
                
                # Solve for V
                N_v = np.zeros((dim_A * dim_C, r))
                for a in range(dim_A):
                    for w in range(dim_C):
                        N_v[a * dim_C + w, :] = U[a, :] * W[w, :]
                
                try:
                    V, _, _, _ = np.linalg.lstsq(N_v.T, T_mat_v.T, rcond=None)
                    V = V.T
                except:
                    pass
                
                # Solve for W
                N_w = np.zeros((dim_A * dim_B, r))
                for a in range(dim_A):
                    for b in range(dim_B):
                        N_w[a * dim_B + b, :] = U[a, :] * V[b, :]
                
                try:
                    W, _, _, _ = np.linalg.lstsq(N_w.T, T_mat_w.T, rcond=None)
                    W = W.T
                except:
                    pass
                
                # Early exit if error is small
                hat_T = np.einsum('ar,br,wr->abw', U, V, W)
                err = np.sum((hat_T - T)**2)
                if err < 1e-10:
                    break
            
            # Final Check
            hat_T = np.einsum('ar,br,wr->abw', U, V, W)
            err = np.sum((hat_T - T)**2)
            
            if err < 1e-8:
                # Try to round
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                if verify_solution(U_int, V_int, W_int):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                U_int2 = np.round(U)
                V_int2 = np.round(V)
                W_int2 = np.round(W)
                if verify_solution(U_int2, V_int2, W_int2):
                    return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
        
        return None
    
    # Try ranks from 24 to 41 with limited time
    for r in range(24, 42):
        if time.time() > deadline - 5:
            break
        
        res = try_rank_optimization(r, num_restarts=2)
        if res:
            best_rank = r
            best_result = res
            break
    
    return best_result
