import numpy as np
import scipy.optimize as so
import time
import sys

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, idx_a, idx_b, idx_w, dim_A, dim_B, dim_C, atol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    hat_T = np.einsum('ar,br,wr->abw', U, V, W)
    vals = hat_T[idx_a, idx_b, idx_w]
    if not np.allclose(vals, 1.0, atol=atol):
        return False
    full_T = np.zeros((dim_A, dim_B, dim_C))
    full_T[idx_a, idx_b, idx_w] = 1.0
    zero_mask = (full_T == 0)
    zero_vals = hat_T[zero_mask]
    if not np.allclose(zero_vals, 0.0, atol=atol):
        return False
    return True

def find_2x2x3_decomposition(rank=10, time_limit=10.0):
    """
    Find a rank-decomposition for 2x2x3 matrix multiplication.
    A: 2x2 (4 entries), B: 2x3 (6 entries), C: 2x3 (6 entries).
    Returns U, V, W or None.
    """
    n, m, p = 2, 2, 3
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
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
    
    start_time = time.time()
    best_cost = float('inf')
    best_x = None
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        np.random.seed(attempt * 123 + 456)
        U0 = np.random.randn(dim_A, rank) * 0.5
        V0 = np.random.randn(dim_B, rank) * 0.5
        W0 = np.random.randn(dim_C, rank) * 0.5
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def residual(x):
            U = x[:dim_A*rank].reshape(dim_A, rank)
            V = x[dim_A*rank:dim_A*rank+dim_B*rank].reshape(dim_B, rank)
            W = x[dim_A*rank+dim_B*rank:].reshape(dim_C, rank)
            
            # Compute hat_T at sparse indices
            vals = np.zeros(len(idx_a))
            for r in range(rank):
                vals += U[:, r][idx_a] * V[:, r][idx_b] * W[:, r][idx_w]
            res_sparse = vals - 1.0
            
            # Compute hat_T at zero indices (sampled)
            # Total zeros = 4*6*6 - 12 = 132
            full_T = np.zeros((dim_A, dim_B, dim_C))
            full_T[idx_a, idx_b, idx_w] = 1.0
            zero_mask = (full_T == 0)
            zero_positions = np.argwhere(zero_mask)
            
            # Use all zeros as it's small
            zero_vals = np.zeros(len(zero_positions))
            for r in range(rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                for zp in zero_positions:
                    a, b, w = zp
                    zero_vals += u_col[a] * v_col[b] * w_col[w]
            
            return np.concatenate([res_sparse, 0.1 * zero_vals])
            
        try:
            res = so.least_squares(residual, x0, method='lm', xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=1000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_x = res.x
                if res.cost < 1e-10:
                    U_sol = best_x[:dim_A*rank].reshape(dim_A, rank)
                    V_sol = best_x[dim_A*rank:dim_A*rank+dim_B*rank].reshape(dim_B, rank)
                    W_sol = best_x[dim_A*rank+dim_B*rank:].reshape(dim_C, rank)
                    # Round to half-integers?
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    if _verify_decomposition(U_int, V_int, W_int, idx_a, idx_b, idx_w, dim_A, dim_B, dim_C):
                        return U_int, V_int, W_int
        except Exception:
            pass
            
    # If we found a good floating point solution, try to round it
    if best_x is not None and best_cost < 1e-6:
        U_sol = best_x[:dim_A*rank].reshape(dim_A, rank)
        V_sol = best_x[dim_A*rank:dim_A*rank+dim_B*rank].reshape(dim_B, rank)
        W_sol = best_x[dim_A*rank+dim_B*rank:].reshape(dim_C, rank)
        
        # Try rounding to integers or half-integers
        for mult in [1, 2, 4]:
            U_int = np.round(U_sol * mult) / mult
            V_int = np.round(V_sol * mult) / mult
            W_int = np.round(W_sol * mult) / mult
            if _verify_decomposition(U_int, V_int, W_int, idx_a, idx_b, idx_w, dim_A, dim_B, dim_C):
                return U_int, V_int, W_int
                
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
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
    
    start_time = time.time()
    deadline = 140.0
    
    # Fallback: Rank 42 using Strassen 2x2x2
    def get_fallback_rank42():
        U = np.zeros((dim_A, 42))
        V = np.zeros((dim_B, 42))
        W = np.zeros((dim_C, 42))
        
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1]),
        ]
        
        def add_block(U, V, W, a_indices, b_block_indices, w_indices, start_col, strassen_terms):
            for t_idx, (u, v, w) in enumerate(strassen_terms):
                col = start_col + t_idx
                for d in range(4):
                    if u[d] != 0: U[a_indices[d], col] += u[d]
                    if v[d] != 0: V[b_block_indices[d], col] += v[d]
                    if w[d] != 0: W[w_indices[d], col] += w[d]
                    
        a_left = [0, 1, 4, 5]
        a_right = [2, 3, 6, 7]
        
        b_top_0 = [0, 1, 6, 7]
        b_top_1 = [2, 3, 8, 9]
        b_top_2 = [4, 5, 10, 11]
        
        b_bot_0 = [12, 13, 18, 19]
        b_bot_1 = [14, 15, 20, 21]
        b_bot_2 = [16, 17, 22, 23]
        
        w_block_0 = [0, 1, 2, 3]
        w_block_1 = [4, 5, 6, 7]
        w_block_2 = [8, 9, 10, 11]
        
        add_block(U, V, W, a_left, b_top_0, w_block_0, 0, strassen_terms)
        add_block(U, V, W, a_left, b_top_1, w_block_1, 7, strassen_terms)
        add_block(U, V, W, a_left, b_top_2, w_block_2, 14, strassen_terms)
        
        add_block(U, V, W, a_right, b_bot_0, w_block_0, 21, strassen_terms)
        add_block(U, V, W, a_right, b_bot_1, w_block_1, 28, strassen_terms)
        add_block(U, V, W, a_right, b_bot_2, w_block_2, 35, strassen_terms)
        
        return U, V, W

    U_fb, V_fb, W_fb = get_fallback_rank42()
    if not _verify_decomposition(U_fb, V_fb, W_fb, idx_a, idx_b, idx_w, dim_A, dim_B, dim_C):
        # Schoolbook
        U48 = np.zeros((dim_A, 48))
        V48 = np.zeros((dim_B, 48))
        W48 = np.zeros((dim_C, 48))
        for idx, (a, b, w) in enumerate(indices):
            U48[a, idx] = 1
            V48[b, idx] = 1
            W48[w, idx] = 1
        U_fb, V_fb, W_fb = U48, V48, W48

    # Try to find rank 40 using 2x2x3 decomposition
    if time.time() < deadline - 20:
        dec_223 = find_2x2x3_decomposition(rank=10, time_limit=15.0)
        if dec_223 is not None:
            U_223, V_223, W_223 = dec_223
            # Construct rank 40 solution
            U_40 = np.zeros((dim_A, 40))
            V_40 = np.zeros((dim_B, 40))
            W_40 = np.zeros((dim_C, 40))
            
            # Blocks:
            # A1 (rows 0,1; cols 0,1) -> a indices 0,1,4,5
            # A2 (rows 0,1; cols 2,3) -> a indices 2,3,6,7
            # B1L (rows 0,1; cols 0,1,2) -> b indices 0..5
            # B1R (rows 0,1; cols 3,4,5) -> b indices 6..11
            # B2L (rows 2,3; cols 0,1,2) -> b indices 12..17
            # B2R (rows 2,3; cols 3,4,5) -> b indices 18..23
            # C1L (rows 0,1; cols 0,1,2) -> w indices 0..5
            # C1R (rows 0,1; cols 3,4,5) -> w indices 6..11
            # C2L (rows 0,1; cols 0,1,2) -> w indices 0..5 (same as C1L? No, C is 2x6. Rows 0,1 are the same for both A1 and A2 products?
            # C = A1 B1 + A2 B2.
            # A1 B1 contributes to C. A2 B2 contributes to C.
            # So C1L and C2L are the same output locations!
            # We must ADD the contributions.
            
            # Mapping for A1: a_local = [0,1,4,5]
            # Mapping for A2: a_local = [2,3,6,7]
            # Mapping for B1L: b_local = [0,1,2,3,4,5] (j=0,1; k=0,1,2)
            #   b = j*6+k.
            #   j=0: k=0,1,2 -> 0,1,2
            #   j=1: k=0,1,2 -> 6,7,8
            #   Wait, B is 4x6.
            #   B1 is rows 0,1.
            #   B1L is cols 0,1,2.
            #   Indices in full B:
            #   j=0, k=0 -> 0
            #   j=0, k=1 -> 1
            #   j=0, k=2 -> 2
            #   j=1, k=0 -> 6
            #   j=1, k=1 -> 7
            #   j=1, k=2 -> 8
            #   So b_local indices in V_223 (which are 0..5) map to full B indices [0,1,2,6,7,8].
            
            # Let's define the mappings explicitly.
            # U_223 is 4x10. V_223 is 6x10. W_223 is 6x10.
            
            # A1: a_full = [0, 1, 4, 5]
            # A2: a_full = [2, 3, 6, 7]
            
            # B1L: b_full = [0, 1, 2, 6, 7, 8]
            # B1R: b_full = [3, 4, 5, 9, 10, 11]
            # B2L: b_full = [12, 13, 14, 18, 19, 20]
            # B2R: b_full = [15, 16, 17, 21, 22, 23]
            
            # C1L: w_full = [0, 1, 2, 6, 7, 8] (k=0,1,2; i=0,1 -> w=k*2+i)
            #   k=0: i=0,1 -> 0,1
            #   k=1: i=0,1 -> 2,3
            #   k=2: i=0,1 -> 4,5
            #   Wait, C is 2x6. w = k*2+i.
            #   k=0: 0,1
            #   k=1: 2,3
            #   k=2: 4,5
            #   k=3: 6,7
            #   k=4: 8,9
            #   k=5: 10,11
            #   So C1L (k=0,1,2) -> w_full = [0,1, 2,3, 4,5]
            #   C1R (k=3,4,5) -> w_full = [6,7, 8,9, 10,11]
            
            # The W_223 corresponds to C of size 2x3.
            # w_local = k*2+i.
            # k=0: 0,1
            # k=1: 2,3
            # k=2: 4,5
            # So w_local indices 0..5 map to w_full for L blocks: [0,1, 2,3, 4,5]
            # And for R blocks: [6,7, 8,9, 10,11]
            
            # Add A1*B1L
            U_40[a_full_A1, 0:10] += U_223
            V_40[b_full_B1L, 0:10] += V_223
            W_40[w_full_C1L, 0:10] += W_223
            
            # Add A1*B1R
            U_40[a_full_A1, 10:20] += U_223
            V_40[b_full_B1R, 10:20] += V_223
            W_40[w_full_C1R, 10:20] += W_223
            
            # Add A2*B2L
            U_40[a_full_A2, 20:30] += U_223
            V_40[b_full_B2L, 20:30] += V_223
            W_40[w_full_C1L, 20:30] += W_223
            
            # Add A2*B2R
            U_40[a_full_A2, 30:40] += U_223
            V_40[b_full_B2R, 30:40] += V_223
            W_40[w_full_C1R, 30:40] += W_223
            
            if _verify_decomposition(U_40, V_40, W_40, idx_a, idx_b, idx_w, dim_A, dim_B, dim_C):
                return {"U": U_40.tolist(), "V": V_40.tolist(), "W": W_40.tolist()}

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
