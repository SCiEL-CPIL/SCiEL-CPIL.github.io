import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target: Rank 48 solution using 4 blocks of Rank 12 (2x4x2)
    # Fallback: Rank 56 solution using 8 blocks of Rank 7 (2x2x2 Strassen)
    
    def get_strassen_56():
        rank = 56
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        strassen_terms = [
            {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
            {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
            {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
            {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
            {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
            {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
            {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
        ]
        
        col = 0
        for a in range(2):
            for b in range(4):
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1
        return U, V, W, rank

    def get_rank_12_block():
        """
        Finds a rank-12 decomposition for 2x4 x 4x2.
        Returns U (8x12), V (8x12), W (4x12) or None if failed.
        """
        # Target tensor T for 2x4x2
        # A: 2x4 (8), B: 4x2 (8), C: 2x2 (4)
        # T[a, b, c] = 1 if a=(i, ja), b=(jb, k), c=(k, i) and ja==jb
        T = np.zeros((8, 8, 4))
        for i in range(2):
            for ja in range(4):
                a = i * 4 + ja
                for k in range(2):
                    b = ja * 2 + k
                    c = k * 2 + i
                    T[a, b, c] = 1.0
        
        rank = 12
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(8, rank) * 0.1
        V = np.random.randn(8, rank) * 0.1
        W = np.random.randn(4, rank) * 0.1
        
        def residuals(params):
            U_r = params[:8*rank].reshape(8, rank)
            V_r = params[8*rank:16*rank].reshape(8, rank)
            W_r = params[16*rank:20*rank].reshape(4, rank)
            
            # Compute tensor approximation
            # Res = sum_r U_r[:, r] outer V_r[:, r] outer W_r[:, r]
            # We need to match T.
            # Flatten T and Res.
            # Res[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is a 3rd order tensor.
            # We can compute the difference for each (a,b,c).
            # To be efficient, we can compute for all a,b and check c?
            # Or just compute the full tensor. 8x8x4 is small.
            Res = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            return (Res - T).flatten()
        
        try:
            # Use least squares
            # This might be slow or get stuck.
            # Given the small size, it's worth a try.
            # We have 240 residuals, 240 parameters.
            # It's a square system.
            
            # To avoid local minima, try a few restarts if time permits?
            # For now, one attempt.
            res = least_squares(residuals, params, method='lm', max_nfev=1000)
            
            if res.cost < 1e-8:
                U_opt = res.x[:8*rank].reshape(8, rank)
                V_opt = res.x[8*rank:16*rank].reshape(8, rank)
                W_opt = res.x[16*rank:20*rank].reshape(4, rank)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify
                Res_check = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                if np.allclose(Res_check, T, atol=1e-9):
                    return U_round, V_round, W_round
        except Exception:
            pass
        return None

    # Try to get the rank 48 solution
    U_block, V_block, W_block = get_rank_12_block()
    
    if U_block is not None:
        rank = 48
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        # 4 blocks for c in 0..3
        # A is shared (same U for all blocks? No, U is part of the tensor decomposition.
        # In the full tensor, U has 8 rows.
        # The block solution has U (8x12).
        # We just copy the same U_block for each of the 4 column blocks?
        # Wait. The tensor is T_total = sum_c T_c.
        # T_c uses A (full), B_c (part of B), C_c (part of C).
        # The rank-1 terms for T_c are u_c,r outer v_c,r outer w_c,r.
        # u_c,r is a vector in A-space (dim 8).
        # v_c,r is a vector in B-space (dim 32). It has non-zeros only in B_c.
        # w_c,r is a vector in C-space (dim 16). It has non-zeros only in C_c.
        
        # So for each c, we take the U_block (8x12) and place it in U[:, c*12:(c+1)*12].
        # For V, we need to map the local B indices (0..7) to global B indices.
        # Local B for block c: rows 0..3, cols 2c, 2c+1.
        # Global B index: j*8 + k.
        # Local B index: j*2 + k_local.
        # k = 2c + k_local.
        # So V_global[j*8 + 2c + k_local, c*12 + r] = V_block[j*2 + k_local, r].
        
        # For W, local C indices (0..3) map to global C indices.
        # Local C index: k_local*2 + i.
        # Global C index: k*2 + i = (2c + k_local)*2 + i = 4c + 2*k_local + i.
        # So W_global[4c + 2*k_local + i, c*12 + r] = W_block[2*k_local + i, r].
        
        for c in range(4):
            col_offset = c * 12
            
            # U
            U[:, col_offset:col_offset+12] = U_block
            
            # V
            for j in range(4):
                for k_local in range(2):
                    local_v_idx = j * 2 + k_local
                    global_v_idx = j * 8 + (2*c + k_local)
                    V[global_v_idx, col_offset:col_offset+12] = V_block[local_v_idx, :]
            
            # W
            for k_local in range(2):
                for i in range(2):
                    local_w_idx = k_local * 2 + i
                    global_w_idx = (2*c + k_local) * 2 + i
                    W[global_w_idx, col_offset:col_offset+12] = W_block[local_w_idx, :]
        
        # Verify full solution
        def verify(U, V, W):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if not np.isclose(val, 1.0, atol=1e-9):
                            return False
            return True

        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback
            U, V, W, rank = get_strassen_56()
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = get_strassen_56()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

construct()
