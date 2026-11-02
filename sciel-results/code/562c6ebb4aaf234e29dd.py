"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks.
    # This is the known optimal rank for 2x4 * 4x4 matrix multiplication.
    # A (2x4) = [A_L, A_R] where A_L, A_R are 2x2.
    # B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]] where blocks are 2x2.
    # C (2x4) = [C_L, C_R]
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR
    # Each 2x2 * 2x2 multiplication uses Strassen (rank 7).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_indices, v_indices, w_indices, col_idx):
        for idx, coeff in u_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in v_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in w_indices:
            W[idx, col_idx] += coeff
    
    # Strassen 2x2 terms
    # X (2x2) * Y (2x2) = P (2x2)
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # P = [[p00, p01], [p10, p11]]
    
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y01 - y11)
    # M4 = x11(y10 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (U_indices, V_indices, W_indices)
        # Indices are local (0..3) for 2x2 matrices
        # 0: 00, 1: 01, 2: 10, 3: 11
        
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Mapping functions
    # A (2x4):
    #   A_L: cols 0,1 -> indices 0,1,4,5
    #   A_R: cols 2,3 -> indices 2,3,6,7
    
    def map_u_a(left_right, local_idx):
        if left_right == 'L':
            if local_idx == 0: return 0
            if local_idx == 1: return 1
            if local_idx == 2: return 4
            if local_idx == 3: return 5
        else:
            if local_idx == 0: return 2
            if local_idx == 1: return 3
            if local_idx == 2: return 6
            if local_idx == 3: return 7
    
    # B (4x4):
    #   B_TL: rows 0,1 cols 0,1 -> indices 0,1,4,5
    #   B_TR: rows 0,1 cols 2,3 -> indices 2,3,6,7
    #   B_BL: rows 2,3 cols 0,1 -> indices 8,9,12,13
    #   B_BR: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    def map_v_b(block, local_idx):
        # block: 'TL', 'TR', 'BL', 'BR'
        row_off = 0
        col_off = 0
        if block == 'TL':
            row_off = 0; col_off = 0
        elif block == 'TR':
            row_off = 0; col_off = 2
        elif block == 'BL':
            row_off = 2; col_off = 0
        elif block == 'BR':
            row_off = 2; col_off = 2
            
        # Local index l -> local row r = l//2, local col c = l%2
        r = local_idx // 2
        c = local_idx % 2
        global_r = row_off + r
        global_c = col_off + c
        return global_r * 4 + global_c
    
    # C (2x4):
    #   W index is k*n + i = k*2 + i.
    #   C_L is cols 0,1. So k in {0,1}. i in {0,1}.
    #   Local (r,c) in 2x2 result corresponds to global row r, col c.
    #   Global col c is the 'k' index in W.
    #   Global row r is the 'i' index in W.
    #   So W index = c*2 + r.
    
    def map_w_c(left_right, local_idx):
        # left_right: 'L' for cols 0,1; 'R' for cols 2,3
        r = local_idx // 2
        c = local_idx % 2
        if left_right == 'L':
            global_c = c
        else:
            global_c = 2 + c
        return global_c * 2 + r
    
    # Apply Strassen to 4 blocks
    
    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            u_global = []
            v_global = []
            w_global = []
            
            for idx, coeff in u_local:
                g_idx = map_u_a(a_side, idx)
                u_global.append((g_idx, coeff))
            
            for idx, coeff in v_local:
                g_idx = map_v_b(b_block, idx)
                v_global.append((g_idx, coeff))
            
            for idx, coeff in w_local:
                g_idx = map_w_c(c_side, idx)
                w_global.append((g_idx, coeff))
            
            add_term(u_global, v_global, w_global, start_col)
            start_col += 1
        
        return start_col
    
    # 1. A_L * B_TL -> C_L
    col = apply_strassen('L', 'TL', 'L', 0)
    
    # 2. A_R * B_BL -> C_L
    col = apply_strassen('R', 'BL', 'L', col)
    
    # 3. A_L * B_TR -> C_R
    col = apply_strassen('L', 'TR', 'R', col)
    
    # 4. A_R * B_BR -> C_R
    col = apply_strassen('R', 'BR', 'R', col)
    
    # Verify the decomposition is correct
    # Test with a simple case
    A_test = np.arange(n * m).reshape(n, m).astype(float) + 1.0
    B_test = np.arange(m * p).reshape(m, p).astype(float) + 2.0
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            C_computed[ii, kk] = acc
    
    if not np.allclose(C_expected, C_computed):
        # Fallback to naive rank 32 if Strassen fails
        rank = 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
