import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Define the 7 terms for 2x2x2 Strassen
    # u, v, w are vectors of length 4 (flattened 2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, -1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]), # M5
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0]), # M7
    ]
    
    # Define the 7 Strassen products for 4x4x4
    # Each product is defined by:
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    # Block indices: 0:00, 1:01, 2:10, 3:11
    
    strassen_4x4_products = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00, C01, C10, C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (1, 1), (2, 1), (3, 1)]),
        # M2 = (A10 + A11) * B00 -> C10, C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),
        # M3 = A00 * (B01 - B11) -> C01, C11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11 * (B10 - B00) -> C10, C11
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (3, 1)]),
        # M5 = (A00 + A01) * B11 -> C00, C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # M6 = (A10 - A00) * (B00 + B01) -> C10, C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),
        # M7 = (A01 - A11) * (B10 + B11) -> C00, C01
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]
    
    # Collect all rank-1 terms
    terms = []  # Each term is (u_vec, v_vec, w_vec) as dicts or lists
    
    def add_4x4_block(k_off):
        """Adds 49 rank-1 terms for a 4x4x4 block at column offset k_off."""
        for a_def, b_def, c_def in strassen_4x4_products:
            for u_t, v_t, w_t in strassen_2x2:
                # Build U, V, W vectors for this rank-1 term
                u_vec = {}
                v_vec = {}
                w_vec = {}
                
                # U: A is 4x4
                for b_idx, coeff in a_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    u_vec[(r0, c0)] = u_vec.get((r0, c0), 0) + coeff * u_t[0]
                    u_vec[(r0, c0+1)] = u_vec.get((r0, c0+1), 0) + coeff * u_t[1]
                    u_vec[(r0+1, c0)] = u_vec.get((r0+1, c0), 0) + coeff * u_t[2]
                    u_vec[(r0+1, c0+1)] = u_vec.get((r0+1, c0+1), 0) + coeff * u_t[3]
                
                # V: B is 4x9, offset k_off
                for b_idx, coeff in b_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    v_vec[(r0, c0 + k_off)] = v_vec.get((r0, c0 + k_off), 0) + coeff * v_t[0]
                    v_vec[(r0, c0 + 1 + k_off)] = v_vec.get((r0, c0 + 1 + k_off), 0) + coeff * v_t[1]
                    v_vec[(r0+1, c0 + k_off)] = v_vec.get((r0+1, c0 + k_off), 0) + coeff * v_t[2]
                    v_vec[(r0+1, c0 + 1 + k_off)] = v_vec.get((r0+1, c0 + 1 + k_off), 0) + coeff * v_t[3]
                
                # W: C is 4x9, offset k_off. W[k*n + i] where k is col of C, i is row of C
                for b_idx, coeff in c_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    # r0 is row of C block -> this is the "i" index (row of C)
                    # c0 is col of C block -> this is the "k" index (col of C)
                    # W index: k*n + i = (c0 + k_off)*4 + r0
                    w_vec[(c0 + k_off, r0)] = w_vec.get((c0 + k_off, r0), 0) + coeff * w_t[0]
                    w_vec[(c0 + 1 + k_off, r0)] = w_vec.get((c0 + 1 + k_off, r0), 0) + coeff * w_t[1]
                    w_vec[(c0 + k_off, r0 + 1)] = w_vec.get((c0 + k_off, r0 + 1), 0) + coeff * w_t[2]
                    w_vec[(c0 + 1 + k_off, r0 + 1)] = w_vec.get((c0 + 1 + k_off, r0 + 1), 0) + coeff * w_t[3]
                
                terms.append((u_vec, v_vec, w_vec))
    
    add_4x4_block(0)
    add_4x4_block(4)
    
    # Block 3: Column 8 of B. Schoolbook rank 16.
    for i in range(4):
        for j in range(4):
            u_vec = {(i, j): 1}
            v_vec = {(j, 8): 1}
            w_vec = {(8, i): 1}
            terms.append((u_vec, v_vec, w_vec))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_vec, v_vec, w_vec) in enumerate(terms):
        for (i, j), val in u_vec.items():
            U[i * m + j][c] = val
        for (j, k), val in v_vec.items():
            V[j * p + k][c] = val
        for (k, i), val in w_vec.items():
            W[k * n + i][c] = val
    
    # Verify
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 9)
    C_expected = A_test @ B_test
    
    C_computed = np.einsum('ijc,jkc,kic->ik', U_r, V_r, W_r)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank_s = 144
        U_s = [[0] * rank_s for _ in range(16)]
        V_s = [[0] * rank_s for _ in range(36)]
        W_s = [[0] * rank_s for _ in range(36)]
        col_s = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_s[i*4+j][col_s] = 1
                    V_s[j*9+k][col_s] = 1
                    W_s[k*4+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
