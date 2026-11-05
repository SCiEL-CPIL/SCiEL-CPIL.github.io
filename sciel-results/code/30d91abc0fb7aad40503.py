import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    # We will build the decomposition using the block structure:
    # A (4x5) = [A00 (4x4) | A01 (4x1)]
    # B (5x5) = [B00 (4x5); B10 (1x5)]
    # C (4x5) = A00 * B00 + A01 * B10
    
    # Further decompose B00 (4x5) = [B00_left (4x4) | B00_right (4x1)]
    # C = A00 * B00_left + A00 * B00_right + A01 * B10
    
    # 1. A00 * B00_left: 4x4 * 4x4. Use 49-term Strassen.
    # 2. A00 * B00_right: 4x4 * 4x1. Schoolbook rank 16.
    # 3. A01 * B10: 4x1 * 1x5. Schoolbook rank 20.
    
    # Total rank = 49 + 16 + 20 = 85.
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # --- Part 1: 4x4 * 4x4 using Strassen (49 terms) ---
    # A00 indices: i, k in 0..3 -> U[i*5+k]
    # B00_left indices: k, j in 0..3 -> V[k*5+j]
    # C indices: i, j in 0..3 -> W[j*4+i]
    
    # We implement the standard nested Strassen for 4x4.
    # Outer Strassen splits 4x4 into four 2x2 blocks.
    # Inner Strassen splits 2x2 into four 1x1 blocks.
    
    # Global index mapping for A00 (4x4)
    # Block (br, bc) of A00 corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # Global U index for A00[i,k] is i*5+k
    
    # Global index mapping for B00_left (4x4)
    # Block (br, bc) of B00_left corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # Global V index for B00_left[k,j] is k*5+j
    
    # Global index mapping for C (4x5)
    # We only care about C[i,j] for j in 0..3 in this part.
    # Global W index for C[i,j] is j*4+i
    
    # Strassen 2x2 factors (local indices 0..3 for 2x2 matrices)
    # U_local[4], V_local[4], W_local[4]
    # 7 terms
    s_u = [
        [1, 0, 0, 1], # T0
        [0, 0, 1, 1], # T1
        [1, 0, 0, 0], # T2
        [0, 0, 0, 1], # T3
        [1, 1, 0, 0], # T4
        [-1, 0, 1, 0],# T5
        [0, 1, 0, -1] # T6
    ]
    s_v = [
        [1, 0, 0, 1], # T0
        [1, 0, 0, 0], # T1
        [0, 1, 0, -1],# T2
        [0, 0, 1, -1],# T3
        [0, 0, 0, 1], # T4
        [1, 1, 0, 0], # T5
        [0, 0, 1, 1]  # T6
    ]
    s_w = [
        [1, 0, 0, 1], # T0
        [0, 0, 1, -1],# T1
        [0, 1, 0, 1], # T2
        [1, 0, 1, 0], # T3
        [-1, 1, 0, 0],# T4
        [0, 0, 0, 1], # T5
        [1, 0, 0, 0]  # T6
    ]
    
    # Outer Strassen coefficients for 4x4
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    # M2 = (A21+A22)B11 -> C21, C22
    # M3 = A11(B12-B22) -> C12, C22
    # M4 = A22(B21-B11) -> C11, C21
    # M5 = (A11+A12)B22 -> C11, C12
    # M6 = (A21-A11)(B11+B12) -> C22
    # M7 = (A12-A22)(B21+B22) -> C11
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # For each outer term t, we have:
    # A_coeffs: dict {block: coeff}
    # B_coeffs: dict {block: coeff}
    # C_coeffs: dict {block: coeff}
    
    outer_terms = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}), # M1
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),          # M2
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),          # M3
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),          # M4
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),          # M5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),          # M6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1})           # M7
    ]
    
    # Function to get global indices for a 2x2 block in a 4x4 matrix
    # block 'brbc' -> (br, bc)
    def block_to_idx(block):
        br = int(block[0])
        bc = int(block[1])
        return br, bc
    
    for a_coeffs, b_coeffs, c_coeffs in outer_terms:
        for t in range(7):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            # Fill U
            for a_block, a_sign in a_coeffs.items():
                ab_r, ab_c = block_to_idx(a_block)
                for i_loc in range(2):
                    for k_loc in range(2):
                        # Local 2x2 index in s_u[t]
                        # The 2x2 block A_ab corresponds to entries (ab_r*2+i_loc, ab_c*2+k_loc)
                        # In the 2x2 Strassen factors, the indices are:
                        # 0: 00, 1: 01, 2: 10, 3: 11
                        # So local index is i_loc * 2 + k_loc
                        local_idx = i_loc * 2 + k_loc
                        coeff = a_sign * s_u[t][local_idx]
                        if coeff != 0:
                            g_i = ab_r * 2 + i_loc
                            g_k = ab_c * 2 + k_loc
                            u_vec[g_i * m + g_k] += coeff
            
            # Fill V
            for b_block, b_sign in b_coeffs.items():
                bb_r, bb_c = block_to_idx(b_block)
                for k_loc in range(2):
                    for j_loc in range(2):
                        local_idx = k_loc * 2 + j_loc
                        coeff = b_sign * s_v[t][local_idx]
                        if coeff != 0:
                            g_k = bb_r * 2 + k_loc
                            g_j = bb_c * 2 + j_loc
                            v_vec[g_k * p + g_j] += coeff
            
            # Fill W
            for c_block, c_sign in c_coeffs.items():
                cb_r, cb_c = block_to_idx(c_block)
                for i_loc in range(2):
                    for j_loc in range(2):
                        local_idx = i_loc * 2 + j_loc
                        coeff = c_sign * s_w[t][local_idx]
                        if coeff != 0:
                            g_i = cb_r * 2 + i_loc
                            g_j = cb_c * 2 + j_loc
                            w_vec[g_j * n + g_i] += coeff
            
            add_rank1(u_vec, v_vec, w_vec)

    # --- Part 2: A00 * B00_right (4x4 * 4x1) ---
    # A00[i,k] for i,k in 0..3 -> U[i*5+k]
    # B00_right[k,4] for k in 0..3 -> V[k*5+4]
    # C[i,4] for i in 0..3 -> W[4*4+i] = W[16+i]
    
    for i in range(4):
        for k in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + k] = 1.0
            v_vec[k * p + 4] = 1.0
            w_vec[4 * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)

    # --- Part 3: A01 * B10 (4x1 * 1x5) ---
    # A01[i,4] for i in 0..3 -> U[i*5+4]
    # B10[4,j] for j in 0..4 -> V[4*5+j] = V[20+j]
    # C[i,j] for i,j in 0..4 -> W[j*4+i]
    
    for i in range(4):
        for j in range(5):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + 4] = 1.0
            v_vec[4 * p + j] = 1.0
            w_vec[j * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    
    if rank == 0:
        # Fallback
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[0] = 1.0
        v_vec[0] = 1.0
        w_vec[0] = 1.0
        add_rank1(u_vec, v_vec, w_vec)
        rank = 1

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
