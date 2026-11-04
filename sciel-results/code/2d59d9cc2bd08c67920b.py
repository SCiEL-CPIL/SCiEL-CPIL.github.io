import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # We split B (4x9) into B_L (4x4), B_M (4x4), B_R (4x1)
    # C = A B_L + A B_M + A B_R
    
    # Rank contributions:
    # A B_L (4x4x4) -> Strassen rank 49
    # A B_M (4x4x4) -> Strassen rank 49
    # A B_R (4x4x1) -> Schoolbook rank 16
    # Total rank = 114
    
    rank = 114
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term to U, V, W
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # Note: U is indexed by (i, j) of A. V by (j, k) of B. W by (k, i) of C.
    
    def add_term(u_indices, v_indices, w_indices, col):
        # u_indices: list of (idx, coeff)
        # v_indices: list of (idx, coeff)
        # w_indices: list of (idx, coeff)
        for idx, coeff in u_indices:
            U[idx, col] += coeff
        for idx, coeff in v_indices:
            V[idx, col] += coeff
        for idx, coeff in w_indices:
            W[idx, col] += coeff

    # Strassen 2x2 decomposition
    # Returns 7 tuples of (u_coeffs, v_coeffs, w_coeffs) for 2x2 matrices
    # u_coeffs: [q00, q01, q10, q11]
    # v_coeffs: [r00, r01, r10, r11]
    # w_coeffs: [p00, p01, p10, p11]
    
    def strassen_2x2_terms():
        terms = []
        # S1 = (q00+q11)(r00+r11) -> p00 += 1, p11 += 1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # S2 = (q10+q11)r00 -> p10 += 1
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # S3 = q00(r01-r11) -> p01 += 1
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        # S4 = q11(r10-r00) -> p00 += 1, p10 += 1
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
        # S5 = (q00+q01)r11 -> p00 -= 1, p01 += 1
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # S6 = (q10-q00)(r00+r01) -> p11 += 1
        terms.append([[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
        # S7 = (q01-q11)(r10+r11) -> p00 += 1
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return terms

    s_terms = strassen_2x2_terms()
    
    # Level 1 Strassen for 4x4x4
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Define the linear combinations for A and B blocks
    # Each entry is a list of (block_idx, coeff)
    # Block indices: 00, 01, 10, 11
    
    l1_terms = [
        # M1
        {
            'A': {0: 1, 3: 1}, # A00 + A11
            'B': {0: 1, 3: 1}, # B00 + B11
            'C': {0: 1, 3: 1}  # C00 + C11
        },
        # M2
        {
            'A': {2: 1, 3: 1}, # A10 + A11
            'B': {0: 1},       # B00
            'C': {2: 1}        # C10
        },
        # M3
        {
            'A': {0: 1},       # A00
            'B': {1: 1, 3: -1},# B01 - B11
            'C': {1: 1}        # C01
        },
        # M4
        {
            'A': {3: 1},       # A11
            'B': {2: 1, 0: -1},# B10 - B00
            'C': {0: 1, 2: 1}  # C00 + C10
        },
        # M5
        {
            'A': {0: 1, 1: 1}, # A00 + A01
            'B': {3: 1},       # B11
            'C': {0: -1, 1: 1} # -C00 + C01
        },
        # M6
        {
            'A': {2: 1, 0: -1},# A10 - A00
            'B': {0: 1, 1: 1}, # B00 + B01
            'C': {3: 1}        # C11
        },
        # M7
        {
            'A': {1: 1, 3: -1},# A01 - A11
            'B': {2: 1, 3: 1}, # B10 + B11
            'C': {0: 1}        # C00
        }
    ]
    
    def get_block_indices(block_idx, matrix_type, k_offset=0):
        # Returns list of (global_idx, coeff) for the 4 elements of a 2x2 block
        # matrix_type: 'A' or 'B'
        # block_idx: 0 (00), 1 (01), 2 (10), 3 (11)
        # For A: i_block = block_idx // 2, j_block = block_idx % 2
        # For B: j_block = block_idx // 2, k_block = block_idx % 2
        # k_offset is for B's column offset (0 for B_L, 4 for B_M)
        
        if matrix_type == 'A':
            ib = block_idx // 2
            jb = block_idx % 2
            base_i = ib * 2
            base_j = jb * 2
            indices = []
            for di in range(2):
                for dj in range(2):
                    i = base_i + di
                    j = base_j + dj
                    idx = i * m + j
                    indices.append((idx, 1))
            return indices
        else: # B
            jb = block_idx // 2
            kb = block_idx % 2
            base_j = jb * 2
            base_k = kb * 2 + k_offset
            indices = []
            for dj in range(2):
                for dk in range(2):
                    j = base_j + dj
                    k = base_k + dk
                    idx = j * p + k
                    indices.append((idx, 1))
            return indices

    def get_c_block_indices(block_idx):
        # C block indices
        # block_idx: 0 (00), 1 (01), 2 (10), 3 (11)
        ib = block_idx // 2
        jb = block_idx % 2
        base_i = ib * 2
        base_j = jb * 2
        indices = []
        for di in range(2):
            for dj in range(2):
                i = base_i + di
                k = base_j + dj # k is row of C, which is col of B
                # W index is k*n + i
                idx = k * n + i
                indices.append((idx, 1))
        return indices

    col = 0
    for k_offset in [0, 4]: # B_L and B_M
        for l1 in l1_terms:
            # For each of the 7 Strassen terms for 2x2
            for s_term in s_terms:
                q_coeffs, r_coeffs, p_coeffs = s_term
                
                u_indices = []
                v_indices = []
                w_indices = []
                
                # Map q_coeffs (4 values) to A indices
                # q_coeffs correspond to [q00, q01, q10, q11]
                # q00 is (0,0) of the 2x2 block
                # We need to distribute these to the A blocks involved
                # l1['A'] maps block_idx -> coeff
                for block_idx, a_coeff in l1['A'].items():
                    block_indices = get_block_indices(block_idx, 'A')
                    # The 2x2 block has elements (0,0), (0,1), (1,0), (1,1)
                    # q_coeffs[0] corresponds to (0,0)
                    # q_coeffs[1] corresponds to (0,1)
                    # q_coeffs[2] corresponds to (1,0)
                    # q_coeffs[3] corresponds to (1,1)
                    for di in range(2):
                        for dj in range(2):
                            q_c = q_coeffs[di * 2 + dj]
                            if q_c != 0:
                                idx, _ = block_indices[di * 2 + dj]
                                u_indices.append((idx, q_c * a_coeff))
                
                # Map r_coeffs (4 values) to B indices
                for block_idx, b_coeff in l1['B'].items():
                    block_indices = get_block_indices(block_idx, 'B', k_offset)
                    for di in range(2):
                        for dj in range(2):
                            r_c = r_coeffs[di * 2 + dj]
                            if r_c != 0:
                                idx, _ = block_indices[di * 2 + dj]
                                v_indices.append((idx, r_c * b_coeff))
                
                # Map p_coeffs (4 values) to C indices
                for block_idx, c_coeff in l1['C'].items():
                    block_indices = get_c_block_indices(block_idx)
                    for di in range(2):
                        for dj in range(2):
                            p_c = p_coeffs[di * 2 + dj]
                            if p_c != 0:
                                idx, _ = block_indices[di * 2 + dj]
                                w_indices.append((idx, p_c * c_coeff))
                
                add_term(u_indices, v_indices, w_indices, col)
                col += 1
                
    # Add Schoolbook for B_R (4x1)
    # A (4x4) * B_R (4x1) = C_R (4x1)
    # C_R[i] = sum_j A[i, j] B_R[j]
    # B_R is at column k=8
    for i in range(4):
        for j in range(4):
            u_indices = [(i * m + j, 1)]
            v_indices = [(j * p + 8, 1)]
            w_indices = [(8 * n + i, 1)]
            add_term(u_indices, v_indices, w_indices, col)
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
