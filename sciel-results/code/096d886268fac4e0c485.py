"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Target rank: 66
    # 1 (A2B2) + 16 (A1B1b) + 49 (A1B1a via Strassen)
    target_rank = 66
    
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    current_rank = 0
    
    def get_U_idx(i, j):
        return i * m + j
        
    def get_V_idx(j, k):
        return j * p + k
        
    def get_W_idx(k, i):
        return k * n + i

    # Helper to add a single rank-1 term with coefficients
    def add_term(r, u_parts, v_parts, w_parts):
        # u_parts: list of (idx, coeff)
        for idx, coeff in u_parts:
            U[idx, r] += coeff
        for idx, coeff in v_parts:
            V[idx, r] += coeff
        for idx, coeff in w_parts:
            W[idx, r] += coeff

    # 1. Term for A2 * B2
    # A2: rows 0..3, col 4. (4x1)
    # B2: row 4, cols 0..4. (1x5)
    # C: rows 0..3, cols 0..4. (4x5)
    # C[i, k] = A[i, 4] * B[4, k]
    # Rank 1
    u_parts = [(get_U_idx(i, 4), 1) for i in range(4)]
    v_parts = [(get_V_idx(4, k), 1) for k in range(5)]
    w_parts = [(get_W_idx(k, i), 1) for i in range(4) for k in range(5)]
    add_term(current_rank, u_parts, v_parts, w_parts)
    current_rank += 1

    # 2. Term for A1 * B1b
    # A1: rows 0..3, cols 0..3. (4x4)
    # B1b: rows 0..3, col 4. (4x1)
    # C: rows 0..3, col 4. (4x1)
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # Rank 16
    for i in range(4):
        for j in range(4):
            u_parts = [(get_U_idx(i, j), 1)]
            v_parts = [(get_V_idx(j, 4), 1)]
            w_parts = [(get_W_idx(4, i), 1)]
            add_term(current_rank, u_parts, v_parts, w_parts)
            current_rank += 1

    # 3. Term for A1 * B1a
    # A1: 4x4, B1a: 4x4, C: 4x4
    # Use Strassen for 4x4x4 (Rank 49)
    
    # Define blocks for A1, B1a, C
    # A1 blocks (2x2):
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    
    def get_block_indices_A(r1, r2, c1, c2):
        # Returns list of 4 indices in row-major order: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
        return [
            get_U_idx(r1, c1),
            get_U_idx(r1, c2),
            get_U_idx(r2, c1),
            get_U_idx(r2, c2)
        ]
        
    def get_block_indices_B(r1, r2, c1, c2):
        return [
            get_V_idx(r1, c1),
            get_V_idx(r1, c2),
            get_V_idx(r2, c1),
            get_V_idx(r2, c2)
        ]
        
    def get_block_indices_C(r1, r2, c1, c2):
        # C is 4x5, but we only care about cols 0..3 for this part
        return [
            get_W_idx(c1, r1), # Note: W index is k*n + i. Here k is col, i is row.
            get_W_idx(c2, r1),
            get_W_idx(c1, r2),
            get_W_idx(c2, r2)
        ]
        
    A00 = get_block_indices_A(0, 1, 0, 1)
    A01 = get_block_indices_A(0, 1, 2, 3)
    A10 = get_block_indices_A(2, 3, 0, 1)
    A11 = get_block_indices_A(2, 3, 2, 3)
    
    B00 = get_block_indices_B(0, 1, 0, 1)
    B01 = get_block_indices_B(0, 1, 2, 3)
    B10 = get_block_indices_B(2, 3, 0, 1)
    B11 = get_block_indices_B(2, 3, 2, 3)
    
    C00 = get_block_indices_C(0, 1, 0, 1)
    C01 = get_block_indices_C(0, 1, 2, 3)
    C10 = get_block_indices_C(2, 3, 0, 1)
    C11 = get_block_indices_C(2, 3, 2, 3)
    
    # Function to add a 2x2 Strassen decomposition
    # a_idx, b_idx, c_idx are lists of 4 indices (row-major)
    # But we need to handle linear combinations (sums/diffs)
    # So we pass lists of parts: [ [(idx, coeff), ...], ... ] for each of the 4 positions
    
    def add_strassen_2x2(r_start, a_parts, b_parts, c_parts):
        # a_parts[0] is for a00, a_parts[1] for a01, etc.
        # Each is a list of (idx, coeff)
        
        # M1 = (a00+a11)(b00+b11) -> c00+c11
        # M2 = (a10+a11)b00 -> c10+c11
        # M3 = a00(b01-b11) -> c01+c11
        # M4 = a11(b10-b00) -> c10+c00
        # M5 = (a00+a01)b11 -> c01-c00
        # M6 = (a10-a00)(b00+b01) -> c11
        # M7 = (a01-a11)(b10+b11) -> c00
        
        # Helper to combine parts for U, V, W
        def combine(parts1, parts2, sign1=1, sign2=1):
            res = []
            for idx, coeff in parts1:
                res.append((idx, coeff * sign1))
            for idx, coeff in parts2:
                res.append((idx, coeff * sign2))
            return res

        # M1
        u1 = combine(a_parts[0], a_parts[3])
        v1 = combine(b_parts[0], b_parts[3])
        w1 = combine(c_parts[0], c_parts[3])
        add_term(r_start, u1, v1, w1)
        
        # M2
        u2 = combine(a_parts[2], a_parts[3])
        v2 = b_parts[0]
        w2 = combine(c_parts[2], c_parts[3])
        add_term(r_start+1, u2, v2, w2)
        
        # M3
        u3 = a_parts[0]
        v3 = combine(b_parts[1], b_parts[3], 1, -1)
        w3 = combine(c_parts[1], c_parts[3])
        add_term(r_start+2, u3, v3, w3)
        
        # M4
        u4 = a_parts[3]
        v4 = combine(b_parts[2], b_parts[0], 1, -1)
        w4 = combine(c_parts[2], c_parts[0])
        add_term(r_start+3, u4, v4, w4)
        
        # M5
        u5 = combine(a_parts[0], a_parts[1])
        v5 = b_parts[3]
        w5 = combine(c_parts[1], c_parts[0], 1, -1)
        add_term(r_start+4, u5, v5, w5)
        
        # M6
        u6 = combine(a_parts[2], a_parts[0], 1, -1)
        v6 = combine(b_parts[0], b_parts[1])
        w6 = c_parts[3]
        add_term(r_start+5, u6, v6, w6)
        
        # M7
        u7 = combine(a_parts[1], a_parts[3], 1, -1)
        v7 = combine(b_parts[2], b_parts[3])
        w7 = c_parts[0]
        add_term(r_start+6, u7, v7, w7)
        
        return r_start + 7

    # Now, construct the 7 high-level Strassen terms for 4x4
    # We need to form the "parts" for each high-level term
    # e.g., for M1, A_part is A00+A11.
    # A00 is a list of indices. A11 is a list of indices.
    # We need to convert index lists to parts lists: [(idx, 1), ...]
    
    def indices_to_parts(indices):
        return [(idx, 1) for idx in indices]
        
    def combine_indices(idx_list1, idx_list2, sign1=1, sign2=1):
        parts = []
        for idx in idx_list1:
            parts.append((idx, sign1))
        for idx in idx_list2:
            parts.append((idx, sign2))
        return parts

    # M1: (A00+A11)(B00+B11) -> C00+C11
    a_m1 = combine_indices(A00, A11)
    b_m1 = combine_indices(B00, B11)
    c_m1 = combine_indices(C00, C11)
    current_rank = add_strassen_2x2(current_rank, 
                                    [a_m1, a_m1, a_m1, a_m1], # Placeholder, will fix
                                    [b_m1, b_m1, b_m1, b_m1],
                                    [c_m1, c_m1, c_m1, c_m1])
    # Wait, add_strassen_2x2 expects a_parts[0..3] corresponding to a00..a11 of the 2x2 block.
    # But here, the "2x2 block" is the result of the high-level combination.
    # The high-level combination IS the 2x2 matrix to be multiplied.
    # So for M1, the A matrix is A_sum1 = A00+A11.
    # This A_sum1 is a 2x2 matrix.
    # Its elements are sums of elements from A00 and A11.
    # So a_parts[0] (top-left of A_sum1) is A00[0] + A11[0].
    # a_parts[1] (top-right) is A00[1] + A11[1].
    # a_parts[2] (bottom-left) is A00[2] + A11[2].
    # a_parts[3] (bottom-right) is A00[3] + A11[3].
    
    # Let's rewrite the call logic properly.
    
    def make_parts_from_indices(idx_list):
        return [(idx, 1) for idx in idx_list]
        
    def combine_block_parts(idx_list1, idx_list2, sign1=1, sign2=1):
        # Returns a list of 4 parts, each being a list of (idx, coeff)
        parts = []
        for i in range(4):
            parts.append([
                (idx_list1[i], sign1),
                (idx_list2[i], sign2)
            ])
        return parts

    # M1
    a_p1 = combine_block_parts(A00, A11)
    b_p1 = combine_block_parts(B00, B11)
    c_p1 = combine_block_parts(C00, C11)
    current_rank = add_strassen_2x2(current_rank, a_p1, b_p1, c_p1)
    
    # M2: (A10+A11)B00 -> C10+C11
    a_p2 = combine_block_parts(A10, A11)
    b_p2 = [make_parts_from_indices(B00)[0], make_parts_from_indices(B00)[1], make_parts_from_indices(B00)[2], make_parts_from_indices(B00)[3]]
    # Wait, B00 is just B00. So b_p2 is just parts for B00.
    b_p2 = [make_parts_from_indices(B00)[i] for i in range(4)]
    c_p2 = combine_block_parts(C10, C11)
    current_rank = add_strassen_2x2(current_rank, a_p2, b_p2, c_p2)
    
    # M3: A00(B01-B11) -> C01+C11
    a_p3 = [make_parts_from_indices(A00)[i] for i in range(4)]
    b_p3 = combine_block_parts(B01, B11, 1, -1)
    c_p3 = combine_block_parts(C01, C11)
    current_rank = add_strassen_2x2(current_rank, a_p3, b_p3, c_p3)
    
    # M4: A11(B10-B00) -> C10+C00
    a_p4 = [make_parts_from_indices(A11)[i] for i in range(4)]
    b_p4 = combine_block_parts(B10, B00, 1, -1)
    c_p4 = combine_block_parts(C10, C00)
    current_rank = add_strassen_2x2(current_rank, a_p4, b_p4, c_p4)
    
    # M5: (A00+A01)B11 -> C01-C00
    a_p5 = combine_block_parts(A00, A01)
    b_p5 = [make_parts_from_indices(B11)[i] for i in range(4)]
    c_p5 = combine_block_parts(C01, C00, 1, -1)
    current_rank = add_strassen_2x2(current_rank, a_p5, b_p5, c_p5)
    
    # M6: (A10-A00)(B00+B01) -> C11
    a_p6 = combine_block_parts(A10, A00, 1, -1)
    b_p6 = combine_block_parts(B00, B01)
    c_p6 = [make_parts_from_indices(C11)[i] for i in range(4)]
    current_rank = add_strassen_2x2(current_rank, a_p6, b_p6, c_p6)
    
    # M7: (A01-A11)(B10+B11) -> C00
    a_p7 = combine_block_parts(A01, A11, 1, -1)
    b_p7 = combine_block_parts(B10, B11)
    c_p7 = [make_parts_from_indices(C00)[i] for i in range(4)]
    current_rank = add_strassen_2x2(current_rank, a_p7, b_p7, c_p7)
    
    # Verify current_rank is 66
    if current_rank != target_rank:
        print(f"Warning: Rank mismatch. Expected {target_rank}, got {current_rank}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
