import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation using Strassen for 2x2 blocks and schoolbook for 1x2 blocks
    # 2x2 A blocks * 2x2 B blocks -> 7 rank units each.
    # 1x2 A blocks * 2x2 B blocks -> 2 rank units each (schoolbook 1x2x2 is rank 2? No, 1x2x2 is rank 2? 
    # 1x2x2: C1 = A1 B1 + A2 B2. Rank 2.
    # So 1x2 A * 2x8 B (four 2x2 blocks) -> 4 * 2 = 8 rank units per 1x4 A block?
    # Wait. 1x2 A * 2x2 B -> 1x2 C.
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # This requires 4 rank units?
    # Term 1: A1 B11 -> C1
    # Term 2: A2 B21 -> C1
    # Term 3: A1 B12 -> C2
    # Term 4: A2 B22 -> C2
    # Yes, 4 rank units per 1x2x2 block.
    
    # Current code used 16 for 1x4x8.
    # 1x4x8 = 4 blocks of 1x2x2? No.
    # A is 1x2. B is 2x8.
    # B is 4 blocks of 2x2.
    # Each 1x2x2 block costs 4.
    # Total 16. Correct.
    
    rank = 88
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_rank_unit(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in zip(u_indices, u_vals):
            U[idx, col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, col] += val
        col += 1

    def strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Add Strassen decomposition for 2x2 A, 2x2 B -> 2x2 C.
        a_r, a_c: row/col indices of A block (2x2)
        b_r, b_c: row/col indices of B block (2x2)
        c_r, c_c: row/col indices of C block (2x2)
        """
        # A indices
        a00 = a_r[0] * m + a_c[0]
        a01 = a_r[0] * m + a_c[1]
        a10 = a_r[1] * m + a_c[0]
        a11 = a_r[1] * m + a_c[1]
        
        # B indices
        b00 = b_r[0] * p + b_c[0]
        b01 = b_r[0] * p + b_c[1]
        b10 = b_r[1] * p + b_c[0]
        b11 = b_r[1] * p + b_c[1]
        
        # C indices (w_idx = col*n + row)
        c00 = b_c[0] * n + c_r[0]
        c01 = b_c[1] * n + c_r[0]
        c10 = b_c[0] * n + c_r[1]
        c11 = b_c[1] * n + c_r[1]
        
        # M1 = (a+d)(e+h)
        add_rank_unit([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        # M2 = (c+d)e
        add_rank_unit([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        # M3 = a(f-h)
        add_rank_unit([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        # M4 = d(g-e)
        add_rank_unit([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        # M5 = (a+b)h
        add_rank_unit([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        # M6 = (c-a)(e+f)
        add_rank_unit([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        # M7 = (b-d)(g+h)
        add_rank_unit([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # B is 2x8, split into four 2x2 blocks
    for k_start in range(0, 8, 2):
        a_r = [0, 1]
        a_c = [0, 1]
        b_r = [0, 1]
        b_c = [k_start, k_start + 1]
        c_r = [0, 1]
        c_c = [k_start, k_start + 1]
        strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:8]
    for k_start in range(0, 8, 2):
        a_r = [0, 1]
        a_c = [2, 3]
        b_r = [2, 3]
        b_c = [k_start, k_start + 1]
        c_r = [0, 1]
        c_c = [k_start, k_start + 1]
        strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c)

    # Block 3: A[2:3, 0:2] * B[0:2, 0:8]
    # Schoolbook 1x2 * 2x8
    for k in range(8):
        # C[2, k] = A[2, 0] B[0, k] + A[2, 1] B[1, k]
        add_rank_unit([2 * m + 0], [1], [0 * p + k], [1], [k * n + 2], [1])
        add_rank_unit([2 * m + 1], [1], [1 * p + k], [1], [k * n + 2], [1])

    # Block 4: A[2:3, 2:4] * B[2:4, 0:8]
    for k in range(8):
        # C[2, k] = A[2, 2] B[2, k] + A[2, 3] B[3, k]
        add_rank_unit([2 * m + 2], [1], [2 * p + k], [1], [k * n + 2], [1])
        add_rank_unit([2 * m + 3], [1], [3 * p + k], [1], [k * n + 2], [1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
