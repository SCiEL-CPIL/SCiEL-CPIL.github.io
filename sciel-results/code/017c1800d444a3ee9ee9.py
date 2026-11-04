import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2 blocks of A (2x2 each)
    # For each A block, we multiply with a 2x7 B block.
    # 2x7 B block is split into three 2x2 and one 2x1.
    # Each 2x2x2 uses Strassen (rank 7).
    # Each 2x2x1 uses direct (rank 4).
    # Rank per A block = 3*7 + 1*4 = 25.
    # Total rank = 2 * 25 = 50.
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 50
    V = [[0] * rank for _ in range(m * p)] # 28 x 50
    W = [[0] * rank for _ in range(p * n)] # 14 x 50
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_coeffs:
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs:
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs:
            if val != 0:
                W[idx][col] += val
        col += 1

    # Helper for Strassen block
    def strassen_block(a_j_start, b_j_start, c_k_start):
        # a_j_start: 0 or 2 (column offset in A)
        # b_j_start: 0 or 2 (row offset in B)
        # c_k_start: 0, 2, 4, 6 (column offset in C)
        
        # Local indices mapping
        # A: i in {0,1}, j_local in {0,1} -> global j = a_j_start + j_local
        # U index: i * m + j_global
        # B: j_local in {0,1}, k_local in {0,1} -> global j = b_j_start + j_local, k = c_k_start + k_local
        # V index: j_global * p + k_global
        # C: i in {0,1}, k_local in {0,1} -> global k = c_k_start + k_local
        # W index: k_global * n + i

        def u_idx(i, j_loc):
            return i * m + (a_j_start + j_loc)
            
        def v_idx(j_loc, k_loc):
            return (b_j_start + j_loc) * p + (c_k_start + k_loc)
            
        def w_idx(i, k_loc):
            return (c_k_start + k_loc) * n + i

        # M1: (A00+A11)(B00+B11) -> C00+C11
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1), (w_idx(1,1), 1)]
        )
        
        # M2: (A10+A11)B00 -> C10
        add_term(
            [(u_idx(1,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1)],
            [(w_idx(1,0), 1)]
        )
        
        # M3: A00 B11 -> C01+C11
        add_term(
            [(u_idx(0,0), 1)],
            [(v_idx(1,1), 1)],
            [(w_idx(0,1), 1), (w_idx(1,1), 1)]
        )
        
        # M4: A11 (B10-B00) -> C00+C10
        add_term(
            [(u_idx(1,1), 1)],
            [(v_idx(1,0), 1), (v_idx(0,0), -1)],
            [(w_idx(0,0), 1), (w_idx(1,0), 1)]
        )
        
        # M5: A01 (B10+B11) -> -C00+C01
        add_term(
            [(u_idx(0,1), 1)],
            [(v_idx(1,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), -1), (w_idx(0,1), 1)]
        )
        
        # M6: (A00-A10)(B01-B11) -> C11
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,0), -1)],
            [(v_idx(0,1), 1), (v_idx(1,1), -1)],
            [(w_idx(1,1), 1)]
        )
        
        # M7: (A01+A11)(B00-B01) -> C00
        add_term(
            [(u_idx(0,1), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1), (v_idx(0,1), -1)],
            [(w_idx(0,0), 1)]
        )

    # Helper for Single block (2x2x1)
    def single_block(a_j_start, b_j_start, c_k_start):
        # k_local is always 0
        
        def u_idx(i, j_loc):
            return i * m + (a_j_start + j_loc)
            
        def v_idx(j_loc):
            return (b_j_start + j_loc) * p + c_k_start
            
        def w_idx(i):
            return c_k_start * n + i

        # M1: A00 B0 -> C0
        add_term(
            [(u_idx(0,0), 1)],
            [(v_idx(0), 1)],
            [(w_idx(0), 1)]
        )
        
        # M2: A01 B1 -> C0
        add_term(
            [(u_idx(0,1), 1)],
            [(v_idx(1), 1)],
            [(w_idx(0), 1)]
        )
        
        # M3: A10 B0 -> C1
        add_term(
            [(u_idx(1,0), 1)],
            [(v_idx(0), 1)],
            [(w_idx(1), 1)]
        )
        
        # M4: A11 B1 -> C1
        add_term(
            [(u_idx(1,1), 1)],
            [(v_idx(1), 1)],
            [(w_idx(1), 1)]
        )

    # Process A1 (cols 0,1) with B1 (rows 0,1)
    # B1 cols: 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6 (Single)
    strassen_block(0, 0, 0)
    strassen_block(0, 0, 2)
    strassen_block(0, 0, 4)
    single_block(0, 0, 6)
    
    # Process A2 (cols 2,3) with B2 (rows 2,3)
    # B2 cols: 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6 (Single)
    strassen_block(2, 2, 0)
    strassen_block(2, 2, 2)
    strassen_block(2, 2, 4)
    single_block(2, 2, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
