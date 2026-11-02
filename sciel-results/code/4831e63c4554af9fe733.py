"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Winograd Fourier-based approach:
    # Decompose the 2x4x5 multiplication tensor using polynomial interpolation.
    # 
    # Strategy:
    # - Split the 4x5 matrix B into blocks that can be handled by low-rank sub-tensors
    # - Use the structure of polynomial evaluation to create symmetric sub-tensors
    # - The key insight is that for small dimensions, we can use FFT-like decompositions
    #
    # For 2x4x5:
    # - Split A (2x4) into two 2x2 blocks: A1 (cols 0,1), A2 (cols 2,3)
    # - Split B (4x5) into two 2x5 blocks: B1 (rows 0,1), B2 (rows 2,3)
    # - C = A1*B1 + A2*B2
    #
    # For each 2x2 * 2x5 block:
    # - Further split B_block (2x5) into B_block[:,0:2], B_block[:,2:4], B_block[:,4:5]
    # - Use rank-7 Strassen for 2x2*2x2 parts
    # - Use rank-4 schoolbook for 2x2*2x1 part
    # Total per 2x2*2x5: 7 + 7 + 4 = 18
    # Total rank: 2 * 18 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    # Strassen for 2x2 * 2x2 -> 2x2
    def add_strassen_2x2(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i
        
        # M1 = (A00+A11)(B00+B11)
        u = {map_a(0): 1, map_a(3): 1}
        v = {map_b(0): 1, map_b(3): 1}
        w = {map_c(0): 1, map_c(3): 1}
        add_rank1_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = {map_a(2): 1, map_a(3): 1}
        v = {map_b(0): 1}
        w = {map_c(2): 1, map_c(3): -1}
        add_rank1_term(u, v, w)
        
        # M3 = A00(B01-B11)
        u = {map_a(0): 1}
        v = {map_b(1): 1, map_b(3): -1}
        w = {map_c(1): 1, map_c(3): 1}
        add_rank1_term(u, v, w)
        
        # M4 = A11(B10-B00)
        u = {map_a(3): 1}
        v = {map_b(2): 1, map_b(0): -1}
        w = {map_c(0): 1, map_c(2): 1}
        add_rank1_term(u, v, w)
        
        # M5 = (A00+A01)B11
        u = {map_a(0): 1, map_a(1): 1}
        v = {map_b(3): 1}
        w = {map_c(0): -1, map_c(1): 1}
        add_rank1_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = {map_a(2): 1, map_a(0): -1}
        v = {map_b(0): 1, map_b(1): 1}
        w = {map_c(3): 1}
        add_rank1_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = {map_a(1): 1, map_a(3): -1}
        v = {map_b(2): 1, map_b(3): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)
    
    # Schoolbook for 2x2 * 2x1 -> 2x1
    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i
        
        for l in range(4):
            a_local = l
            b_local = l % 2
            c_local = l // 2
            u = {map_a(a_local): 1}
            v = {map_b(b_local): 1}
            w = {map_c(c_local): 1}
            add_rank1_term(u, v, w)
    
    # Block 1: A1 (rows 0,1; cols 0,1), B1 (rows 0,1; cols 0,4)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)      # A1 * B1[:,0:2]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)      # A1 * B1[:,2:4]
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)  # A1 * B1[:,4:5]
    
    # Block 2: A2 (rows 0,1; cols 2,3), B2 (rows 2,3; cols 0,4)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)      # A2 * B2[:,0:2]
    add_strassen_2x2(0, 2, 2, 2, 0, 2)      # A2 * B2[:,2:4]
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)  # A2 * B2[:,4:5]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
