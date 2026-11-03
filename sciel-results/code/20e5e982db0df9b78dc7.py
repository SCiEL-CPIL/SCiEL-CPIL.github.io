"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Decomposition of rank 36 using Strassen + schoolbook blocks.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
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

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        # M1 = (A00+A11)(B00+B11)
        add_rank1_term(
            {map_a(0): 1, map_a(3): 1},
            {map_b(0): 1, map_b(3): 1},
            {map_c(0): 1, map_c(3): 1}
        )
        # M2 = (A10+A11)B00
        add_rank1_term(
            {map_a(2): 1, map_a(3): 1},
            {map_b(0): 1},
            {map_c(2): 1, map_c(3): -1}
        )
        # M3 = A00(B01-B11)
        add_rank1_term(
            {map_a(0): 1},
            {map_b(1): 1, map_b(3): -1},
            {map_c(1): 1, map_c(3): 1}
        )
        # M4 = A11(B10-B00)
        add_rank1_term(
            {map_a(3): 1},
            {map_b(2): 1, map_b(0): -1},
            {map_c(0): 1, map_c(2): 1}
        )
        # M5 = (A00+A01)B11
        add_rank1_term(
            {map_a(0): 1, map_a(1): 1},
            {map_b(3): 1},
            {map_c(0): -1, map_c(1): 1}
        )
        # M6 = (A10-A00)(B00+B01)
        add_rank1_term(
            {map_a(2): 1, map_a(0): -1},
            {map_b(0): 1, map_b(1): 1},
            {map_c(3): 1}
        )
        # M7 = (A01-A11)(B10+B11)
        add_rank1_term(
            {map_a(1): 1, map_a(3): -1},
            {map_b(2): 1, map_b(3): 1},
            {map_c(0): 1}
        )

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
            add_rank1_term(
                {map_a(a_local): 1},
                {map_b(b_local): 1},
                {map_c(c_local): 1}
            )

    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
