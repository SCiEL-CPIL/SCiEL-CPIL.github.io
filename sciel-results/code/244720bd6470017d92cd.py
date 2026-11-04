"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Split m=5 into m1=2 and m2=3.
    # Split p=6 into three blocks of size 2.
    # Block (m1=2, p_block=2): 3 * R(2,2,2) = 3 * 7 = 21
    # Block (m2=3, p_block=2): 3 * R(3,2,2) = 3 * 12 = 36
    # Total Rank = 57
    
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_map, v_map, w_map):
        nonlocal col
        for (i, j), coeff in u_map.items():
            if coeff != 0:
                U[i * m + j][col] += coeff
        for (j, k), coeff in v_map.items():
            if coeff != 0:
                V[j * p + k][col] += coeff
        for (k, i), coeff in w_map.items():
            if coeff != 0:
                W[k * n + i][col] += coeff
        col += 1

    def strassen_2x2x2(i_off, j_off, k_off):
        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            {(i_off, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
            {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
        )
        
        # M2 = (A10 + A11)B00
        add_term(
            {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1},
            {(k_off, i_off+1): 1, (k_off+1, i_off+1): -1}
        )
        
        # M3 = A00(B10 - B11)
        add_term(
            {(i_off, j_off): 1},
            {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1},
            {(k_off, i_off+1): 1, (k_off+1, i_off+1): 1}
        )
        
        # M4 = A11(B01 - B00)
        add_term(
            {(i_off+1, j_off+1): 1},
            {(j_off, k_off+1): 1, (j_off, k_off): -1},
            {(k_off, i_off): 1, (k_off, i_off+1): 1}
        )
        
        # M5 = (A00 + A01)B11
        add_term(
            {(i_off, j_off): 1, (i_off, j_off+1): 1},
            {(j_off+1, k_off+1): 1},
            {(k_off, i_off): -1, (k_off+1, i_off+1): 1}
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            {(i_off+1, j_off): 1, (i_off, j_off): -1},
            {(j_off, k_off): 1, (j_off, k_off+1): 1},
            {(k_off+1, i_off+1): 1}
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
            {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
            {(k_off, i_off): 1}
        )

    def schoolbook_3x2x2(i_off, j_off, k_off):
        for di in range(3):
            for dj in range(2):
                for dk in range(2):
                    add_term(
                        {(i_off+di, j_off+dj): 1},
                        {(j_off+dj, k_off+dk): 1},
                        {(k_off+dk, i_off+di): 1}
                    )

    m_blocks = [(0, 2), (2, 3)]
    p_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for (m_start, m_size) in m_blocks:
        for (p_start, p_size) in p_blocks:
            if m_size == 2 and p_size == 2:
                strassen_2x2x2(0, m_start, p_start)
            else:
                schoolbook_3x2x2(0, m_start, p_start)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
