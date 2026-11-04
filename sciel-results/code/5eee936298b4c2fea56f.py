"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Max possible rank is 60. We use 28.
    U = [[0] * 60 for _ in range(n * m)]
    V = [[0] * 60 for _ in range(m * p)]
    W = [[0] * 60 for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx):
        nonlocal col
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # Block 1: 2x2x3 (m=0..1, k=0..2)
    # Split k=0 and k=1..2
    # 2x2x1 (k=0): Rank 2
    # C_i = A_i0 B0 + A_i1 B1
    # Term 1: A_i0 B0
    for i in range(2):
        add_term(i*m + 0, 0*p + 0, 0*n + i)
    # Term 2: A_i1 B1
    for i in range(2):
        add_term(i*m + 1, 1*p + 0, 0*n + i)
        
    # 2x2x2 (k=1..2): Rank 4
    # Schoolbook
    for i in range(2):
        for j in range(2):
            for k in range(2):
                k_abs = 1 + k
                add_term(i*m + j, j*p + k_abs, k_abs*n + i)

    # Block 2: 2x2x3 (m=0..1, k=3..5)
    # Split k=3 and k=4..5
    # 2x2x1 (k=3): Rank 2
    for i in range(2):
        add_term(i*m + 0, 0*p + 3, 3*n + i)
    for i in range(2):
        add_term(i*m + 1, 1*p + 3, 3*n + i)
        
    # 2x2x2 (k=4..5): Rank 4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                k_abs = 4 + k
                add_term(i*m + j, j*p + k_abs, k_abs*n + i)

    # Block 3: 2x3x3 (m=2..4, k=0..2)
    # Split m=2 and m=3..4
    # Split k=0 and k=1..2
    
    # 2x1x1 (m=2, k=0): Rank 1
    # C_i = A_i2 B20
    for i in range(2):
        add_term(i*m + 2, 2*p + 0, 0*n + i)
        
    # 2x1x2 (m=2, k=1..2): Rank 1
    # C_i,k = A_i2 B2k
    for i in range(2):
        for k in range(2):
            k_abs = 1 + k
            add_term(i*m + 2, 2*p + k_abs, k_abs*n + i)
            
    # 2x2x1 (m=3..4, k=0): Rank 2
    # C_i = A_i3 B30 + A_i4 B40
    # Term 1: A_i3 B30
    for i in range(2):
        add_term(i*m + 3, 3*p + 0, 0*n + i)
    # Term 2: A_i4 B40
    for i in range(2):
        add_term(i*m + 4, 4*p + 0, 0*n + i)

    # 2x2x2 (m=3..4, k=1..2): Rank 4
    for i in range(2):
        for j in range(2):
            m_abs = 3 + j
            for k in range(2):
                k_abs = 1 + k
                add_term(i*m + m_abs, m_abs*p + k_abs, k_abs*n + i)

    # Block 4: 2x3x3 (m=2..4, k=3..5)
    # Split m=2 and m=3..4
    # Split k=3 and k=4..5
    
    # 2x1x1 (m=2, k=3): Rank 1
    for i in range(2):
        add_term(i*m + 2, 2*p + 3, 3*n + i)
        
    # 2x1x2 (m=2, k=4..5): Rank 1
    for i in range(2):
        for k in range(2):
            k_abs = 4 + k
            add_term(i*m + 2, 2*p + k_abs, k_abs*n + i)
            
    # 2x2x1 (m=3..4, k=3): Rank 2
    for i in range(2):
        add_term(i*m + 3, 3*p + 3, 3*n + i)
    for i in range(2):
        add_term(i*m + 4, 4*p + 3, 3*n + i)

    # 2x2x2 (m=3..4, k=4..5): Rank 4
    for i in range(2):
        for j in range(2):
            m_abs = 3 + j
            for k in range(2):
                k_abs = 4 + k
                add_term(i*m + m_abs, m_abs*p + k_abs, k_abs*n + i)

    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
