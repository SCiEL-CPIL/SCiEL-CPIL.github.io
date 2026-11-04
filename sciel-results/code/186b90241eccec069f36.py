"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 block
    # L_A and L_B are 2x2 matrices of coefficients for the 7 products
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-g)
    # M4: d(g+h)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b+d)(g+h)
    
    L_A = [
        [[1, 0], [1, 1]], # M1: a+d
        [[0, 1], [1, 1]], # M2: c+d
        [[1, 0], [0, 0]], # M3: a
        [[0, 0], [0, 1]], # M4: d
        [[1, 1], [0, 0]], # M5: a+b
        [[-1, 0], [1, 0]],# M6: c-a
        [[0, 1], [1, 1]]  # M7: b+d
    ]
    L_B = [
        [[1, 1], [1, 1]], # M1: e+h
        [[1, 0], [0, 0]], # M2: e
        [[0, 1], [1, -1]],# M3: f-g
        [[0, 1], [1, 1]], # M4: g+h
        [[0, 0], [0, 1]], # M5: h
        [[1, 1], [0, 0]], # M6: e+f
        [[1, 1], [1, 1]]  # M7: g+h
    ]
    
    # Reconstruction coefficients c_ik_l
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, -1], # i=0, k=0
        [0, 0, 1, 0, 1, 0, 1],   # i=0, k=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1, k=0
        [1, -1, 1, 0, 0, 1, 0]   # i=1, k=1
    ]

    def add_strassen_block(I_start, J_start, K_start):
        nonlocal col
        for l in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (I_start + i) * m + (J_start + j)
                    U[u_idx][col] = L_A[l][i][j]
            for j in range(2):
                for k in range(2):
                    v_idx = (J_start + j) * p + (K_start + k)
                    V[v_idx][col] = L_B[l][j][k]
            for k in range(2):
                for i in range(2):
                    w_idx = (K_start + k) * n + (I_start + i)
                    W[w_idx][col] = C_coeffs[i][k][l]
            col += 1

    # Block 1: C[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    
    # Block 2: C[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    
    # Block 3: C[0:2, 4:5] (2x4x1)
    # Use 2x2x1 schoolbook blocks
    def add_2x2x1_block(I_start, J_start, K_start):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    u_idx = (I_start + i) * m + (J_start + j)
                    v_idx = (J_start + j) * p + (K_start + k)
                    w_idx = (K_start + k) * n + (I_start + i)
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

    add_2x2x1_block(0, 0, 4)
    add_2x2x1_block(0, 2, 4)
    
    # Block 4: C[2:3, 0:5] (1x4x5)
    # Schoolbook
    for i in range(1):
        for j in range(4):
            for k in range(5):
                u_idx = (2 + i) * m + j
                v_idx = j * p + k
                w_idx = k * n + (2 + i)
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
