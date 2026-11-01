"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (6, 28)
    V = [[0] * rank for _ in range(m * p)]  # (15, 28)
    W = [[0] * rank for _ in range(p * n)]  # (10, 28)
    
    col = 0
    
    # Helper: add a Strassen decomposition for 2x2 * 2x2
    def add_strassen(a_indices, b_indices, w_indices):
        # a_indices: [a1, a2, a3, a4]
        # b_indices: [b1, b2, b3, b4]
        # w_indices: [[w11, w12], [w21, w22]]
        global col
        # P1 = (a1+a4)(b1+b4) -> +1 to C11, +1 to C22
        U[a_indices[0]][col] += 1
        U[a_indices[3]][col] += 1
        V[b_indices[0]][col] += 1
        V[b_indices[3]][col] += 1
        W[w_indices[0][0]][col] += 1
        W[w_indices[1][1]][col] += 1
        col += 1
        
        # P2 = (a3+a4)b1 -> +1 to C21, -1 to C22
        U[a_indices[2]][col] += 1
        U[a_indices[3]][col] += 1
        V[b_indices[0]][col] += 1
        W[w_indices[1][0]][col] += 1
        W[w_indices[1][1]][col] -= 1
        col += 1
        
        # P3 = a1(b2-b4) -> +1 to C12, +1 to C22
        U[a_indices[0]][col] += 1
        V[b_indices[1]][col] += 1
        V[b_indices[3]][col] -= 1
        W[w_indices[0][1]][col] += 1
        W[w_indices[1][1]][col] += 1
        col += 1
        
        # P4 = a4(b3-b1) -> +1 to C11, +1 to C21
        U[a_indices[3]][col] += 1
        V[b_indices[2]][col] += 1
        V[b_indices[0]][col] -= 1
        W[w_indices[0][0]][col] += 1
        W[w_indices[1][0]][col] += 1
        col += 1
        
        # P5 = (a1+a2)b4 -> -1 to C11, +1 to C12
        U[a_indices[0]][col] += 1
        U[a_indices[1]][col] += 1
        V[b_indices[3]][col] += 1
        W[w_indices[0][0]][col] -= 1
        W[w_indices[0][1]][col] += 1
        col += 1
        
        # P6 = (a3-a1)(b1+b2) -> +1 to C22
        U[a_indices[2]][col] += 1
        U[a_indices[0]][col] -= 1
        V[b_indices[0]][col] += 1
        V[b_indices[1]][col] += 1
        W[w_indices[1][1]][col] += 1
        col += 1
        
        # P7 = (a2-a4)(b3+b4) -> +1 to C11
        U[a_indices[1]][col] += 1
        U[a_indices[3]][col] -= 1
        V[b_indices[2]][col] += 1
        V[b_indices[3]][col] += 1
        W[w_indices[0][0]][col] += 1
        col += 1
    
    # Helper: add schoolbook for 2x2 * 2x1
    def add_schoolbook_2x2_2x1(a_indices, b_indices, w_indices):
        global col
        U[a_indices[0]][col] += 1
        V[b_indices[0]][col] += 1
        W[w_indices[0]][col] += 1
        col += 1
        U[a_indices[1]][col] += 1
        V[b_indices[1]][col] += 1
        W[w_indices[0]][col] += 1
        col += 1
        U[a_indices[2]][col] += 1
        V[b_indices[0]][col] += 1
        W[w_indices[1]][col] += 1
        col += 1
        U[a_indices[3]][col] += 1
        V[b_indices[1]][col] += 1
        W[w_indices[1]][col] += 1
        col += 1
    
    # Helper: add outer product for 2x1 * 1x5
    def add_outer_2x1_1x5(a_indices, b_indices, w_indices):
        global col
        for i in range(2):
            for j in range(5):
                U[a_indices[i]][col] += 1
                V[b_indices[j]][col] += 1
                W[w_indices[i][j]][col] += 1
                col += 1
    
    # Index mappings
    a1_strassen = [0, 1, 3, 4]  # A1: A[0,0], A[0,1], A[1,0], A[1,1]
    
    # Strassen for A1*B11
    b11 = [0, 1, 5, 6]  # B11: B[0,0], B[0,1], B[1,0], B[1,1]
    w11 = [[0, 2], [1, 3]]  # C[0,0], C[0,1], C[1,0], C[1,1]
    add_strassen(a1_strassen, b11, w11)
    
    # Strassen for A1*B12
    b12 = [2, 3, 7, 8]  # B12: B[0,2], B[0,3], B[1,2], B[1,3]
    w12 = [[4, 6], [5, 7]]  # C[0,2], C[0,3], C[1,2], C[1,3]
    add_strassen(a1_strassen, b12, w12)
    
    # Schoolbook for A1*B13
    b13 = [4, 9]  # B13: B[0,4], B[1,4]
    w13 = [8, 9]  # C[0,4], C[1,4]
    add_schoolbook_2x2_2x1(a1_strassen, b13, w13)
    
    # Outer product for A2*B2
    a2 = [2, 5]  # A2: A[0,2], A[1,2]
    b2 = [10, 11, 12, 13, 14]  # B2: B[2,0]..B[2,4]
    w_outer = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
    add_outer_2x1_1x5(a2, b2, w_outer)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
