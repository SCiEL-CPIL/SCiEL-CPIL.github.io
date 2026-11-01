import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(j_base, k_base):
        nonlocal col
        # Indices
        u_00 = 0 * m + j_base
        u_01 = 0 * m + j_base + 1
        u_10 = 1 * m + j_base
        u_11 = 1 * m + j_base + 1
        
        v_00 = j_base * p + k_base
        v_01 = j_base * p + k_base + 1
        v_10 = (j_base + 1) * p + k_base
        v_11 = (j_base + 1) * p + k_base + 1
        
        w_00 = k_base * n + 0
        w_01 = (k_base + 1) * n + 0
        w_10 = k_base * n + 1
        w_11 = (k_base + 1) * n + 1
        
        # M1: (A00 + A11)(B00 + B11)
        U[u_00][col] = 1
        U[u_11][col] = 1
        V[v_00][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M2: (A10 + A11)B00
        U[u_10][col] = 1
        U[u_11][col] = 1
        V[v_00][col] = 1
        W[w_10][col] = 1
        W[w_11][col] = -1
        col += 1
        
        # M3: A00 B11
        U[u_00][col] = 1
        V[v_11][col] = 1
        W[w_01][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M4: A11 (B01 - B11)
        U[u_11][col] = 1
        V[v_01][col] = 1
        V[v_11][col] = -1
        W[w_00][col] = 1
        col += 1
        
        # M5: (A00 + A01)B11
        U[u_00][col] = 1
        U[u_01][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = -1
        W[w_01][col] = 1
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[u_10][col] = 1
        U[u_00][col] = -1
        V[v_00][col] = 1
        V[v_01][col] = 1
        W[w_10][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M7: (A01 - A11)(B10 + B11)
        U[u_01][col] = 1
        U[u_11][col] = -1
        V[v_10][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1
        col += 1

    def add_schoolbook_block(j_base, k_base):
        nonlocal col
        # Indices
        u_00 = 0 * m + j_base
        u_01 = 0 * m + j_base + 1
        u_10 = 1 * m + j_base
        u_11 = 1 * m + j_base + 1
        
        v_0k = j_base * p + k_base
        v_1k = (j_base + 1) * p + k_base
        
        w_0k = k_base * n + 0
        w_1k = k_base * n + 1
        
        # P1: A00 B0k
        U[u_00][col] = 1
        V[v_0k][col] = 1
        W[w_0k][col] = 1
        col += 1
        
        # P2: A01 B0k
        U[u_01][col] = 1
        V[v_0k][col] = 1
        W[w_0k][col] = 1
        col += 1
        
        # P3: A10 B1k
        U[u_10][col] = 1
        V[v_1k][col] = 1
        W[w_1k][col] = 1
        col += 1
        
        # P4: A11 B1k
        U[u_11][col] = 1
        V[v_1k][col] = 1
        W[w_1k][col] = 1
        col += 1

    # Part 1: A_L (j=0,1) * B_T (j=0,1)
    # k=0,1
    add_strassen_block(0, 0)
    # k=2,3
    add_strassen_block(0, 2)
    # k=4,5
    add_strassen_block(0, 4)
    # k=6
    add_schoolbook_block(0, 6)
    
    # Part 2: A_R (j=2,3) * B_B (j=2,3)
    # k=0,1
    add_strassen_block(2, 0)
    # k=2,3
    add_strassen_block(2, 2)
    # k=4,5
    add_strassen_block(2, 4)
    # k=6
    add_schoolbook_block(2, 6)
    
    return {"U": U, "V": V, "W": W}
