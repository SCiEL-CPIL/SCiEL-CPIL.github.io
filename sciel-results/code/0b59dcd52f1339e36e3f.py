"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(j_off, k_off, start_col):
        nonlocal col
        # j_off: 0 or 2
        # k_off: 0 or 2
        
        # Term 1: (x00+x11)(y00+y11)
        U[0 * m + (j_off + 0)][start_col] += 1
        U[1 * m + (j_off + 1)][start_col] += 1
        V[(j_off + 0) * p + (k_off + 0)][start_col] += 1
        V[(j_off + 1) * p + (k_off + 1)][start_col] += 1
        W[(k_off + 0) * n + 0][start_col] += 1
        W[(k_off + 1) * n + 1][start_col] += 1
        col += 1
        
        # Term 2: (x10+x11)y00
        U[1 * m + (j_off + 0)][start_col] += 1
        U[1 * m + (j_off + 1)][start_col] += 1
        V[(j_off + 0) * p + (k_off + 0)][start_col] += 1
        W[(k_off + 0) * n + 1][start_col] += 1
        W[(k_off + 1) * n + 1][start_col] += 1
        col += 1
        
        # Term 3: x00(y01-y11)
        U[0 * m + (j_off + 0)][start_col] += 1
        V[(j_off + 0) * p + (k_off + 1)][start_col] += 1
        V[(j_off + 1) * p + (k_off + 1)][start_col] += -1
        W[(k_off + 1) * n + 0][start_col] += 1
        W[(k_off + 1) * n + 1][start_col] += -1
        col += 1
        
        # Term 4: x11(y10-y00)
        U[1 * m + (j_off + 1)][start_col] += 1
        V[(j_off + 1) * p + (k_off + 0)][start_col] += 1
        V[(j_off + 0) * p + (k_off + 0)][start_col] += -1
        W[(k_off + 0) * n + 0][start_col] += -1
        W[(k_off + 0) * n + 1][start_col] += 1
        col += 1
        
        # Term 5: (x00+x01)y11
        U[0 * m + (j_off + 0)][start_col] += 1
        U[0 * m + (j_off + 1)][start_col] += 1
        V[(j_off + 1) * p + (k_off + 1)][start_col] += 1
        W[(k_off + 1) * n + 0][start_col] += 1
        W[(k_off + 1) * n + 1][start_col] += -1
        col += 1
        
        # Term 6: (x10-x00)(y00+y01)
        U[1 * m + (j_off + 0)][start_col] += 1
        U[0 * m + (j_off + 0)][start_col] += -1
        V[(j_off + 0) * p + (k_off + 0)][start_col] += 1
        V[(j_off + 0) * p + (k_off + 1)][start_col] += 1
        W[(k_off + 0) * n + 0][start_col] += 1
        W[(k_off + 0) * n + 1][start_col] += -1
        col += 1
        
        # Term 7: (x01-x11)(y10+y11)
        U[0 * m + (j_off + 1)][start_col] += 1
        U[1 * m + (j_off + 1)][start_col] += -1
        V[(j_off + 1) * p + (k_off + 0)][start_col] += 1
        V[(j_off + 1) * p + (k_off + 1)][start_col] += 1
        W[(k_off + 0) * n + 0][start_col] += -1
        W[(k_off + 1) * n + 0][start_col] += 1
        col += 1

    def add_schoolbook_221(j_off, k_val, start_col):
        nonlocal col
        # j_off: 0 or 2
        # k_val: 4
        
        # 1. A[0, j_off] * B[j_off, k_val] -> C[0, k_val]
        U[0 * m + (j_off + 0)][start_col] += 1
        V[(j_off + 0) * p + k_val][start_col] += 1
        W[k_val * n + 0][start_col] += 1
        col += 1
        
        # 2. A[0, j_off+1] * B[j_off+1, k_val] -> C[0, k_val]
        U[0 * m + (j_off + 1)][start_col] += 1
        V[(j_off + 1) * p + k_val][start_col] += 1
        W[k_val * n + 0][start_col] += 1
        col += 1
        
        # 3. A[1, j_off] * B[j_off, k_val] -> C[1, k_val]
        U[1 * m + (j_off + 0)][start_col] += 1
        V[(j_off + 0) * p + k_val][start_col] += 1
        W[k_val * n + 1][start_col] += 1
        col += 1
        
        # 4. A[1, j_off+1] * B[j_off+1, k_val] -> C[1, k_val]
        U[1 * m + (j_off + 1)][start_col] += 1
        V[(j_off + 1) * p + k_val][start_col] += 1
        W[k_val * n + 1][start_col] += 1
        col += 1

    # Term 1: A_L B_T
    # A_L: j in {0,1}. j_off=0.
    # B_T: j in {0,1}. j_off=0.
    
    # B_T T_1: k in {0,1}. k_off=0.
    add_strassen(0, 0, col)
    
    # B_T T_2: k in {2,3}. k_off=2.
    add_strassen(0, 2, col)
    
    # B_T T_3: k in {4}. k_val=4.
    add_schoolbook_221(0, 4, col)
    
    # Term 2: A_R B_B
    # A_R: j in {2,3}. j_off=2.
    # B_B: j in {2,3}. j_off=2.
    
    # B_B U_1: k in {0,1}. k_off=0.
    add_strassen(2, 0, col)
    
    # B_B U_2: k in {2,3}. k_off=2.
    add_strassen(2, 2, col)
    
    # B_B U_3: k in {4}. k_val=4.
    add_schoolbook_221(2, 4, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
