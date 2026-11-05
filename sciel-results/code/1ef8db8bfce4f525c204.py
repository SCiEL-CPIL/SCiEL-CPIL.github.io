DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 46
    # Term 1 (A1, B1): 7 (Strassen k=0,1) + 7 (Strassen k=2,3) + 4 (Schoolbook k=4) = 18
    # Term 2 (A2, B2): 7 (Strassen k=0,1) + 7 (Strassen k=2,3) + 4 (Schoolbook k=4) = 18
    # Term 3 (A3, B3): 10 (Schoolbook)
    # Total: 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        U[i_off * m + j_off][col] += 1
        U[(i_off+1) * m + (j_off+1)][col] += 1
        # V: B00=1, B11=1
        V[j_off * p + k_off][col] += 1
        V[(j_off+1) * p + (k_off+1)][col] += 1
        # W: C00+=1, C11+=1
        W[k_off * n + i_off][col] += 1
        W[(k_off+1) * n + (i_off+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_off+1) * m + j_off][col] += 1
        U[(i_off+1) * m + (j_off+1)][col] += 1
        V[j_off * p + k_off][col] += 1
        W[(k_off+1) * n + i_off][col] += 1 # C01
        col += 1
        
        # M3 = A00(B01-B11)
        U[i_off * m + j_off][col] += 1
        V[j_off * p + (k_off+1)][col] += 1
        V[(j_off+1) * p + (k_off+1)][col] -= 1
        W[k_off * n + (i_off+1)][col] += 1 # C10
        W[(k_off+1) * n + (i_off+1)][col] -= 1 # C11
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i_off+1) * m + (j_off+1)][col] += 1
        V[(j_off+1) * p + k_off][col] += 1
        V[j_off * p + k_off][col] -= 1
        W[k_off * n + i_off][col] += 1 # C00
        W[(k_off+1) * n + i_off][col] += 1 # C01
        W[(k_off+1) * n + (i_off+1)][col] += 1 # C11
        col += 1
        
        # M5 = (A00+A01)B11
        U[i_off * m + j_off][col] += 1
        U[i_off * m + (j_off+1)][col] += 1
        V[(j_off+1) * p + (k_off+1)][col] += 1
        W[k_off * n + i_off][col] -= 1 # C00
        W[k_off * n + (i_off+1)][col] += 1 # C10
        col += 1
        
        # M6 = (A10-A01)(B00+B01)
        U[(i_off+1) * m + j_off][col] += 1
        U[i_off * m + (j_off+1)][col] -= 1
        V[j_off * p + k_off][col] += 1
        V[j_off * p + (k_off+1)][col] += 1
        W[(k_off+1) * n + (i_off+1)][col] += 1 # C11
        col += 1
        
        # M7 = (A00-A11)(B10+B11)
        U[i_off * m + j_off][col] += 1
        U[(i_off+1) * m + (j_off+1)][col] -= 1
        V[(j_off+1) * p + k_off][col] += 1
        V[(j_off+1) * p + (k_off+1)][col] += 1
        W[k_off * n + i_off][col] += 1 # C00
        col += 1

    # Helper to add Schoolbook block for 2x2x1 (A is 2x2, B is 2x1)
    def add_schoolbook_2x2x1(i_off, j_off, k_off):
        nonlocal col
        # C00 = A00 B00 + A01 B10
        U[i_off * m + j_off][col] += 1
        V[j_off * p + k_off][col] += 1
        W[k_off * n + i_off][col] += 1
        col += 1
        
        U[i_off * m + (j_off+1)][col] += 1
        V[(j_off+1) * p + k_off][col] += 1
        W[k_off * n + i_off][col] += 1
        col += 1
        
        # C10 = A10 B00 + A11 B10
        U[(i_off+1) * m + j_off][col] += 1
        V[j_off * p + k_off][col] += 1
        W[k_off * n + (i_off+1)][col] += 1
        col += 1
        
        U[(i_off+1) * m + (j_off+1)][col] += 1
        V[(j_off+1) * p + k_off][col] += 1
        W[k_off * n + (i_off+1)][col] += 1
        col += 1

    # Helper to add Schoolbook block for 2x1x5 (A is 2x1, B is 1x5)
    def add_schoolbook_2x1x5(i_off, j_off, k_off):
        nonlocal col
        # A is 2x1, so j is fixed at j_off.
        # B is 1x5, so j is fixed at j_off. k varies.
        # C is 2x5.
        for i in range(2):
            for k in range(5):
                U[(i_off+i) * m + j_off][col] += 1
                V[j_off * p + (k_off+k)][col] += 1
                W[(k_off+k) * n + (i_off+i)][col] += 1
                col += 1

    # Term 1: A1 (i=0,1, j=0,1) * B1 (j=0,1, k=0..4)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook_2x2x1(0, 0, 4)
    
    # Term 2: A2 (i=0,1, j=2,3) * B2 (j=2,3, k=0..4)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook_2x2x1(0, 2, 4)
    
    # Term 3: A3 (i=0,1, j=4) * B3 (j=4, k=0..4)
    add_schoolbook_2x1x5(0, 4, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
