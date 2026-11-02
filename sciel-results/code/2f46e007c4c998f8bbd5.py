"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition using two Strassen blocks
    rank = 14
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U rows: 0:A00, 1:A01, 2:A10, 3:A11
    # V rows (block 1): 0:B00, 1:B01, 2:B10, 3:B11
    # V rows (block 2): 4:B02, 5:B03, 6:B12, 7:B13
    # W indices (block 1): 0:C00, 1:C10, 2:C01, 3:C11
    # W indices (block 2): 4:C02, 5:C12, 6:C03, 7:C13
    
    # Define the 7 Strassen terms for one 2x2x2 block
    # Each term is (U_vec, V_vec, W_vec)
    # W_vec has 4 elements corresponding to C00, C10, C01, C11
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'U': [1, 0, 0, 1],
            'V': [1, 0, 0, 1],
            'W': [1, 0, 0, 1] # C00=1, C10=0, C01=0, C11=1
        },
        # M2 = (A10+A11)B00
        {
            'U': [0, 0, 1, 1],
            'V': [1, 0, 0, 0],
            'W': [0, 1, 0, -1] # C00=0, C10=1, C01=0, C11=-1
        },
        # M3 = A00(B01-B11)
        {
            'U': [1, 0, 0, 0],
            'V': [0, 1, 0, -1],
            'W': [0, 0, 1, 1] # C00=0, C10=0, C01=1, C11=1
        },
        # M4 = A11(B10-B00)
        {
            'U': [0, 0, 0, 1],
            'V': [-1, 0, 1, 0],
            'W': [-1, 0, 1, 0] # C00=-1, C10=0, C01=1, C11=0
        },
        # M5 = (A00+A01)B11
        {
            'U': [1, 1, 0, 0],
            'V': [0, 0, 0, 1],
            'W': [1, 0, 0, 0] # C00=1, C10=0, C01=0, C11=0
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'U': [-1, 0, 1, 0],
            'V': [1, 1, 0, 0],
            'W': [0, 0, 0, 1] # C00=0, C10=0, C01=0, C11=1
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'U': [0, 1, 0, -1],
            'V': [0, 0, 1, 1],
            'W': [-1, 1, 0, 0] # C00=-1, C10=1, C01=0, C11=0
        }
    ]
    
    # Fill in the matrices for block 1 (columns 0,1 of B)
    for r in range(7):
        term = strassen_terms[r]
        for i in range(4):
            U[i][r] = term['U'][i]
            V[i][r] = term['V'][i]
        # W mapping: C00->0, C10->1, C01->2, C11->3
        W[0][r] = term['W'][0]
        W[1][r] = term['W'][1]
        W[2][r] = term['W'][2]
        W[3][r] = term['W'][3]
        
    # Fill in the matrices for block 2 (columns 2,3 of B)
    for r in range(7):
        term = strassen_terms[r]
        col_offset = 7
        r2 = r + col_offset
        for i in range(4):
            U[i][r2] = term['U'][i]
            # V indices for block 2 are 4,5,6,7
            V[i+4][r2] = term['V'][i]
        # W mapping: C02->4, C12->5, C03->6, C13->7
        W[4][r2] = term['W'][0]
        W[5][r2] = term['W'][1]
        W[6][r2] = term['W'][2]
        W[7][r2] = term['W'][3]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
