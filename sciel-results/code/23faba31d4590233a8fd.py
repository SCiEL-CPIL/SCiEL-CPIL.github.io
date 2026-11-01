import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x2 matrix by a 2x7 matrix.
    
    We use a block structure:
    - Split the 7 columns into three 2-column blocks and one 1-column block.
    - For each 2-column block, use Strassen's algorithm for 2x2x2 (rank 7).
    - For the 1-column block, use schoolbook (rank 4).
    - Total rank = 7*3 + 4 = 25.
    
    The decomposition follows the convention:
    U: (n*m, rank) = (4, 25)
    V: (m*p, rank) = (14, 25)
    W: (p*n, rank) = (14, 25)
    
    Indexing:
    U[i*m + j, c] = coefficient for A[i,j] in rank component c
    V[j*p + k, c] = coefficient for B[j,k] in rank component c
    W[k*n + i, c] = coefficient for C[k,i] in rank component c
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 25
    V = [[0] * rank for _ in range(m * p)]  # 14 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    # Strassen's 7 terms for 2x2x2
    # U terms: coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
    # V terms: coefficients for B[0,0], B[0,1], B[1,0], B[1,1] (within a 2-column block)
    # W terms: coefficients for C[0,0], C[0,1], C[1,0], C[1,1] (within a 2-column block)
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: (A00 + A11)
        [0, 0, 1, 1],   # M2: (A10 + A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00 + A01)
        [1, 0, 1, 0],   # M6: (A10 - A00)
        [0, 1, 0, -1],  # M7: (A01 - A11)
    ]
    
    strassen_V = [
        [1, 1, 0, 0],   # M1: (B00 + B11)
        [1, 0, 0, 0],   # M2: B00
        [0, -1, 1, 0],  # M3: (B10 - B11)
        [1, 0, -1, 0],  # M4: (B01 - B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00 + B01)
        [0, 0, 1, 1],   # M7: (B10 + B11)
    ]
    
    strassen_W = [
        [1, 0, 0, -1],  # C00: M1 - M4 + M5 - M7
        [0, 1, 1, 0],   # C01: M3 + M5
        [0, 1, 1, 0],   # C10: M2 + M4
        [-1, -1, 1, 1], # C11: M1 - M2 + M3 + M6
    ]
    
    col = 0
    
    # Three 2-column blocks: columns (0,1), (2,3), (4,5)
    for block in range(3):
        k1 = 2 * block
        k2 = 2 * block + 1
        
        for t in range(7):
            # U: add strassen_U[t] to U rows 0-3
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] += strassen_U[t][i * m + j]
            
            # V: add strassen_V[t] to V rows for column k1 and k2
            # strassen_V[t] = [B[0,k1], B[0,k2], B[1,k1], B[1,k2]]
            V[0 * p + k1][col] += strassen_V[t][0]
            V[0 * p + k2][col] += strassen_V[t][1]
            V[1 * p + k1][col] += strassen_V[t][2]
            V[1 * p + k2][col] += strassen_V[t][3]
            
            # W: add strassen_W[t] to W rows for columns k1 and k2
            # strassen_W[t] = [C[k1,0], C[k1,1], C[k2,0], C[k2,1]]
            W[k1 * n + 0][col] += strassen_W[t][0]
            W[k1 * n + 1][col] += strassen_W[t][1]
            W[k2 * n + 0][col] += strassen_W[t][2]
            W[k2 * n + 1][col] += strassen_W[t][3]
            
            col += 1
    
    # One 1-column block: column 6, schoolbook rank 4
    k = 6
    schoolbook_terms = [
        # (U_row, V_row, W_row) for each schoolbook term
        (0 * m + 0, 0 * p + k, k * n + 0),  # A[0,0] * B[0,6] -> C[6,0]
        (0 * m + 1, 1 * p + k, k * n + 0),  # A[0,1] * B[1,6] -> C[6,0]
        (1 * m + 0, 0 * p + k, k * n + 1),  # A[1,0] * B[0,6] -> C[6,1]
        (1 * m + 1, 1 * p + k, k * n + 1),  # A[1,1] * B[1,6] -> C[6,1]
    ]
    
    for u_row, v_row, w_row in schoolbook_terms:
        U[u_row][col] = 1
        V[v_row][col] = 1
        W[w_row][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
