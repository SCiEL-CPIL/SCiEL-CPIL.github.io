"""Program for matrix multiplication 2x4 x 4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    The decomposition uses 6 independent 2x2x2 Strassen blocks, one for each
    (i-block, j-block, k-block) combination. Since n=2 (1 i-block), m=4 (2 j-blocks),
    p=6 (3 k-blocks), we have 1*2*3 = 6 blocks. Each Strassen block has rank 7,
    so the total rank is 6 * 7 = 42.
    
    The factor matrices are:
    - U: shape (n*m, rank) = (8, 42)
    - V: shape (m*p, rank) = (24, 42)
    - W: shape (p*n, rank) = (12, 42)
    
    Each column c corresponds to a term in the bilinear form:
    C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where j is determined by the column index.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # For matrices A (2x2), B (2x2), C (2x2):
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # Strassen's 7 products:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # Strassen coefficients for U, V, W
    # For each of the 7 products, we have coefficients for each of the 4 entries
    # in U (for A), V (for B), and W (for C)
    
    # U coefficients: for each product c (0-6), for each A entry (0-3)
    # A entry 0 = a11, entry 1 = a12, entry 2 = a21, entry 3 = a22
    U_strassen = [
        [1, 0, 0, 1],  # M1: (a11 + a22)
        [0, 0, 1, 1],  # M2: (a21 + a22)
        [1, 0, 0, 0],  # M3: a11
        [0, 0, 0, 1],  # M4: a22
        [1, 1, 0, 0],  # M5: (a11 + a12)
        [-1, 0, 1, 0], # M6: (a21 - a11)
        [0, 1, 0, -1]  # M7: (a12 - a22)
    ]
    
    # V coefficients: for each product c (0-6), for each B entry (0-3)
    # B entry 0 = b11, entry 1 = b12, entry 2 = b21, entry 3 = b22
    V_strassen = [
        [1, 0, 0, 1],  # M1: (b11 + b22)
        [1, 0, 0, 0],  # M2: b11
        [0, 1, 0, -1], # M3: (b12 - b22)
        [0, -1, 1, 0], # M4: (b21 - b11)
        [0, 0, 0, 1],  # M5: b22
        [1, 1, 0, 0],  # M6: (b11 + b12)
        [0, 0, 1, 1]   # M7: (b21 + b22)
    ]
    
    # W coefficients: for each product c (0-6), for each C entry (0-3)
    # C entry 0 = c11, entry 1 = c12, entry 2 = c21, entry 3 = c22
    W_strassen = [
        [1, 0, 0, -1],  # M1 contributes to C11, not C22
        [0, 0, 1, 0],   # M2 contributes to C21
        [0, 1, 0, 0],   # M3 contributes to C12
        [0, 0, 1, 0],   # M4 contributes to C21
        [1, 1, 0, 0],   # M5 contributes to C11 and C12
        [0, 0, 0, 1],   # M6 contributes to C22
        [1, 0, 0, 1]    # M7 contributes to C11 and C22
    ]
    
    # Now build the full decomposition
    # j-blocks: 0, 1 (each of size 2)
    # k-blocks: 0, 1, 2 (each of size 2)
    # i-block: 0 (size 2)
    
    col = 0
    for jb in range(m // 2):  # 2 j-blocks
        for kb in range(p // 2):  # 3 k-blocks
            for c in range(7):  # 7 Strassen products
                # U indices: i (0-1), j (jb*2 to jb*2+1)
                for i in range(2):
                    for j_local in range(2):
                        j = jb * 2 + j_local
                        U[i * m + j][col] += U_strassen[c][j_local]
                
                # V indices: j (jb*2 to jb*2+1), k (kb*2 to kb*2+1)
                for j_local in range(2):
                    for k_local in range(2):
                        j = jb * 2 + j_local
                        k = kb * 2 + k_local
                        V[j * p + k][col] += V_strassen[c][j_local * 2 + k_local]
                
                # W indices: k (kb*2 to kb*2+1), i (0-1)
                for k_local in range(2):
                    for i in range(2):
                        k = kb * 2 + k_local
                        W[k * n + i][col] += W_strassen[c][i * 2 + k_local]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
