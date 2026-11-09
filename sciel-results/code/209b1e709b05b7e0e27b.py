"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition of rank 56 for 2x4x8 matrix multiplication.
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We have 8 independent 2x2x2 sub-problems.
    # j in {0,1} and k_block in {0,1,2,3}
    # j_block in {0,1}
    
    # Indices:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    col = 0
    for j_block in range(2): # 0, 1
        j_start = j_block * 2
        for k_block in range(4): # 0, 1, 2, 3
            k_start = k_block * 2
            
            # 7 Strassen terms
            # 1
            # A: A[i, j] for i in 0,1 and j in j_start, j_start+1
            # B: B[j, k] for j in j_start, j_start+1 and k in k_start, k_start+1
            
            # M1
            # A: A00 + A11 -> U[0, j_start] = 1, U[1, j_start+1] = 1
            # B: B00 + B11 -> V[j_start, k_start] = 1, V[j_start+1, k_start+1] = 1
            # W: C00 += M1, C11 += M1
            U[0 * m + j_start][col] = 1
            U[1 * m + j_start + 1][col] = 1
            V[j_start * p + k_start][col] = 1
            V[(j_start + 1) * p + k_start + 1][col] = 1
            W[k_start * n + 0][col] = 1
            W[(k_start + 1) * n + 1][col] = 1
            col += 1
            
            # M2
            # A: A10 + A11
            # B: B00
            U[1 * m + j_start][col] = 1
            U[1 * m + j_start + 1][col] = 1
            V[j_start * p + k_start][col] = 1
            W[k_start * n + 1][col] = 1
            col += 1
            
            # M3
            # A: A00
            # B: B10 - B11
            U[0 * m + j_start][col] = 1
            V[(j_start + 1) * p + k_start][col] = 1
            V[(j_start + 1) * p + k_start + 1][col] = -1
            W[k_start * n + 0][col] = 1
            col += 1
            
            # M4
            # A: A11
            # B: B01 - B00
            U[1 * m + j_start + 1][col] = 1
            V[j_start * p + k_start + 1][col] = 1
            V[j_start * p + k_start][col] = -1
            W[(k_start + 1) * n + 1][col] = 1
            col += 1
            
            # M5
            # A: A00 + A01
            # B: B11
            U[0 * m + j_start][col] = 1
            U[0 * m + j_start + 1][col] = 1
            V[(j_start + 1) * p + k_start + 1][col] = 1
            W[(k_start + 1) * n + 0][col] = 1
            col += 1
            
            # M6
            # A: A10 - A00
            # B: B00 + B01
            U[1 * m + j_start][col] = 1
            U[0 * m + j_start][col] = -1
            V[j_start * p + k_start][col] = 1
            V[j_start * p + k_start + 1][col] = 1
            W[k_start * n + 1][col] = 1
            col += 1
            
            # M7
            # A: A01 - A11
            # B: B10 + B11
            U[0 * m + j_start + 1][col] = 1
            U[1 * m + j_start + 1][col] = -1
            V[(j_start + 1) * p + k_start][col] = 1
            V[(j_start + 1) * p + k_start + 1][col] = 1
            W[(k_start + 1) * n + 1][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
