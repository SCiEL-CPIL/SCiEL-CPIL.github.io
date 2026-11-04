"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a hybrid decomposition:
    # 1. Strassen decomposition for the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # 2. Schoolbook decomposition for the remaining terms
    
    # Rank 7 for Strassen part, Rank 92 for Schoolbook part
    # Total rank = 99
    rank = 99
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for sub-block i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Term 1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    col = 0
    U[0][col] = 1; U[6][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # Term 2: (a10+a11)b00 -> c10 += 1, c11 -= 1
    U[5][col] = 1; U[6][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[5][col] = -1
    col += 1
    
    # Term 3: a00(b01-b11) -> c01 += 1, c11 += 1
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1
    
    # Term 4: a11(b10-b00) -> c00 += 1, c10 += 1
    U[6][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # Term 5: (a00+a01)b11 -> c00 -= 1, c01 += 1
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[4][col] = 1
    col += 1
    
    # Term 6: (a10-a00)(b00+b01) -> c11 += 1
    U[5][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # Term 7: (a01-a11)(b10+b11) -> c00 += 1, c01 += 1
    U[1][col] = 1; U[6][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # Schoolbook terms for the rest
    # Iterate over all i, j, k
    # Skip if i in {0,1} and j in {0,1} and k in {0,1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
