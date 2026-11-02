"""Matrix multiplication tensor decomposition for (2,3,9) using block Strassen."""

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 block
    def add_strassen(j_start, k_start):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        for i in range(n):
            for j in range(j_start, j_start + 2):
                U[i * m + j][col] += 1
        for j in range(j_start, j_start + 2):
            for k in range(k_start, k_start + 2):
                V[j * p + k][col] += 1
        W[0 * n + 0][col] += 1  # C[0,0]
        W[1 * n + 1][col] += 1  # C[1,1]
        col += 1
        
        # M2 = (A10+A11)B00
        for j in range(j_start, j_start + 2):
            U[1 * m + j][col] += 1
        V[j_start * p + k_start][col] += 1
        W[0 * n + 1][col] += 1  # C[1,0]
        W[1 * n + 1][col] += 1  # C[1,1]
        col += 1
        
        # M3 = A00(B10-B11)
        U[0 * m + j_start][col] += 1
        V[(j_start + 1) * p + k_start][col] += 1
        V[(j_start + 1) * p + (k_start + 1)][col] -= 1
        W[1 * n + 0][col] += 1  # C[0,1]
        W[1 * n + 1][col] -= 1  # C[1,1]
        col += 1
        
        # M4 = A11(B01-B10)
        U[1 * m + (j_start + 1)][col] += 1
        V[j_start * p + (k_start + 1)][col] += 1
        V[(j_start + 1) * p + k_start][col] -= 1
        W[1 * n + 0][col] += 1  # C[0,1]
        W[0 * n + 0][col] -= 1  # C[0,0]
        col += 1
        
        # M5 = (A00+A01)B11
        for j in range(j_start, j_start + 2):
            U[0 * m + j][col] += 1
        V[(j_start + 1) * p + (k_start + 1)][col] += 1
        W[0 * n + 0][col] += 1  # C[0,0]
        W[0 * n + 1][col] += 1  # C[1,0]
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + j_start][col] += 1
        U[0 * m + j_start][col] -= 1
        V[j_start * p + k_start][col] += 1
        V[j_start * p + (k_start + 1)][col] += 1
        W[1 * n + 1][col] += 1  # C[1,1]
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + (j_start + 1)][col] += 1
        U[1 * m + (j_start + 1)][col] -= 1
        V[(j_start + 1) * p + k_start][col] += 1
        V[(j_start + 1) * p + (k_start + 1)][col] += 1
        W[0 * n + 0][col] += 1  # C[0,0]
        W[1 * n + 1][col] -= 1  # C[1,1]
        col += 1
    
    # Helper: add a rank-1 term
    def add_rank1(i, j, k):
        nonlocal col
        U[i * m + j][col] += 1
        V[j * p + k][col] += 1
        W[k * n + i][col] += 1
        col += 1
    
    # 4 Strassen blocks: j in [0,2), k in [0,2), [2,4), [4,6), [6,8)
    for k_start in range(0, 8, 2):
        add_strassen(0, k_start)
    
    # 1 block of 2x2x1: j in [0,2), k=8
    for j in range(2):
        for i in range(n):
            add_rank1(i, j, 8)
    
    # 4 blocks of 1x2x2: j=2, k in [0,2), [2,4), [4,6), [6,8)
    for k_start in range(0, 8, 2):
        for k in range(k_start, k_start + 2):
            for i in range(n):
                add_rank1(i, 2, k)
    
    # 1 block of 1x2x1: j=2, k=8
    for i in range(n):
        add_rank1(i, 2, 8)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
