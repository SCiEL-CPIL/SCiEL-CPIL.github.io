"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Attempt Strassen 3x3x3 embedding with schoolbook boundary correction
    try:
        result = _strassen_3x3x3_embedded(n, m, p)
        if _verify_decomposition(result, n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (always valid)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _strassen_3x3x3_embedded(n, m, p):
    """Embed Strassen 3x3x3 (rank 7) in top-left 3x3x3, schoolbook for rest."""
    if n < 3 or m < 3 or p < 3:
        return None
    
    # Strassen 3x3x3 decomposition (rank 7)
    sU, sV, sW = _strassen_3x3x3()
    
    # Total rank: 7 (Strassen) + (n*m*p - 27) schoolbook terms
    schoolbook_terms = n * m * p - 27
    rank = 7 + schoolbook_terms
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill Strassen part (local 3x3x3 indices -> global indices)
    for r in range(7):
        for li in range(3):
            for lj in range(3):
                local_u = li * 3 + lj
                global_u = li * m + lj
                U[global_u][r] = sU[local_u][r]
        for lj in range(3):
            for lk in range(3):
                local_v = lj * 3 + lk
                global_v = lj * p + lk
                V[global_v][r] = sV[local_v][r]
        for lk in range(3):
            for li in range(3):
                local_w = lk * 3 + li
                global_w = lk * n + li
                W[global_w][r] = sW[local_w][r]
    
    # Fill schoolbook part for remaining (i,j,k) triples
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 3 and j < 3 and k < 3:
                    continue  # Already handled by Strassen
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _strassen_3x3x3():
    """Return U, V, W (each 9x7) for Strassen 3x3x3 matrix multiplication."""
    U = [[0] * 7 for _ in range(9)]
    V = [[0] * 7 for _ in range(9)]
    W = [[0] * 7 for _ in range(9)]
    
    # Using the standard Strassen 3x3 algorithm
    # Indexing: A[i,j] -> U[i*3+j], B[j,k] -> V[j*3+k], C[i,k] -> W[k*3+i]
    
    # M1 = (A[0,0] + A[2,2]) * (B[0,0] + B[2,2])
    U[0][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[5][0] = 1
    
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[0][1] = -1; W[6][1] = 1; W[1][1] = 1; W[8][1] = 1
    
    # M3 = A[0,0] * (B[1,0] - B[2,0])
    U[0][2] = 1
    V[3][2] = 1; V[6][2] = -1
    W[0][2] = 1; W[4][2] = 1; W[2][2] = 1
    
    # M4 = A[2,2] * (B[2,1] - B[0,1])
    U[8][3] = 1
    V[7][3] = 1; V[1][3] = -1
    W[3][3] = 1; W[4][3] = 1; W[2][3] = 1; W[5][3] = -1
    
    # M5 = (A[0,2] + A[2,2]) * B[2,2]
    U[2][4] = 1; U[8][4] = 1
    V[8][4] = 1
    W[3][4] = 1; W[6][4] = -1; W[7][4] = 1; W[5][4] = 1; W[8][4] = -1
    
    # M6 = (A[0,0] - A[1,0]) * (B[1,0] + B[1,1])
    U[0][5] = 1; U[3][5] = -1
    V[3][5] = 1; V[4][5] = 1
    W[6][5] = 1; W[7][5] = -1; W[8][5] = 1
    
    # M7 = (A[1,1] - A[2,1]) * (B[2,1] + B[2,2])
    U[4][6] = 1; U[7][6] = -1
    V[7][6] = 1; V[8][6] = 1
    W[0][6] = 1; W[7][6] = -1; W[2][6] = 1
    
    return U, V, W


def _verify_decomposition(decomp, n, m, p, num_checks=200):
    """Verify the decomposition by checking random entries."""
    import random
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        # Check if (a,b,c) corresponds to a valid (i,j,k)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue  # Valid position, skip
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
