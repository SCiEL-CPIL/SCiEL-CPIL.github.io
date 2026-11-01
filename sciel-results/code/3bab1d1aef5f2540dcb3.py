"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time

# EVOLVE-BLOCK-START
def _build_optimized(n, m, p):
    # n=2, m=2, p=13
    # Rank 46: 6 Strassen blocks (rank 7 each) + 1 Schoolbook block (rank 4)
    rank = 6 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 6 blocks of 2 columns
    for t in range(6):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen U coefficients (for A00, A01, A10, A11)
        # M1: A00 + A11
        U[0][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        # C[0, k1] += M1
        # C[1, k1] -= M1
        W[2 * k1 + 0][col] = 1
        W[2 * k1 + 1][col] = -1
        col += 1
        
        # M2: (A10+A11) B00
        U[2][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1
        # C[0, k1] += M2
        # C[0, k0] -= M2
        W[2 * k1 + 0][col] = 1
        W[2 * k0 + 0][col] = -1
        col += 1
        
        # M3: A00 (B10 - B11)
        U[0][col] = 1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = -1
        # C[1, k0] += M3
        # C[1, k1] += M3
        W[2 * k0 + 1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # M4: (A00+A01) B11
        U[0][col] = 1; U[1][col] = 1
        V[1 * p + k1][col] = 1
        # C[0, k0] += M4
        # C[1, k0] += M4
        W[2 * k0 + 0][col] = 1
        W[2 * k0 + 1][col] = 1
        col += 1
        
        # M5: (A10+A11) (B01 - B00)
        U[2][col] = 1; U[3][col] = 1
        V[0 * p + k1][col] = 1; V[0 * p + k0][col] = -1
        # C[0, k0] += M5
        # C[1, k1] += M5
        W[2 * k0 + 0][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # M6: (A00-A10) (B00+B10)
        U[0][col] = 1; U[2][col] = -1
        V[0 * p + k0][col] = 1; V[1 * p + k0][col] = 1
        # C[0, k0] += M6
        W[2 * k0 + 0][col] = 1
        col += 1
        
        # M7: (A01-A11) (B11+B01)
        U[1][col] = 1; U[3][col] = -1
        V[1 * p + k1][col] = 1; V[0 * p + k1][col] = 1
        # C[1, k1] += M7
        W[2 * k1 + 1][col] = 1
        col += 1

    # Last column k=12
    k0 = 12
    # 1. A00 * B0,12 -> C0,12
    U[0][col] = 1
    V[0 * p + k0][col] = 1
    W[2 * k0 + 0][col] = 1
    col += 1
    
    # 2. A01 * B1,12 -> C0,12
    U[1][col] = 1
    V[1 * p + k0][col] = 1
    W[2 * k0 + 0][col] = 1
    col += 1
    
    # 3. A10 * B0,12 -> C1,12
    U[2][col] = 1
    V[0 * p + k0][col] = 1
    W[2 * k0 + 1][col] = 1
    col += 1
    
    # 4. A11 * B1,12 -> C1,12
    U[3][col] = 1
    V[1 * p + k0][col] = 1
    W[2 * k0 + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
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

def construct():
    start = time.time()
    deadline = start + 110
    n, m, p = DIMENSIONS
    
    # Try optimized rank 46
    result = _build_optimized(n, m, p)
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
