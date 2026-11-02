"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _strassen_block(n, m, p):
    """
    Rank-56 decomposition using 8 blocks of 2x2x2 Strassen multiplication.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # Indices for B in this block (k=2t, 2t+1)
        # B is 2x16. Rows j=0,1. Cols k.
        # b = j*p + k
        v_B00 = 0 * p + 2 * t
        v_B01 = 0 * p + 2 * t + 1
        v_B10 = 1 * p + 2 * t
        v_B11 = 1 * p + 2 * t + 1
        
        # Indices for C in this block
        # C is 16x2. Rows k, Cols i.
        # c = k*n + i
        w_C00 = 2 * t * n + 0
        w_C01 = 2 * t * n + 1
        w_C10 = (2 * t + 1) * n + 0
        w_C11 = (2 * t + 1) * n + 1
        
        # Strassen 7 steps for 2x2x2
        # A is 2x2. a=0:00, 1:01, 2:10, 3:11
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[v_B00][col] += 1; V[v_B11][col] += 1
        W[w_C00][col] += 1; W[w_C11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[v_B00][col] += 1
        W[w_C10][col] += 1; W[w_C11][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[v_B01][col] += 1; V[v_B11][col] -= 1
        W[w_C01][col] += 1; W[w_C11][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[3][col] += 1
        V[v_B10][col] += 1; V[v_B00][col] -= 1
        W[w_C00][col] += 1; W[w_C10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0][col] += 1; U[1][col] += 1
        V[v_B11][col] += 1
        W[w_C00][col] -= 1; W[w_C01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][col] += 1; U[0][col] -= 1
        V[v_B00][col] += 1; V[v_B01][col] += 1
        W[w_C11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[v_B10][col] += 1; V[v_B11][col] += 1
        W[w_C00][col] += 1
        col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    U, V, W = _strassen_block(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
