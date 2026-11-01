import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _strassen_2x2_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    U[i*m+j, r] where i,j in {0,1}
    V[j*p+k, r] where j,k in {0,1}
    W[k*n+i, r] where k,i in {0,1}
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen algorithm for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U entries: U[i*m+j, r] = U[i*2+j, r]
    # V entries: V[j*p+k, r] = V[j*2+k, r]
    # W entries: W[k*n+i, r] = W[k*2+i, r]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 (row 0), A11 (row 3)
    # V: B00 (col 0), B11 (col 3)
    # W: C00 (W[0*2+0]=W[0]), C01 (W[1*2+0]=W[2]), C10 (W[0*2+1]=W[1]), C11 (W[1*2+1]=W[3])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = -1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3: A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[2][2] = -1
    
    # M4: A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[3][3] = -1
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = 1
    
    return U, V, W

def _construct_rank14():
    """
    Construct a rank-14 decomposition for 2x2 * 2x4 using two Strassen blocks.
    Columns 0,1 use one Strassen block, columns 2,3 use another.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Get Strassen decomposition for 2x2*2x2
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Block 1: columns 0,1 (offset 0)
    for r in range(7):
        for i in range(n*m):
            U[i][r] = U_s[i][r]
        for j in range(m):
            for k in range(2):  # k=0,1
                V[j*p+k][r] = V_s[j*2+k][r]
        for k in range(2):  # k=0,1
            for i in range(n):
                W[k*n+i][r] = W_s[k*2+i][r]
    
    # Block 2: columns 2,3 (offset 7)
    for r in range(7):
        for i in range(n*m):
            U[i][r+7] = U_s[i][r]
        for j in range(m):
            for k in range(2):  # k=2,3
                V[j*p+k+2][r+7] = V_s[j*2+k][r]
        for k in range(2):  # k=2,3
            for i in range(n):
                W[(k+2)*n+i][r+7] = W_s[k*2+i][r]
    
    return U, V, W

def _try_rank_search(rank, n, m, p, deadline):
    """
    Try to find a rank-rank decomposition using local search.
    Start from a known decomposition and try to reduce rank.
    """
    if time.time() > deadline:
        return None
    
    # This is a placeholder for more sophisticated search
    # For now, we rely on the constructed rank-14 solution
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s safety margin for 150s limit
    
    # Start with the trivial solution (rank 16)
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
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try the rank-14 Strassen-based construction
    if time.time() < deadline:
        U14, V14, W14 = _construct_rank14()
        if _verify_decomposition(U14, V14, W14, n, m, p):
            best_rank = 14
            best_result = {"U": U14, "V": V14, "W": W14}
    
    # Try to find even lower rank using search
    for try_rank in range(7, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_search(try_rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = try_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
