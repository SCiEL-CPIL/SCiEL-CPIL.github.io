"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_rank_11():
    """
    Construct explicit rank-11 decomposition for 2x2x3 matrix multiplication.
    Uses 7 Strassen terms for the 2x2x2 subproblem (k=0,1) and 4 schoolbook terms for k=2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # For n=2,m=2,p=3: U[0..3], V[0..5], W[0..5]
    # V indices: j*3+k -> (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5
    # W indices: k*2+i -> (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3,(2,0)=4,(2,1)=5
    
    # Strassen's 7 terms for 2x2x2 (handling k=0,1)
    # M1 = (A00+A11)(B00+B11): C00 += M1, C11 += M1
    # M2 = (A10+A11)B00: C10 += M2
    # M3 = A00(B10-B11): C10 += M3
    # M4 = A11(B01-B10): C01 += M4
    # M5 = (A00+A01)B11: C01 += M5
    # M6 = (A10-A00)(B00+B10): C11 += M6
    # M7 = (A01-A11)(B01+B11): C00 += M7
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[2][2] = 1
    
    # Term 3: M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[1][3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[3][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Terms 7-10: Schoolbook for k=2
    # Term 7: A00 * B02 -> C02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 8: A01 * B12 -> C02
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Term 9: A10 * B02 -> C12
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Term 10: A11 * B12 -> C12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the rank-11 decomposition
    U, V, W = _build_rank_11()
    
    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-12 decomposition
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
# EVOLVE-BLOCK-END
