import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _build_2x2x3_rank10():
    """
    Attempt to find a rank-10 decomposition for 2x2 matrix A and 2x3 matrix B.
    A is 2x2, B is 2x3, C is 2x3.
    Indices:
    A: 0,1,2,3 (i*2+j)
    B: 0,1,2,3,4,5 (j*3+k)
    C: 0,1,2,3,4,5 (k*2+i)
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # This is a search problem. We can use a heuristic or a known decomposition.
    # For now, return None to indicate failure, triggering fallback.
    # In a real improvement, we would solve the system.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try Rank 27 strategy
    # Part 1: j=0, k=0..4 (2x1 A * 1x5 B) -> Rank 10
    # Part 2: j=1,2, k=0,1 (2x2 A * 2x2 B) -> Rank 7 (Strassen)
    # Part 3: j=1,2, k=2,3,4 (2x2 A * 2x3 B) -> Rank 10 (Hypothetical)
    
    rank_27 = _try_rank_27()
    if rank_27 is not None:
        return rank_27
        
    # Fallback to Rank 28
    return _strassen_rank28(n, m, p)

def _try_rank_27():
    n, m, p = 2, 3, 5
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    # Part 1: j=0, k=0..4 (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=1,2, k=0,1 (7 terms, Strassen)
    # A rows 0,1 cols 1,2. B rows 1,2 cols 0,1. C rows 0,1 cols 0,1.
    # Map:
    # A local (i,j) -> global U[i*m + j]
    # B local (j,k) -> global V[j*p + k]
    # C local (i,k) -> global W[k*n + i]
    
    # Strassen for 2x2x2
    # i1=0, i2=1, j1=1, j2=2, k1=0, k2=1
    def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
        nonlocal r
        # M1 = (A00+A11)(B00+B11)
        U[i1*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        # M2 = (A10+A11)B00
        U[i2*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1
        W[k1*n+i2][r] = 1; W[k2*n+i2][r] = -1
        r += 1
        # M3 = A00(B01-B11)
        U[i1*m+j1][r] = 1
        V[j1*p+k2][r] = 1; V[j2*p+k2][r] = -1
        W[k2*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        # M4 = A11(B10-B00)
        U[i2*m+j2][r] = 1
        V[j2*p+k1][r] = 1; V[j1*p+k1][r] = -1
        W[k1*n+i1][r] = 1; W[k1*n+i2][r] = 1
        r += 1
        # M5 = (A00+A01)B11
        U[i1*m+j1][r] = 1; U[i1*m+j2][r] = 1
        V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = -1; W[k2*n+i1][r] = 1
        r += 1
        # M6 = (A10-A00)(B00+B01)
        U[i2*m+j1][r] = 1; U[i1*m+j1][r] = -1
        V[j1*p+k1][r] = 1; V[j1*p+k2][r] = 1
        W[k2*n+i2][r] = 1
        r += 1
        # M7 = (A01-A11)(B10+B11)
        U[i1*m+j2][r] = 1; U[i2*m+j2][r] = -1
        V[j2*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1
        r += 1

    add_strassen_2x2(0, 1, 1, 2, 0, 1)
    
    # Part 3: j=1,2, k=2,3,4 (10 terms?)
    # We need a rank 10 decomposition for 2x2 A * 2x3 B.
    # If we can't find it, this fails.
    # For now, use schoolbook for this part (12 terms) -> Total 10+7+12 = 29. Worse.
    # Use Strassen for 2x2x2 (k=2,3) + Direct for 2x2x1 (k=4)?
    # 7 + 4 = 11 terms. Total 10+7+11 = 28. Same as before.
    
    # So Rank 27 is not achievable with known simple blocks unless 2x2x3 is rank 10.
    # I will return None to fallback to 28.
    return None

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A0 B0 (j=0, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A12 B12 (j=1,2, k=0,1) - 7 terms (Strassen)
    def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
        nonlocal r
        U[i1*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        U[i2*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1
        W[k1*n+i2][r] = 1; W[k2*n+i2][r] = -1
        r += 1
        U[i1*m+j1][r] = 1
        V[j1*p+k2][r] = 1; V[j2*p+k2][r] = -1
        W[k2*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        U[i2*m+j2][r] = 1
        V[j2*p+k1][r] = 1; V[j1*p+k1][r] = -1
        W[k1*n+i1][r] = 1; W[k1*n+i2][r] = 1
        r += 1
        U[i1*m+j1][r] = 1; U[i1*m+j2][r] = 1
        V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = -1; W[k2*n+i1][r] = 1
        r += 1
        U[i2*m+j1][r] = 1; U[i1*m+j1][r] = -1
        V[j1*p+k1][r] = 1; V[j1*p+k2][r] = 1
        W[k2*n+i2][r] = 1
        r += 1
        U[i1*m+j2][r] = 1; U[i2*m+j2][r] = -1
        V[j2*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1
        r += 1

    add_strassen_2x2(0, 1, 1, 2, 0, 1)
    
    # Part 3: A12 B12 (j=1,2, k=2,3) - 7 terms (Strassen)
    add_strassen_2x2(0, 1, 1, 2, 2, 3)
    
    # Part 4: A12 B12 (j=1,2, k=4) - 4 terms (Direct)
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
            
    if ok:
        return {"U": U, "V": V, "W": W}
        
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
