import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _try_rank_10_2x2x3():
    """
    Try to find a rank-10 decomposition for 2x2 A * 2x3 B = 2x3 C.
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # We need to find U (4x10), V (6x10), W (6x10) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all (i,j,k)
    
    # This is a system of 12 equations with 3*40 = 120 unknowns.
    # Let's try a structured approach.
    
    # Start with the 12 direct terms and try to merge 2 pairs.
    # Direct terms: for each (i,j,k), one term with U[i,j]=1, V[j,k]=1, W[i,k]=1
    
    # Try to find 10 terms that work.
    # Use a greedy/iterative approach.
    
    # Initialize with 12 direct terms
    U = np.zeros((4, 12))
    V = np.zeros((6, 12))
    W = np.zeros((6, 12))
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, col] = 1
                V[j*3+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    
    # Now try to reduce to 10 terms by finding linear dependencies.
    # This is complex. Let's try a different known decomposition.
    
    # Actually, let's just return None and use a different strategy.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try Rank 27 strategy
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
            
    # Part 2: j=1,2, k=0..4 (17 terms needed for 2x2x5)
    # Use: 2x2x2 (k=0,1) rank 7 + 2x2x2 (k=2,3) rank 7 + 2x2x1 (k=4) rank 4 = 18
    # That gives 10+18=28. Need 17 for 2x2x5.
    
    # Try: 2x2x3 (k=0,1,2) rank 10 + 2x2x2 (k=3,4) rank 7 = 17
    # If 2x2x3 has rank 10, total = 10+17 = 27.
    
    # For now, use the known 18-term decomposition and accept rank 28.
    # Or try a different split.
    
    # Alternative: j=0,1,2 all together with a smarter decomposition.
    # This is complex. Return None to fallback.
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
