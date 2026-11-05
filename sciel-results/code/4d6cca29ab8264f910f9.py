import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_rank28():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    r = 0

    # Part 1: A col 2 (2x1) * B row 2 (1x5) -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i*m+2][r] = 1
            V[2*p+k][r] = 1
            W[k*n+i][r] = 1
            r += 1

    # Part 2: A cols 0,1 (2x2) * B rows 0,1 cols 0,1 (2x2) -> Strassen 7 terms
    def add_strassen(i1,i2,j1,j2,k1,k2):
        nonlocal r
        # M1 = (A00+A11)(B00+B11)
        U[i1*m+j1][r]=1; U[i2*m+j2][r]=1
        V[j1*p+k1][r]=1; V[j2*p+k2][r]=1
        W[k1*n+i1][r]=1; W[k2*n+i2][r]=1
        r+=1
        # M2 = (A10+A11)B00
        U[i2*m+j1][r]=1; U[i2*m+j2][r]=1
        V[j1*p+k1][r]=1
        W[k1*n+i2][r]=1; W[k2*n+i2][r]=-1
        r+=1
        # M3 = A00(B01-B11)
        U[i1*m+j1][r]=1
        V[j1*p+k2][r]=1; V[j2*p+k2][r]=-1
        W[k2*n+i1][r]=1; W[k2*n+i2][r]=1
        r+=1
        # M4 = A11(B10-B00)
        U[i2*m+j2][r]=1
        V[j2*p+k1][r]=1; V[j1*p+k1][r]=-1
        W[k1*n+i1][r]=1; W[k1*n+i2][r]=1
        r+=1
        # M5 = (A00+A01)B11
        U[i1*m+j1][r]=1; U[i1*m+j2][r]=1
        V[j2*p+k2][r]=1
        W[k1*n+i1][r]=-1; W[k2*n+i1][r]=1
        r+=1
        # M6 = (A10-A00)(B00+B01)
        U[i2*m+j1][r]=1; U[i1*m+j1][r]=-1
        V[j1*p+k1][r]=1; V[j1*p+k2][r]=1
        W[k2*n+i2][r]=1
        r+=1
        # M7 = (A01-A11)(B10+B11)
        U[i1*m+j2][r]=1; U[i2*m+j2][r]=-1
        V[j2*p+k1][r]=1; V[j2*p+k2][r]=1
        W[k1*n+i1][r]=1
        r+=1

    add_strassen(0,1,0,1,0,1)
    add_strassen(0,1,0,1,2,3)

    # Part 3: A cols 0,1 (2x2) * B rows 0,1 col 4 (2x1) -> 4 terms
    for i in range(n):
        for j in range(2):
            U[i*m+j][r] = 1
            V[j*p+4][r] = 1
            W[4*n+i][r] = 1
            r += 1

    assert r == 28, f"Expected 28, got {r}"
    return {"U": U, "V": V, "W": W}

def _try_rank_27():
    """
    Attempt rank-27 by finding a rank-10 decomposition for 2x2 * 2x3.
    Strategy: Use 7 Strassen terms for the 2x2x2 part (cols 0,1) 
    and 3 terms for the remaining column (col 2), but that's 7+3=10 only if
    the 2x2x1 part can be done in 3 terms, which it cannot (needs 4).
    
    Alternative: Use a known rank-10 decomposition for 2x2x3 if one exists.
    This is an open problem area; we attempt a heuristic construction.
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    r = 0

    # Part 1: A col 2 * B row 2 -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i*m+2][r] = 1
            V[2*p+k][r] = 1
            W[k*n+i][r] = 1
            r += 1

    # Part 2: A cols 0,1 * B rows 0,1 cols 0,1,2 -> need rank 10
    # This is the 2x2 * 2x3 -> 2x3 tensor
    # Try: 7 Strassen for cols 0,1 + 4 direct for col 2 = 11. Too many.
    # Try: 6 + 4 = 10? Or some other split?
    # Known: rank(2x2x3) >= 10, and it is known to be 10.
    # Let's try to construct it.
    
    # For 2x2 * 2x3:
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # We need 10 rank-1 terms.
    
    # Use Strassen for the first two columns (7 terms), then 3 terms for the third column?
    # But 2x2 * 2x1 needs 4 terms. So 7+4=11.
    # Unless we can share terms between columns.
    
    # Let's try a different approach: use 5 terms for each "half" with sharing.
    # This is complex. For now, return None to indicate failure.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Try rank 27 first
    res27 = _try_rank_27()
    if res27 is not None:
        if _verify(res27['U'], res27['V'], res27['W'], n, m, p):
            return res27

    # Fallback to rank 28
    res28 = _strassen_rank28()
    if _verify(res28['U'], res28['V'], res28['W'], n, m, p):
        return res28

    # Ultimate fallback: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
