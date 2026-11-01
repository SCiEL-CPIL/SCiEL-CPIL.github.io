import time

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U)>0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    # check that for j != i all zero already covered; also need only matching produce
    return True

# EVOLVE-BLOCK-START
def _strassen7_factors():
    """Return (U7, V7, W7) for 2x2x2 matmul, each 4x7 with integer entries.
    U rows: A00,A01,A10,A11 (index i*2+j)
    V rows: B00,B01,B10,B11 (index j*2+k)
    W rows: C00,C01,C10,C11 (index k*2+i)
    
    Uses 7 bilinear forms. Each column r defines a product term:
    term_r = (sum_i,j U7[i,j,r]*A[i,j]) * (sum_j,k V7[j,k,r]*B[j,k])
    C[i,k] += W7[k,i,r] * term_r
    """
    # Verified formulation with 0, ±1 entries:
    # col 0: A00*B00
    # col 1: A01*B10
    # col 2: A10*B00
    # col 3: A11*B11
    # col 4: (A00+A11)*(B00+B11)
    # col 5: (A01+A11)*(B01+B11)  -- wait, this gives wrong C entries
    # Let me use a different approach: start with schoolbook 4 terms and add 3 more
    # to create cancellations that reduce needed terms.
    
    # Actually, let me use the well-known decomposition:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A11*(B01-B00)
    # M3 = A00*(B11-B10)
    # M4 = (A00+A10)*(B00+B10)
    # M5 = (A01+A11)*(B01+B11)
    # M6 = (A00-A10)*(B10+B11)
    #
    # C00 = M0 - M4 + M5 - M6
    # C01 = M3 + M5
    # C10 = M1 + M4
    # C11 = M0 - M1 + M2 + M4
    #
    # Verify C00 = A00*B00 + A01*B10:
    # M0 = A00B00+A00B11+A11B00+A11B11
    # M4 = A00B00+A00B10+A10B00+A10B10
    # M5 = A01B01+A01B11+A11B01+A11B11
    # M6 = A00B10+A00B11-A10B10-A10B11
    # C00 = (A00B00+A00B11+A11B00+A11B11) - (A00B00+A00B10+A10B00+A10B10)
    #      + (A01B01+A01B11+A11B01+A11B11) - (A00B10+A00B11-A10B10-A10B11)
    # = A00B00+A00B11+A11B00+A11B11-A00B00-A00B10-A10B00-A10B10
    #   +A01B01+A01B11+A11B01+A11B11-A00B10-A00B11+A10B10+A10B11
    # = A11B00+A11B11-A00B10-A10B00+A01B01+A01B11+A11B01+A10B11-A00B10
    # = A11B00+A11B11-2A00B10-A10B00+A01B01+A01B11+A11B01+A10B11
    # This is NOT A00B00+A01B10. So this formulation is WRONG.
    
    # The correct standard Strassen (from Wikipedia, 0-indexed):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A11(B01-B00)
    # M4 = A00(B11-B10)
    # M5 = (A00+A10)(B00+B10)
    # M6 = (A01+A11)(B01+B11)
    # M7 = (A00-A10)(B10+B11)
    #
    # C00 = M1 - M5 + M6 - M7
    # C01 = M4 + M6
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M5
    #
    # Verify C00 = A00B00+A01B10:
    # M1 = A00B00+A00B11+A11B00+A11B11
    # M5 = A00B00+A00B10+A10B00+A10B10
    # M6 = A01B01+A01B11+A11B01+A11B11
    # M7 = A00B10+A00B11-A10B10-A10B11
    # C00 = M1-M5+M6-M7
    # = A00B00+A00B11+A11B00+A11B11 - A00B00-A00B10-A10B00-A10B10
    #   + A01B01+A01B11+A11B01+A11B11 - A00B10-A00B11+A10B10+A10B11
    # = A11B00+A11B11-A00B10-A10B00+A01B01+A01B11+A11B01+A10B11-A00B10
    # = A11B00+A11B11-2A00B10-A10B00+A01B01+A01B11+A11B01+A10B11
    # Still not A00B00+A01B10!
    
    # I think the issue is that standard Strassen computes C=A*B where
    # C[i][j] = sum_k A[i][k]*B[k][j], but my tensor convention is
    # C[i][k] = sum_j A[i][j]*B[j][k]. These are the same if I
    # transpose B. Let me just verify numerically in _verify_strassen7.
    # 
    # For now, use a formulation and let runtime verification catch errors.
    U7 = [
        [ 1, 0, 0, 1, 1, 0, 1],
        [ 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1, 0,-1],
        [ 1, 1, 1, 0, 0, 1, 0],
    ]
    V7 = [
        [ 1, 1, 1, 0, 1, 0, 0],
        [ 0, 0, 1, 0, 0, 1, 0],
        [ 0, 0, 0, 1, 1, 0, 1],
        [ 1, 0, 0,-1, 0, 1, 1],
    ]
    W7 = [
        [ 1, 0, 0, 0,-1, 1, 1],
        [ 0, 0, 0, 1, 0, 1, 0],
        [ 0, 1, 0, 0, 1, 0, 0],
        [ 1,-1, 1, 0, 1, 0, 0],
    ]
    return U7, V7, W7

def _verify_strassen7(U7, V7, W7):
    """Verify the Strassen-7 factors produce correct 2x2x2 matmul."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U7[i*2+j][r] * V7[j*2+k][r] * W7[k*2+i][r]
                expected = 1.0 if j == i else 0.0
                if abs(s - expected) > 1e-8:
                    return False
    return True

def _strassen7_candidates():
    """Yield multiple candidate (U7,V7,W7) triples for 2x2x2 matmul."""
    # Candidate 1: Based on standard Strassen (may have sign/index errors)
    U7a = [
        [ 1, 0, 0, 1, 1, 0, 1],
        [ 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1, 0,-1],
        [ 1, 1, 1, 0, 0, 1, 0],
    ]
    V7a = [
        [ 1, 1, 1, 0, 1, 0, 0],
        [ 0, 0, 1, 0, 0, 1, 0],
        [ 0, 0, 0, 1, 1, 0, 1],
        [ 1, 0, 0,-1, 0, 1, 1],
    ]
    W7a = [
        [ 1, 0, 0, 0,-1, 1, 1],
        [ 0, 0, 0, 1, 0, 1, 0],
        [ 0, 1, 0, 0, 1, 0, 0],
        [ 1,-1, 1, 0, 1, 0, 0],
    ]
    yield U7a, V7a, W7a
    
    # Candidate 2: Try transposing B convention (swap j,k in V)
    U7b = [
        [ 1, 0, 0, 1, 1, 0, 1],
        [ 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1, 0,-1],
        [ 1, 1, 1, 0, 0, 1, 0],
    ]
    V7b = [
        [ 1, 0, 0, 0, 1, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 0],
        [ 1, 0, 0, 0, 1, 0, 1],
        [ 1, 1, 0,-1, 0, 1, 1],
    ]
    W7b = [
        [ 1, 0, 0, 0,-1, 1, 1],
        [ 0, 0, 0, 1, 0, 1, 0],
        [ 0, 1, 0, 0, 1, 0, 0],
        [ 1,-1, 1, 0, 1, 0, 0],
    ]
    yield U7b, V7b, W7b
    
    # Candidate 3: Different assignment
    U7c = [
        [ 1, 0, 0, 0, 1, 0, 1],
        [ 0, 1, 0, 0, 0, 1, 0],
        [ 0, 0, 1, 0, 1, 0,-1],
        [ 1, 0, 1, 1, 0, 1, 0],
    ]
    V7c = [
        [ 1, 0, 1, 0, 1, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 0],
        [ 0, 1, 0, 0, 1, 0, 1],
        [ 1, 0, 0, 1, 1, 1, 1],
    ]
    W7c = [
        [ 1, 0, 0, 0,-1, 1, 1],
        [ 0, 1, 0, 0, 0, 1, 0],
        [ 0, 0, 1, 0, 1, 0, 0],
        [ 1,-1, 1, 1, 1, 0, 0],
    ]
    yield U7c, V7c, W7c

def construct():
    start_t = time.time()
    deadline = start_t + 105
    n, m, p = DIMENSIONS  # 2,2,12

    # Trivial rank-48 (schoolbook) is always valid
    fb = _schoolbook(n, m, p)
    if not _check(fb["U"], fb["V"], fb["W"], n, m, p):
        return fb

    # Attempt rank-42: tile p=12 into 6 groups of width 2, use Strassen-7 per group
    for U7, V7, W7 in _strassen7_candidates():
        if not _verify_strassen7(U7, V7, W7):
            continue
        rank = 6 * 7  # 42
        U = [[0.0] * rank for _ in range(n * m)]   # 4 x 42
        V = [[0.0] * rank for _ in range(m * p)]   # 24 x 42
        W = [[0.0] * rank for _ in range(p * n)]   # 24 x 42

        for g in range(6):
            base = g * 7
            for i in range(n):
                for j in range(m):
                    for r in range(7):
                        U[i*m+j][base + r] = U7[i*2+j][r]
            for j in range(m):
                for k in range(2):
                    for r in range(7):
                        V[j*p + (2*g+k)][base + r] = V7[j*2+k][r]
            for k in range(2):
                for i in range(n):
                    for r in range(7):
                        W[(2*g+k)*n + i][base + r] = W7[k*2+i][r]

        if _check(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    return fb

# EVOLVE-BLOCK-END
