import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    # Part 1: A1 * B1 (14 terms)
    # A1 is col 0 of A: A[0], A[3] (indices 0, 1 in U)
    # B1 is row 0 of B: B[0..6] (indices 0..6 in V)
    # C1 is row 0 of C: C[0..6] (indices 0..6 in W)
    # C2 is row 1 of C: C[7..13] (indices 7..13 in W)
    
    for k in range(7):
        # Term for C[0, k] from A[0,0]*B[0,k]
        U[0][c] = 1
        V[k][c] = 1
        W[k][c] = 1
        c += 1
        
        # Term for C[1, k] from A[1,0]*B[0,k]
        U[1][c] = 1
        V[k][c] = 1
        W[7 + k][c] = 1
        c += 1

    # Part 2: A2 * B2
    # A2 is cols 1,2 of A:
    # a=A[0,1]=U[1], b=A[0,2]=U[2]
    # c=A[1,1]=U[4], d=A[1,2]=U[5]
    # B2 is rows 1,2 of B:
    # Row 1: B[7..13], Row 2: B[14..20]
    
    blocks = [(0, 1), (2, 3), (4, 5)]
    for k1, k2 in blocks:
        # Strassen 7 terms
        # M1: (a+d)(e+h)
        U[1][c] = 1; U[5][c] = 1
        V[7 + k1][c] = 1; V[14 + k2][c] = 1
        W[k1][c] += 1      # P11
        W[7 + k2][c] += 1  # P22
        c += 1
        
        # M2: (c+d)e
        U[4][c] = 1; U[5][c] = 1
        V[7 + k1][c] = 1
        W[7 + k1][c] += 1  # P21
        W[7 + k2][c] += -1 # P22
        c += 1
        
        # M3: a(f-h)
        U[1][c] = 1
        V[7 + k2][c] = 1; V[14 + k2][c] = -1
        W[k2][c] += 1      # P12
        W[7 + k2][c] += 1  # P22
        c += 1
        
        # M4: d(g-e)
        U[5][c] = 1
        V[14 + k1][c] = 1; V[7 + k1][c] = -1
        W[k1][c] += 1      # P11
        W[7 + k1][c] += 1  # P21
        c += 1
        
        # M5: (a+b)h
        U[1][c] = 1; U[2][c] = 1
        V[14 + k2][c] = 1
        W[k1][c] += -1     # P11
        W[k2][c] += 1      # P12
        c += 1
        
        # M6: (c-a)(e+f)
        U[4][c] = 1; U[1][c] = -1
        V[7 + k1][c] = 1; V[7 + k2][c] = 1
        W[7 + k2][c] += 1  # P22
        c += 1
        
        # M7: (b-d)(g+h)
        U[2][c] = 1; U[5][c] = -1
        V[14 + k1][c] = 1; V[14 + k2][c] = 1
        W[k1][c] += 1      # P11
        c += 1

    # Part 3: Single column k=6
    # x=B[1,6]=V[13], y=B[2,6]=V[20]
    # Q1: a*x
    U[1][c] = 1
    V[13][c] = 1
    W[6][c] += 1
    c += 1
    
    # Q2: b*y
    U[2][c] = 1
    V[20][c] = 1
    W[6][c] += 1
    c += 1
    
    # Q3: c*x
    U[4][c] = 1
    V[13][c] = 1
    W[13][c] += 1
    c += 1
    
    # Q4: d*y
    U[5][c] = 1
    V[20][c] = 1
    W[13][c] += 1
    c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
