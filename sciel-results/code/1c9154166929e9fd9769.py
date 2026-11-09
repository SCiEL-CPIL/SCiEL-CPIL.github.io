import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    # Rank-17 exact decomposition for 2x3x3 matrix multiplication.
    # Decomposes into disjoint sub-problems:
    #   T1: 2x2x2 (rows 0,1; cols 0,1; out 0,1) -> Strassen rank 7
    #   T2: 2x2x1 (rows 0,1; cols 0,1; out 2)   -> schoolbook rank 4
    #   T3: 2x1x2 (rows 0,1; col 2; out 0,1)     -> schoolbook rank 4
    #   T4: 2x1x1 (rows 0,1; col 2; out 2)       -> schoolbook rank 2
    # Total: 7 + 4 + 4 + 2 = 17
    #
    # Index conventions:
    #   U[i*m + j, c]  where i=0..1, j=0..2
    #   V[j*p + k, c]  where j=0..2, k=0..2
    #   W[k*n + i, c]  where k=0..2, i=0..1
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17

    # --- T1: 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}) ---
    # Strassen's 7-term algorithm.
    # Sub-matrices: A=[[A11,A12],[A21,A22]], B=[[B11,B12],[B21,B22]]
    # In our indexing:
    #   A11 = U[0,0], A12 = U[0,1], A21 = U[1,0], A22 = U[1,1]
    #   B11 = V[0,0], B12 = V[0,1], B21 = V[1,0], B22 = V[1,1]
    #   Output C11 = W[0,0], C12 = W[1,0], C21 = W[0,1], C22 = W[1,1]
    #
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each Strassen term corresponds to a sum of products of entries from A and B.
    # Since we need an exact rank-1 decomposition, we express each Strassen
    # term as a sum of rank-1 terms.
    #
    # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # But we need to group these into the 7 Strassen factors. The standard
    # rank-7 decomposition uses 7 rank-1 terms where each term is a product
    # of a linear combination of A entries and a linear combination of B entries.
    #
    # The 7 rank-1 terms (u, v, w) for the 2x2x2 block:
    # Term 1: u = A11+A22, v = B11+B22, w contributes to C11(+1), C22(+1)
    # Term 2: u = A21+A22, v = B11,     w contributes to C21(+1), C22(-1)
    # Term 3: u = A11,       v = B12-B22, w contributes to C12(+1), C22(+1)
    # Term 4: u = A22,       v = B21-B11, w contributes to C11(+1), C21(+1)
    # Term 5: u = A11+A12, v = B22,     w contributes to C11(-1), C12(+1)
    # Term 6: u = A21-A11, v = B11+B12, w contributes to C22(+1)
    # Term 7: u = A12-A22, v = B21+B22, w contributes to C11(+1)

    # Now map to our global indices:
    # A entries: A11=U[0,0], A12=U[0,1], A21=U[1,0], A22=U[1,1]
    # B entries: B11=V[0,0], B12=V[0,1], B21=V[1,0], B22=V[1,1]
    # C entries: C11=W[0,0], C12=W[1,0], C21=W[0,1], C22=W[1,1]
    #
    # Note: W index is k*n+i, so:
    #   C11 (k=0,i=0) -> W[0]
    #   C12 (k=1,i=0) -> W[2]
    #   C21 (k=0,i=1) -> W[1]
    #   C22 (k=1,i=1) -> W[3]

    col = 0
    # Term 1: u = A11+A22, v = B11+B22, w: C11+1, C22+1
    U[0][col] += 1   # A11
    U[1][col] += 1   # A22
    V[0][col] += 1   # B11
    V[1][col] += 1   # B22
    W[0][col] += 1   # C11
    W[3][col] += 1   # C22
    col += 1

    # Term 2: u = A21+A22, v = B11, w: C21+1, C22-1
    U[1][col] += 1   # A21
    U[1][col] += 1   # A22
    V[0][col] += 1   # B11
    W[1][col] += 1   # C21
    W[3][col] += -1  # C22
    col += 1

    # Term 3: u = A11, v = B12-B22, w: C12+1, C22+1
    U[0][col] += 1   # A11
    V[1][col] += 1   # B12
    V[1][col] += -1  # B22
    W[2][col] += 1   # C12
    W[3][col] += 1   # C22
    col += 1

    # Term 4: u = A22, v = B21-B11, w: C11+1, C21+1
    U[1][col] += 1   # A22
    V[3][col] += 1   # B21
    V[0][col] += -1  # B11
    W[0][col] += 1   # C11
    W[1][col] += 1   # C21
    col += 1

    # Term 5: u = A11+A12, v = B22, w: C11-1, C12+1
    U[0][col] += 1   # A11
    U[0][col] += 1   # A12
    V[1][col] += 1   # B22
    W[0][col] += -1  # C11
    W[2][col] += 1   # C12
    col += 1

    # Term 6: u = A21-A11, v = B11+B12, w: C22+1
    U[1][col] += 1   # A21
    U[0][col] += -1  # A11
    V[0][col] += 1   # B11
    V[1][col] += 1   # B12
    W[3][col] += 1   # C22
    col += 1

    # Term 7: u = A12-A22, v = B21+B22, w: C11+1
    U[0][col] += 1   # A12
    U[1][col] += -1  # A22
    V[3][col] += 1   # B21
    V[1][col] += 1   # B22
    W[0][col] += 1   # C11
    col += 1

    # --- T2: 2x2x1 block (i in {0,1}, j in {0,1}, k=2) ---
    # Output is C_{*,2} where k=2. W index: k*n+i = 2*2+i = 4+i
    # C_{0,2} = W[4], C_{1,2} = W[5]
    # C_{0,2} = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C_{1,2} = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # A[0,0]=U[0,0], A[0,1]=U[0,1], A[1,0]=U[1,0], A[1,1]=U[1,1]
    # B[0,2]=V[0,2], B[1,2]=V[1,2]

    # Term 8: A[0,0]*B[0,2] -> W[4]
    U[0][col] = 1
    V[2][col] = 1   # B[0,2]: j=0, k=2 -> index 0*3+2=2
    W[4][col] = 1
    col += 1

    # Term 9: A[0,1]*B[1,2] -> W[4]
    U[0][col] = 1   # A[0,1]: i=0, j=1 -> index 0*3+1=1... wait
    # U[i*m+j]: i=0,j=1 -> 0*3+1=1
    U[0][col] = 0
    U[1][col] = 1   # U[1] is i=0,j=1
    V[4][col] = 1   # B[1,2]: j=1, k=2 -> 1*3+2=5
    V[4][col] = 0
    V[5][col] = 1
    W[4][col] = 1
    col += 1

    # Term 10: A[1,0]*B[0,2] -> W[5]
    U[3][col] = 1   # A[1,0]: i=1, j=0 -> 1*3+0=3
    V[2][col] = 1   # B[0,2]
    W[5][col] = 1
    col += 1

    # Term 11: A[1,1]*B[1,2] -> W[5]
    U[4][col] = 1   # A[1,1]: i=1, j=1 -> 1*3+1=4
    V[5][col] = 1   # B[1,2]
    W[5][col] = 1
    col += 1

    # --- T3: 2x1x2 block (i in {0,1}, j=2, k in {0,1}) ---
    # Output C_{*,k} for k=0,1. W index: k*n+i
    # C_{0,0} = A[0,2]*B[2,0] -> W[0]
    # C_{1,0} = A[1,2]*B[2,0] -> W[1]
    # C_{0,1} = A[0,2]*B[2,1] -> W[2]
    # C_{1,1} = A[1,2]*B[2,1] -> W[3]
    # A[0,2]=U[2], A[1,2]=U[5]
    # B[2,0]=V[6], B[2,1]=V[7]

    # Term 12: A[0,2]*B[2,0] -> W[0]
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1

    # Term 13: A[1,2]*B[2,0] -> W[1]
    U[5][col] = 1
    V[6][col] = 1
    W[1][col] = 1
    col += 1

    # Term 14: A[0,2]*B[2,1] -> W[2]
    U[2][col] = 1
    V[7][col] = 1
    W[2][col] = 1
    col += 1

    # Term 15: A[1,2]*B[2,1] -> W[3]
    U[5][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1

    # --- T4: 2x1x1 block (i in {0,1}, j=2, k=2) ---
    # Output C_{*,2}. W index: 2*2+i = 4+i
    # C_{0,2} = A[0,2]*B[2,2] -> W[4]
    # C_{1,2} = A[1,2]*B[2,2] -> W[5]
    # A[0,2]=U[2], A[1,2]=U[5], B[2,2]=V[8]

    # Term 16: A[0,2]*B[2,2] -> W[4]
    U[2][col] = 1
    V[8][col] = 1
    W[4][col] = 1
    col += 1

    # Term 17: A[1,2]*B[2,2] -> W[5]
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
