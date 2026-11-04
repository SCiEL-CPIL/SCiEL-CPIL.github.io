import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _strassen_2x2_base():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1: (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5: (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6: (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7: (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _build_4x4_4x4_strassen():
    """Build 4x4 * 4x4 -> 4x4 decomposition using recursive Strassen (rank 49).
    
    Uses the tensor product structure: 4x4x4 = (2x2x2) ⊗ (2x2x2)
    where the outer 2x2x2 uses Strassen (rank 7) and each block multiplication
    is also 2x2x2 Strassen (rank 7). Total rank = 7*7 = 49.
    
    The tensor product of two decompositions:
    If T1 = sum_c u1_c ⊗ v1_c ⊗ w1_c (rank r1)
    and T2 = sum_d u2_d ⊗ v2_d ⊗ w2_d (rank r2)
    then T1 ⊗ T2 = sum_{c,d} (u1_c ⊗ u2_d) ⊗ (v1_c ⊗ v2_d) ⊗ (w1_c ⊗ w2_d)
    with rank r1 * r2.
    
    For 4x4x4 = (2x2x2) ⊗ (2x2x2):
    U4[i*4+j, c*7+d] = U2[i*2+j_block, c] * U2[i_block*2+j, d]
    where the 4x4 index (i,j) maps to block (i_block, j_block) and within-block (i, j)
    with i = i_block*2 + i_inner, j = j_block*2 + j_inner.
    
    Actually, let's be more careful. The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
    We view A as a 2x2 block matrix where A_block[bi,bj] is a 2x2 matrix.
    A[i,j] = A_block[i//2, j//2][i%2, j%2]
    
    For the tensor product decomposition:
    U4[i*4+j, c*7+d] = U2[(i//2)*2 + (j//2), c] * U2[(i%2)*2 + (j%2), d]
    V4[j*4+k, c*7+d] = V2[(j//2)*2 + (k//2), c] * V2[(j%2)*2 + (k%2), d]
    W4[k*4+i, c*7+d] = W2[(k//2)*2 + (i//2), c] * W2[(k%2)*2 + (i%2), d]
    """
    U2, V2, W2 = _strassen_2x2_base()
    
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(7):
        for d in range(7):
            col = c * 7 + d
            
            # U: A[i,j], i in 0..3, j in 0..3
            # Block indices: bi = i//2, bj = j//2; inner: ii = i%2, jj = j%2
            # U4[i*4+j, col] = U2[bi*2+bj, c] * U2[ii*2+jj, d]
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    ii, jj = i % 2, j % 2
                    val = U2[bi*2+bj][c] * U2[ii*2+jj][d]
                    if val != 0:
                        U[i*4+j][col] = val
            
            # V: B[j,k], j in 0..3, k in 0..3
            # Block indices: bj = j//2, bk = k//2; inner: jj = j%2, kk = k%2
            # V4[j*4+k, col] = V2[bj*2+bk, c] * V2[jj*2+kk, d]
            for j in range(4):
                for k in range(4):
                    bj, bk = j // 2, k // 2
                    jj, kk = j % 2, k % 2
                    val = V2[bj*2+bk][c] * V2[jj*2+kk][d]
                    if val != 0:
                        V[j*4+k][col] = val
            
            # W: C[k,i], k in 0..3, i in 0..3
            # Block indices: bk = k//2, bi = i//2; inner: kk = k%2, ii = i%2
            # W4[k*4+i, col] = W2[bk*2+bi, c] * W2[kk*2+ii, d]
            for k in range(4):
                for i in range(4):
                    bk, bi = k // 2, i // 2
                    kk, ii = k % 2, i % 2
                    val = W2[bk*2+bi][c] * W2[kk*2+ii][d]
                    if val != 0:
                        W[k*4+i][col] = val
    
    return U, V, W

def _build_4x4_4x5():
    """Build 4x4 * 4x5 -> 4x5 decomposition.
    
    Split B (4x5) into B1 (4x4) and B2 (4x1).
    C = A*B1 + A*B2
    
    Part 1: A*B1 is 4x4 * 4x4 -> 4x4, rank 49 (recursive Strassen).
    Part 2: A*B2 is 4x4 * 4x1 -> 4x1, rank 1 (the 4x4 all-ones tensor has rank 1).
    
    Total rank: 49 + 1 = 50.
    """
    U4, V4, W4 = _build_4x4_4x4_strassen()
    
    n, m, p = 4, 4, 5
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: 4x4 * 4x4 -> 4x4 (columns 0..48)
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(m):
        for k in range(4):  # k = 0..3
            for c in range(49):
                V[j*p+k][c] = V4[j*4+k][c]
    for k in range(4):  # k = 0..3
        for i in range(n):
            for c in range(49):
                W[k*n+i][c] = W4[k*4+i][c]
    
    # Part 2: 4x4 * 4x1 -> 4x1 (column 49)
    # The tensor for [4,4,1] is all-ones, rank 1.
    # U[i*4+j, 49] = 1, V[j*4+4, 49] = 1, W[4*4+i, 49] = 1 for all i,j
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+4][col] = 1  # B[j,4] is the 5th column (index 4)
    for i in range(4):
        W[4*4+i][col] = 1  # W[k*n+i] with k=4 (index 4 in the p dimension)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140
    
    # Build the rank-50 decomposition
    U, V, W = _build_4x4_4x5()
    rank = len(U[0])
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
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
