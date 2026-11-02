import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-28 tensor decomposition for 2x3 matrix * 3x5 matrix.
    Uses a hybrid blocking: a 2x2x5 Strassen block on the first two
    column/row indices and schoolbook on the last one.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0]*rank for _ in range(n*m)]   # 6 x 28
    V = [[0]*rank for _ in range(m*p)]   # 15 x 28
    W = [[0]*rank for _ in range(p*n)]   # 10 x 28
    col = 0

    # ---- Block A: i in {0,1}, j in {0,1}, k in {0,1,2,3,4} ----
    # This is M_{2,2,5}. We implement it using:
    #   k={0,1}: rank-7 Strassen  (7 cols)
    #   k=2:     rank-3 custom    (3 cols)
    #   k={3,4}: rank-7 Strassen  (7 cols)
    # Total for Block A: 17 cols
    #
    # Strassen terms for a 2x2x2 sub-block at (i_off, j_off, k_off):
    def strassen222(i_off, j_off, k_off, rank_offset):
        nonlocal col
        r = rank_offset

        # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
        U[(i_off+0)*m + (j_off+0)][r] = 1
        U[(i_off+1)*m + (j_off+1)][r] = 1
        V[(j_off+0)*p + (k_off+0)][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = 1
        W[(k_off+0)*n + (i_off+0)][r] = 1
        W[(k_off+1)*n + (i_off+1)][r] = 1
        r += 1

        # M2: (A10+A11)B00 -> C10+=, C11-=
        U[(i_off+1)*m + (j_off+0)][r] = 1
        U[(i_off+1)*m + (j_off+1)][r] = 1
        V[(j_off+0)*p + (k_off+0)][r] = 1
        W[(k_off+0)*n + (i_off+1)][r] = 1
        W[(k_off+1)*n + (i_off+1)][r] = -1
        r += 1

        # M3: A00(B01-B11) -> C01+=, C11+=
        U[(i_off+0)*m + (j_off+0)][r] = 1
        V[(j_off+0)*p + (k_off+1)][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = -1
        W[(k_off+1)*n + (i_off+0)][r] = 1
        W[(k_off+1)*n + (i_off+1)][r] = 1
        r += 1

        # M4: A11(B10-B00) -> C00+=, C10+=
        U[(i_off+1)*m + (j_off+1)][r] = 1
        V[(j_off+1)*p + (k_off+0)][r] = 1
        V[(j_off+0)*p + (k_off+0)][r] = -1
        W[(k_off+0)*n + (i_off+0)][r] = 1
        W[(k_off+0)*n + (i_off+1)][r] = 1
        r += 1

        # M5: (A00+A01)B11 -> C00-=, C01+=
        U[(i_off+0)*m + (j_off+0)][r] = 1
        U[(i_off+0)*m + (j_off+1)][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = 1
        W[(k_off+0)*n + (i_off+0)][r] = -1
        W[(k_off+1)*n + (i_off+0)][r] = 1
        r += 1

        # M6: (A10-A00)(B00+B01) -> C11+=
        U[(i_off+1)*m + (j_off+0)][r] = 1
        U[(i_off+0)*m + (j_off+0)][r] = -1
        V[(j_off+0)*p + (k_off+0)][r] = 1
        V[(j_off+0)*p + (k_off+1)][r] = 1
        W[(k_off+1)*n + (i_off+1)][r] = 1
        r += 1

        # M7: (A01-A11)(B10+B11) -> C00+=
        U[(i_off+0)*m + (j_off+1)][r] = 1
        U[(i_off+1)*m + (j_off+1)][r] = -1
        V[(j_off+1)*p + (k_off+0)][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = 1
        W[(k_off+0)*n + (i_off+0)][r] = 1
        r += 1
        return r

    col = strassen222(0, 0, 0, col)       # k={0,1}, 7 cols
    # k=2 single output: M_{2,2,1} = 3 terms
    # C[2,0] = A[0,0]B[0,2] + A[0,1]B[1,2]  (3 terms)
    # C[2,1] = A[1,0]B[0,2] + A[1,1]B[1,2]
    # Optimal 3-term for 2x2x1:
    # T1 = A[0,0]*B[0,2] -> C[2,0]
    # T2 = A[0,1]*B[1,2] -> C[2,0]
    # T3 = (A[1,0]+A[1,1])*(B[0,2]+B[1,2]) -> C[2,1] += T3, C[2,0] -= A[1,1]*B[1,2] already in T2? 
    # Actually 2x2x1 has rank 3:
    # t1 = A00 B0 -> C0
    # t2 = A01 B1 -> C0
    # t3 = (A10+A11)(B0+B1) -> C1 += t3, C0 -= A11*B1 ... no that's 4.
    # Simpler: just use 3 schoolbook-style for C[2,0] and C[2,1]:
    # C[2,0] = A00*B[0,2] + A01*B[1,2]
    # C[2,1] = A10*B[0,2] + A11*B[1,2]
    # That's 4 terms. But rank of M_{2,2,1} is 3.
    # Use:
    # M1 = A00 * B[0,2] -> C[2,0] += M1
    # M2 = A01 * B[1,2] -> C[2,0] += M2
    # M3 = (A10 + A11) * B[0,2] -> C[2,1] += M3, C[2,0] += A11*B[0,2]... 
    # Hmm let me just use the known rank-3 for 2x2x1:
    # t1 = A[0,0] * B[0,2]  -> C[2,0] += t1
    # t2 = A[0,1] * B[1,2]  -> C[2,0] += t2
    # t3 = (A[1,0] + A[1,1]) * (B[0,2] + B[1,2])  -> C[2,1] += t3
    #   then C[2,0] -= A[1,1]*B[1,2] ... that requires a 4th term.
    #
    # Let me just verify: the 2x2x1 tensor has 4 non-zero components and rank 3.
    # Decomposition:
    # t1 = A[0,0] * B[0,2] -> C[2,0] += t1
    # t2 = A[0,1] * B[1,2] -> C[2,0] += t2
    # t3 = (A[1,0] + A[1,1]) * B[0,2] -> C[2,1] += t3, C[2,0] += A[1,1]*B[0,2]... 
    #
    # Actually the simplest rank-3 for (2,2,1):
    # u1 = A[0,0], v1 = B[0,2], w1 = C[2,0]  (contribution A00*B0)
    # u2 = A[0,1], v2 = B[1,2], w2 = C[2,0]  (contribution A01*B1)
    # u3 = A[1,0]+A[1,1], v3 = B[0,2]+B[1,2], w3 = C[2,1]
    # Check: C[2,1] gets (A10+A11)(B0+B1) = A10*B0 + A10*B1 + A11*B0 + A11*B1 ✓
    # C[2,0] gets A00*B0 + A01*B1, but we also need A10*B0 + A11*B1... missing!
    # So that doesn't work directly.
    #
    # The correct rank-3 for M_{2,2,1} is:
    # t1: A[0,0]*B[0,2] -> C[2,0]
    # t2: A[0,1]*B[1,2] -> C[2,0]
    # t3: (A[1,0]+A[1,1])*(B[0,2]+B[1,2]) -> C[2,1]
    # But C[2,0] should = A[0,0]B[0,2]+A[0,1]B[1,2]+A[1,0]B[0,2]+A[1,1]B[1,2]
    # We're missing A[1,0]B[0,2]+A[1,1]B[1,2].
    #
    # OK the issue is that M_{2,2,1} tensor components are:
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1 (where last index is the single output)
    # Wait no, for matrix multiplication C[i,k] = sum_j A[i,j] B[j,k], with k fixed=2:
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    # This is the tensor with 4 ones. Rank is 3.
    # One known rank-3:
    # t1 = (A[0,0]+A[1,0]) * B[0,2]  -> C[0,2]+=t1, C[1,2]+=t1  ... no
    #
    # Let me use a different approach: just use 4 schoolbook terms. That gives rank 18 for block A
    # which is worse. Let me just use 4 terms here.
    U[0*m+0][col] = 1; V[0*p+2][col] = 1; W[2*n+0][col] = 1; col += 1
    U[0*m+1][col] = 1; V[1*p+2][col] = 1; W[2*n+0][col] = 1; col += 1
    U[1*m+0][col] = 1; V[0*p+2][col] = 1; W[2*n+1][col] = 1; col += 1
    U[1*m+1][col] = 1; V[1*p+2][col] = 1; W[2*n+1][col] = 1; col += 1

    col = strassen222(0, 0, 3, col)       # k={3,4}, 7 cols
    # Block A total: 7 + 4 + 7 = 18

    # ---- Block B: i in {0,1}, j=2, k in {0,1,2,3,4} ----
    # M_{2,1,5} = 10 terms (schoolbook, rank 10 is optimal)
    for i in range(2):
        for k in range(5):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}
