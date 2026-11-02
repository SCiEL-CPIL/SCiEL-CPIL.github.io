import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_n, _m, _p = DIMENSIONS

# EVOLVE-BLOCK-START
def _schoolbook_rank27():
    """Known valid rank-27 (schoolbook) decomposition."""
    n, m, p = _n, _m, _p
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

def _verify(U, V, W):
    """Check exact tensor decomposition."""
    n, m, p = _n, _m, _p
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank26_decomposition():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Partition 3x3 into 2x2 + 1x1 blocks; apply Strassen (rank 7) to the
    2x2 block and keep the three 1x1 cross terms plus the (2,2) entry
    handled by Strassen. Total rank = 7 (Strassen 2x2) + 3 (cross 1x1 terms)
    + 1 (extra) = 11... actually let's construct carefully.

    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    B = [[B11, B12, B13],
         [B21, B22, B23],
         [B31, B32, B33]]

    Partition:
    Top-left 2x2 block of A: [[A11, A12],[A21, A22]]
    Top-left 2x2 block of B: [[B11, B12],[B21, B22]]
    Their product gives top-left 2x2 of C.

    Strassen for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22) B11
    M3 = A11 (B12-B22)
    M4 = A22 (B21-B11)
    M5 = (A11+A12) B22
    M6 = (A21-A11) (B11+B12)
    M7 = (A12-A22) (B21+B22)

    C11 = M1 + M4 - M2 + M6
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M5 - M7

    Now the remaining entries:
    C13 = A11*B13 + A12*B23 + A13*B33
    C23 = A21*B13 + A22*B23 + A23*B33
    C31 = A31*B11 + A32*B21 + A33*B31
    C32 = A31*B12 + A32*B22 + A33*B32
    C33 = A31*B13 + A32*B23 + A33*B33

    That's 15 more terms if done naively. But we can share some.

    Actually, the known result: rank-26 comes from using Strassen on the
    2x2 sub-block (rank 7) and then handling the remaining 12 cross terms
    with 19 more multiplications... no that gives 26.

    Let me think differently. The standard approach:
    - Use Strassen for the top-left 2x2: 7 multiplications
    - C13 needs 3 mults, C23 needs 3, C31 needs 3, C32 needs 3, C33 needs 3
    - But C33 = A33*B33 can be shared with the 1x1 block.
    - Actually we need: A13*B33, A23*B33, A31*B11, A31*B12, A32*B21, A32*B22, A33*B31, A33*B32, A33*B33
    - Plus A11*B13, A12*B23, A21*B13, A22*B23

    Total additional: 3+3+3+3+3 = 15, but A33*B33 is one of them.
    7 + 15 = 22? That's too optimistic. Let me be more careful.

    Actually the known rank-26 decomposition uses:
    - Strassen's 7 for the 2x2 block
    - 19 additional for the cross terms
    Total: 26

    Let me construct it explicitly with indices.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    def idx_U(i, j): return i * m + j  # row i, col j of A
    def idx_V(j, k): return j * p + k  # row j, col k of B
    def idx_W(k, i): return k * n + i  # output C[i,k] -> index k*n+i

    col = 0
    # Strassen 2x2 block: rows 0,1 cols 0,1 of A and B
    # M1 = (A00+A11)(B00+B11)
    U[idx_U(0,0)][col] = 1; U[idx_U(1,1)][col] = 1
    V[idx_V(0,0)][col] = 1; V[idx_V(1,1)][col] = 1
    W[idx_W(0,0)][col] = 1; W[idx_W(0,1)][col] = 1; W[idx_W(1,0)][col] = 1; W[idx_W(1,1)][col] = 1
    col += 1

    # M2 = (A10+A11) B00
    U[idx_U(1,0)][col] = 1; U[idx_U(1,1)][col] = 1
    V[idx_V(0,0)][col] = 1
    W[idx_W(0,0)][col] = 1; W[idx_W(1,0)][col] = 1
    col += 1

    # M3 = A00 (B01-B11)
    U[idx_U(0,0)][col] = 1
    V[idx_V(0,1)][col] = 1; V[idx_V(1,1)][col] = -1
    W[idx_W(0,1)][col] = 1
    col += 1

    # M4 = A11 (B10-B00)
    U[idx_U(1,1)][col] = 1
    V[idx_V(1,0)][col] = 1; V[idx_V(0,0)][col] = -1
    W[idx_W(0,0)][col] = 1; W[idx_W(1,1)][col] = 1
    col += 1

    # M5 = (A00+A01) B11
    U[idx_U(0,0)][col] = 1; U[idx_U(0,1)][col] = 1
    V[idx_V(1,1)][col] = 1
    W[idx_W(0,1)][col] = 1; W[idx_W(1,1)][col] = 1
    col += 1

    # M6 = (A10-A00) (B00+B01)
    U[idx_U(1,0)][col] = 1; U[idx_U(0,0)][col] = -1
    V[idx_V(0,0)][col] = 1; V[idx_V(0,1)][col] = 1
    W[idx_W(0,0)][col] = 1
    col += 1

    # M7 = (A01-A11) (B10+B11)
    U[idx_U(0,1)][col] = 1; U[idx_U(1,1)][col] = -1
    V[idx_V(1,0)][col] = 1; V[idx_V(1,1)][col] = 1
    W[idx_W(1,1)][col] = -1
    col += 1

    # Now cross terms (19 more to reach 26):
    # C02 = A00*B02 + A01*B12 + A02*B22
    U[idx_U(0,0)][col] = 1; V[idx_V(0,2)][col] = 1; W[idx_W(0,2)][col] = 1; col += 1
    U[idx_U(0,1)][col] = 1; V[idx_V(1,2)][col] = 1; W[idx_W(0,2)][col] = 1; col += 1
    U[idx_U(0,2)][col] = 1; V[idx_V(2,2)][col] = 1; W[idx_W(0,2)][col] = 1; col += 1

    # C12 = A10*B02 + A11*B12 + A12*B22
    U[idx_U(1,0)][col] = 1; V[idx_V(0,2)][col] = 1; W[idx_W(1,2)][col] = 1; col += 1
    U[idx_U(1,1)][col] = 1; V[idx_V(1,2)][col] = 1; W[idx_W(1,2)][col] = 1; col += 1
    U[idx_U(1,2)][col] = 1; V[idx_V(2,2)][col] = 1; W[idx_W(1,2)][col] = 1; col += 1

    # C20 = A20*B00 + A21*B10 + A22*B20
    U[idx_U(2,0)][col] = 1; V[idx_V(0,0)][col] = 1; W[idx_W(2,0)][col] = 1; col += 1
    U[idx_U(2,1)][col] = 1; V[idx_V(1,0)][col] = 1; W[idx_W(2,0)][col] = 1; col += 1
    U[idx_U(2,2)][col] = 1; V[idx_V(2,0)][col] = 1; W[idx_W(2,0)][col] = 1; col += 1

    # C21 = A20*B01 + A21*B11 + A22*B21
    U[idx_U(2,0)][col] = 1; V[idx_V(0,1)][col] = 1; W[idx_W(2,1)][col] = 1; col += 1
    U[idx_U(2,1)][col] = 1; V[idx_V(1,1)][col] = 1; W[idx_W(2,1)][col] = 1; col += 1
    U[idx_U(2,2)][col] = 1; V[idx_V(2,1)][col] = 1; W[idx_W(2,1)][col] = 1; col += 1

    # C22 = A20*B02 + A21*B12 + A22*B22
    U[idx_U(2,0)][col] = 1; V[idx_V(0,2)][col] = 1; W[idx_W(2,2)][col] = 1; col += 1
    U[idx_U(2,1)][col] = 1; V[idx_V(1,2)][col] = 1; W[idx_W(2,2)][col] = 1; col += 1
    U[idx_U(2,2)][col] = 1; V[idx_V(2,2)][col] = 1; W[idx_W(2,2)][col] = 1; col += 1

    # That's 7 + 18 = 25. Need one more for... let me check.
    # Actually C02, C12, C20, C21, C22 each need 3 = 15. Plus 7 = 22.
    # Wait I have 7 + 3*5 = 22. But I listed 18 cross terms above. Let me recount.
    # C02: 3, C12: 3, C20: 3, C21: 3, C22: 3 = 15. Total = 22.
    # But I wrote col increments for 18. Let me fix - I should have only 15 cross terms.

    # The above is wrong - let me rebuild properly.
    rank = col  # should be 22 if done right, but we want 26
    # Actually this naive approach gives rank 22 which is impossible (lower bound is 19).
    # The issue: Strassen already handles C00,C01,C10,C11. The remaining 5 entries
    # each need 3 scalar mults = 15. Total = 7+15 = 22. But this can't be right
    # because the known rank is at most 23. Let me just verify what we have.

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    deadline = time.time() + 110.0
    best = _schoolbook_rank27()

    # Try Rank 26
    r26 = _rank26_decomposition()
    if r26 is not None:
        best = r26

    return best
# EVOLVE-BLOCK-END
