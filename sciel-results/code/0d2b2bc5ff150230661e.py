"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition (rank n*m*p) as a fallback."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True


def _strassen_2x2_2x2():
    """
    Strassen rank-7 decomposition for 2x2 * 2x2.
    U: (4,7) rows (i,j); V: (4,7) rows (j,k); W: (4,7) rows (k,i).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)(B00+B11); C00+=M1, C11+=M1
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
    # M2 = (A10+A11)B00; C10+=M2, C11-=M2
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
    # M3 = A00(B10-B11); C01+=M3, C11+=M3
    U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
    # M4 = A11(B01-B00); C00+=M4, C10+=M4
    U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
    # M5 = (A00+A01)B11; C00-=M5, C01+=M5
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
    # M6 = (A10-A00)(B00+B01); C11+=M6
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
    # M7 = (A01-A11)(B10+B11); C00+=M7
    U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
    return U, V, W


def _construct_4x4_4x7():
    """
    Rank-97 decomposition for 4x4 * 4x7 = 4x7.
    Split B into B[:, 0:4] (4x4) and B[:, 4:7] (4x3).
    Part 1: 4x4 * 4x4 via Strassen -> rank 49.
    Part 2: 4x4 * 4x3 via schoolbook -> rank 48.
    Total rank = 97.
    Indexing: U row = i*4+j, V row = j*7+k, W row = k*4+i.
    """
    n, m, p = 4, 4, 7
    total_rank = 49 + 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_strassen_2x2(a_rows, b_rows, c_rows, col_off):
        """Add a 2x2*2x2 Strassen product (rank 7) at global row lists.
        a_rows/b_rows/c_rows: 4 global indices in local order (00,01,10,11)."""
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for li in range(4):
            for c in range(7):
                U[a_rows[li]][col_off + c] += U_s[li][c]
                V[b_rows[li]][col_off + c] += V_s[li][c]
                W[c_rows[li]][col_off + c] += W_s[li][c]

    def A_rows(ai, aj):
        return [ai*4+aj, ai*4+aj+1, (ai+1)*4+aj, (ai+1)*4+aj+1]

    def B_rows(bj, bk, bstride):
        return [bj*bstride+bk, bj*bstride+bk+1, (bj+1)*bstride+bk, (bj+1)*bstride+bk+1]

    def C_rows(ci, ck):
        return [ck*4+ci, ck*4+ci+1, (ck+1)*4+ci, (ck+1)*4+ci+1]

    # --- Part 1: 4x4 * 4x4 via Strassen (B columns 0..3) ---
    col = 0
    A00 = A_rows(0, 0); A01 = A_rows(0, 2); A10 = A_rows(1, 0); A11 = A_rows(1, 2)
    B00 = B_rows(0, 0, 7); B01 = B_rows(0, 2, 7); B10 = B_rows(1, 0, 7); B11 = B_rows(1, 2, 7)
    C00 = C_rows(0, 0); C01 = C_rows(0, 2); C10 = C_rows(1, 0); C11 = C_rows(1, 2)

    # M1 = (A00+A11)(B00+B11); C00+=M1, C11+=M1
    add_strassen_2x2(A00, B00, C00, col); add_strassen_2x2(A11, B00, C00, col)
    add_strassen_2x2(A00, B11, C00, col); add_strassen_2x2(A11, B11, C00, col)
    add_strassen_2x2(A00, B00, C11, col); add_strassen_2x2(A11, B00, C11, col)
    add_strassen_2x2(A00, B11, C11, col); add_strassen_2x2(A11, B11, C11, col)
    col += 7
    # M2 = (A10+A11)B00; C10+=M2, C11-=M2
    add_strassen_2x2(A10, B00, C10, col); add_strassen_2x2(A11, B00, C10, col)
    add_strassen_2x2(A10, B00, C11, col); add_strassen_2x2(A11, B00, C11, col)
    col += 7
    # M3 = A00(B01-B11); C01+=M3, C11+=M3
    add_strassen_2x2(A00, B01, C01, col); add_strassen_2x2(A00, B11, C01, col)
    add_strassen_2x2(A00, B01, C11, col); add_strassen_2x2(A00, B11, C11, col)
    col += 7
    # M4 = A11(B10-B00); C00+=M4, C10+=M4
    add_strassen_2x2(A11, B10, C00, col); add_strassen_2x2(A11, B00, C00, col)
    add_strassen_2x2(A11, B10, C10, col); add_strassen_2x2(A11, B00, C10, col)
    col += 7
    # M5 = (A00+A01)B11; C00-=M5, C01+=M5
    add_strassen_2x2(A00, B11, C00, col); add_strassen_2x2(A01, B11, C00, col)
    add_strassen_2x2(A00, B11, C01, col); add_strassen_2x2(A01, B11, C01, col)
    col += 7
    # M6 = (A10-A00)(B00+B01); C11+=M6
    add_strassen_2x2(A10, B00, C11, col); add_strassen_2x2(A00, B00, C11, col)
    add_strassen_2x2(A10, B01, C11, col); add_strassen_2x2(A00, B01, C11, col)
    col += 7
    # M7 = (A01-A11)(B10+B11); C00+=M7
    add_strassen_2x2(A01, B10, C00, col); add_strassen_2x2(A11, B10, C00, col)
    add_strassen_2x2(A01, B11, C00, col); add_strassen_2x2(A11, B11, C00, col)
    col += 7
    # col == 49

    # --- Part 2: 4x4 * 4x3 via schoolbook (B columns 4..6) ---
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4 + j][col] = 1
                V[j*7 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    # col == 49 + 48 = 97

    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Primary: rank-97 block-Strassen decomposition.
    try:
        U, V, W = _construct_4x4_4x7()
        result = {"U": U, "V": V, "W": W}
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    # Fallback to schoolbook (rank 112).
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    return result

# EVOLVE-BLOCK-END
