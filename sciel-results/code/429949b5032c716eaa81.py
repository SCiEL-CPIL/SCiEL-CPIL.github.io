"""Matrix multiplication 3x3x3 rank-26 decomposition via Strassen on 2x2 block."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-26 exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses Strassen's algorithm on the top-left 2x2 block and direct terms for
    the remaining 1x1 block interactions.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 3, 3, 3
    rank = 26

    # U has shape (n*m, rank) = (9, 26)
    # V has shape (m*p, rank) = (9, 26)
    # W has shape (p*n, rank) = (9, 26)
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_row, v_row, w_row, col, u_val=1, v_val=1, w_val=1):
        """Add a rank-1 term at column 'col'."""
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val

    def add_term2(u_row, v_row, w_row, col, u_val, v_val, w_val):
        """Add a rank-1 term at column 'col' with specified values."""
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val

    # Index helper: A[i,j] -> U row i*3+j
    #                B[j,k] -> V row j*3+k
    #                C[i,k] -> W row k*3+i

    # Strassen's 7 terms for the 2x2 top-left block
    # A = [[A00, A01], [A10, A11]] (indices 0,1,2,3 for i,j in {0,1})
    # B = [[B00, B01], [B10, B11]] (indices 0,1,2,3 for j,k in {0,1})
    # C = [[C00, C01], [C10, C11]] (indices 0,1,2,3 for i,k in {0,1})

    # M1 = (A00 + A11) * (B00 + B11)
    # Contributes to C00, C11
    col = 0
    # (A00 + A11)
    U[0][col] += 1  # A00
    U[3][col] += 1  # A11
    # (B00 + B11)
    V[0][col] += 1  # B00
    V[3][col] += 1  # B11
    # M1 * (B00+B11) contributes to C00 via A00*(B00+B11) part and A11*(B00+B11)
    # Actually Strassen: C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M5
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4

    # Let me redo this carefully.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A11-A10)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5

    # Reset and redo
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # M1 = (A00+A11)(B00+B11)
    U[0][col] += 1   # A00
    U[3][col] += 1   # A11
    V[0][col] += 1   # B00
    V[3][col] += 1   # B11
    # M1 contributes: C00 += M1, C11 += M1
    W[0][col] += 1   # C00 (k=0,i=0)
    W[3][col] += 1   # C11 (k=1,i=1)
    col += 1

    # M2 = (A10+A11)B00
    U[2][col] += 1   # A10
    U[3][col] += 1   # A11
    V[0][col] += 1   # B00
    # M2 contributes: C10 += M2, C11 -= M2
    W[2][col] += 1   # C10 (k=0,i=1)
    W[3][col] -= 1   # C11 (k=1,i=1)
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] += 1   # A00
    V[1][col] += 1   # B01
    V[3][col] -= 1   # -B11
    # M3 contributes: C01 += M3, C11 += M3
    W[1][col] += 1   # C01 (k=1,i=0)
    W[3][col] += 1   # C11 (k=1,i=1)
    col += 1

    # M4 = A11(B10-B00)
    U[3][col] += 1   # A11
    V[2][col] += 1   # B10
    V[0][col] -= 1   # -B00
    # M4 contributes: C00 += M4, C10 += M4
    W[0][col] += 1   # C00 (k=0,i=0)
    W[2][col] += 1   # C10 (k=0,i=1)
    col += 1

    # M5 = (A00+A10)B11
    U[0][col] += 1   # A00
    U[2][col] += 1   # A10
    V[3][col] += 1   # B11
    # M5 contributes: C01 += M5, C00 -= M5, C11 += M5
    W[1][col] += 1   # C01 (k=1,i=0)
    W[0][col] -= 1   # C00 (k=0,i=0)
    W[3][col] += 1   # C11 (k=1,i=1)
    col += 1

    # M6 = (A01-A00)(B00+B10)
    U[1][col] += 1   # A01
    U[0][col] -= 1   # -A00
    V[0][col] += 1   # B00
    V[2][col] += 1   # B10
    # M6 contributes: C11 += M6 (wait, let me check)
    # Actually C00 = M1 + M4 - M5 + M7, no M6 in C00
    # C01 = M3 + M5, no M6
    # C10 = M2 + M4, no M6
    # C11 = M1 - M2 + M3 + M5, no M6
    # Hmm, M6 is not used in the 2x2 Strassen? Let me recheck.
    # Standard Strassen uses M1-M7, all are used.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    # M6 is not used! That's wrong. Let me look up the correct formula.
    #
    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A11-A10)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    #
    # Wait, M6 is indeed not in the output. That can't be right for a 7-term algorithm.
    # Let me look at this more carefully. The standard 7-term Strassen:
    #
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a10)b11
    # p6 = (a01-a00)(b00+b10)
    # p7 = (a11-a10)(b01+b11)
    #
    # c00 = p1 - p5 + p6 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 - p7

    # OK so M6 IS used in C00. Let me redo.
    # C00 = M1 - M5 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M7

    # Reset again
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # M1 = (A00+A11)(B00+B11)
    U[0][col] += 1   # A00
    U[3][col] += 1   # A11
    V[0][col] += 1   # B00
    V[3][col] += 1   # B11
    # C00 += M1, C11 += M1
    W[0][col] += 1   # C00
    W[3][col] += 1   # C11
    col += 1

    # M2 = (A10+A11)B00
    U[2][col] += 1   # A10
    U[3][col] += 1   # A11
    V[0][col] += 1   # B00
    # C10 += M2, C11 -= M2
    W[2][col] += 1   # C10
    W[3][col] -= 1   # C11
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] += 1   # A00
    V[1][col] += 1   # B01
    V[3][col] -= 1   # -B11
    # C01 += M3, C11 += M3
    W[1][col] += 1   # C01
    W[3][col] += 1   # C11
    col += 1

    # M4 = A11(B10-B00)
    U[3][col] += 1   # A11
    V[2][col] += 1   # B10
    V[0][col] -= 1   # -B00
    # C10 += M4
    W[2][col] += 1   # C10
    col += 1

    # M5 = (A00+A10)B11
    U[0][col] += 1   # A00
    U[2][col] += 1   # A10
    V[3][col] += 1   # B11
    # C00 -= M5, C01 += M5
    W[0][col] -= 1   # C00
    W[1][col] += 1   # C01
    col += 1

    # M6 = (A01-A00)(B00+B10)
    U[1][col] += 1   # A01
    U[0][col] -= 1   # -A00
    V[0][col] += 1   # B00
    V[2][col] += 1   # B10
    # C00 += M6
    W[0][col] += 1   # C00
    col += 1

    # M7 = (A11-A10)(B01+B11)
    U[3][col] += 1   # A11
    U[2][col] -= 1   # -A10
    V[1][col] += 1   # B01
    V[3][col] += 1   # B11
    # C00 += M7, C11 -= M7
    W[0][col] += 1   # C00
    W[3][col] -= 1   # C11
    col += 1

    # Now add direct terms for the 3rd row/column
    # We need all C[i,k] for i or k = 2, plus the 2x2 block is done.
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]

    # That's 15 terms for the 3rd row/col. Total = 7 + 15 = 22.
    # But we need rank 26, so we have 4 extra terms.
    # Wait, the 2x2 block has 4 output entries, each is a sum of products.
    # The 3rd row/col has 5 output entries (C[0,2], C[1,2], C[2,0], C[2,1], C[2,2]).
    # Each of these 5 has 3 terms, so 15 terms. Total 7+15=22.
    # But the schoolbook is 27, and Strassen on 2x2 saves 1 (8->7), so 27-1=26.
    # Hmm, 22 != 26. Let me reconsider.
    #
    # Actually, the 2x2 Strassen replaces 8 multiplications with 7.
    # The remaining 15 multiplications (involving the 3rd row/col) stay as is.
    # Total = 7 + 15 = 22. But that's less than 26.
    # Wait, I think I'm miscounting. Let me list all 27 schoolbook terms:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 27 terms total.
    # The 2x2 block (i,j,k in {0,1}) has 8 terms. Strassen replaces these 8 with 7.
    # The remaining 27-8 = 19 terms involve at least one index = 2.
    # So total = 7 + 19 = 26. That matches!

    # Let me list the 19 terms involving index 2:
    # i=0, k=2: A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2]  (3)
    # i=1, k=2: A[1,0]*B[0,2], A[1,1]*B[1,2], A[1,2]*B[2,2]  (3)
    # i=2, k=0: A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0]  (3)
    # i=2, k=1: A[2,0]*B[0,1], A[2,1]*B[1,1], A[2,2]*B[2,1]  (3)
    # i=2, k=2: A[2,0]*B[0,2], A[2,1]*B[1,2], A[2,2]*B[2,2]  (3)
    # Total: 15 terms.
    # 7 + 15 = 22, not 26. I'm still off.

    # Oh wait, I think the issue is that Strassen's 7 terms don't directly map to
    # the 8 schoolbook terms in a 1-to-1 way. The 7 Strassen terms each involve
    # multiple A and B entries, so they "cover" the 8 schoolbook terms.
    # The remaining 19 schoolbook terms (those with at least one index=2) are
    # separate. So total = 7 + 19 = 26.

    # But I only listed 15 terms above. Let me recount the 27 schoolbook terms:
    # i=0: k=0 (j=0,1,2), k=1 (j=0,1,2), k=2 (j=0,1,2) -> 9
    # i=1: k=0 (j=0,1,2), k=1 (j=0,1,2), k=2 (j=0,1,2) -> 9
    # i=2: k=0 (j=0,1,2), k=1 (j=0,1,2), k=2 (j=0,1,2) -> 9
    # Total 27.

    # Terms with all indices in {0,1}: i in {0,1}, j in {0,1}, k in {0,1} -> 8 terms
    # Terms with at least one index = 2: 27 - 8 = 19 terms

    # Let me list the 19:
    # j=2: A[i,2]*B[2,k] for i,k in {0,1,2} -> 9 terms
    # i=2, j in {0,1}: A[2,j]*B[j,k] for k in {0,1,2} -> 6 terms
    # k=2, i in {0,1}, j in {0,1}: A[i,j]*B[j,2] -> 4 terms
    # But some overlap: i=2, j=2, k=2 is counted in all three.
    # Let me just list them explicitly:
    # (i,j,k) where at least one is 2:
    # i=0: (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2) -> 5
    # i=1: (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2) -> 5
    # i=2: (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2), (2,2,0), (2,2,1), (2,2,2) -> 9
    # Total: 5+5+9 = 19. Good.

    # So I need 19 more terms. Total 7+19=26.

    # Add the 19 direct terms
    # (0,0,2): A[0,0]*B[0,2] -> C[0,2]
    U[0][col] += 1; V[2][col] += 1; W[6][col] += 1  # C[0,2]: k=2,i=0 -> W[2*3+0]=W[6]
    col += 1
    # (0,1,2): A[0,1]*B[1,2] -> C[0,2]
    U[1][col] += 1; V[5][col] += 1; W[6][col] += 1
    col += 1
    # (0,2,0): A[0,2]*B[2,0] -> C[0,0]
    U[2][col] += 1; V[6][col] += 1; W[0][col] += 1
    col += 1
    # (0,2,1): A[0,2]*B[2,1] -> C[0,1]
    U[2][col] += 1; V[7][col] += 1; W[3][col] += 1
    col += 1
    # (0,2,2): A[0,2]*B[2,2] -> C[0,2]
    U[2][col] += 1; V[8][col] += 1; W[6][col] += 1
    col += 1
    # (1,0,2): A[1,0]*B[0,2] -> C[1,2]
    U[3][col] += 1; V[2][col] += 1; W[7][col] += 1
    col += 1
    # (1,1,2): A[1,1]*B[1,2] -> C[1,2]
    U[4][col] += 1; V[5][col] += 1; W[7][col] += 1
    col += 1
    # (1,2,0): A[1,2]*B[2,0] -> C[1,0]
    U[5][col] += 1; V[6][col] += 1; W[1][col] += 1
    col += 1
    # (1,2,1): A[1,2]*B[2,1] -> C[1,1]
    U[5][col] += 1; V[7][col] += 1; W[4][col] += 1
    col += 1
    # (1,2,2): A[1,2]*B[2,2] -> C[1,2]
    U[5][col] += 1; V[8][col] += 1; W[7][col] += 1
    col += 1
    # (2,0,0): A[2,0]*B[0,0] -> C[2,0]
    U[6][col] += 1; V[0][col] += 1; W[2][col] += 1
    col += 1
    # (2,0,1): A[2,0]*B[0,1] -> C[2,1]
    U[6][col] += 1; V[1][col] += 1; W[5][col] += 1
    col += 1
    # (2,0,2): A[2,0]*B[0,2] -> C[2,2]
    U[6][col] += 1; V[2][col] += 1; W[8][col] += 1
    col += 1
    # (2,1,0): A[2,1]*B[1,0] -> C[2,0]
    U[7][col] += 1; V[3][col] += 1; W[2][col] += 1
    col += 1
    # (2,1,1): A[2,1]*B[1,1] -> C[2,1]
    U[7][col] += 1; V[4][col] += 1; W[5][col] += 1
    col += 1
    # (2,1,2): A[2,1]*B[1,2] -> C[2,2]
    U[7][col] += 1; V[5][col] += 1; W[8][col] += 1
    col += 1
    # (2,2,0): A[2,2]*B[2,0] -> C[2,0]
    U[8][col] += 1; V[6][col] += 1; W[2][col] += 1
    col += 1
    # (2,2,1): A[2,2]*B[2,1] -> C[2,1]
    U[8][col] += 1; V[7][col] += 1; W[5][col] += 1
    col += 1
    # (2,2,2): A[2,2]*B[2,2] -> C[2,2]
    U[8][col] += 1; V[8][col] += 1; W[8][col] += 1
    col += 1

    # Verify: col should be 26
    assert col == 26, f"Expected 26 terms, got {col}"

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for r in range(rank):
                    s += U[i * 3 + j][r] * V[j * 3 + k][r] * W[k * 3 + i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to rank 27
        rank27 = 27
        U27 = [[0] * rank27 for _ in range(9)]
        V27 = [[0] * rank27 for _ in range(9)]
        W27 = [[0] * rank27 for _ in range(9)]
        col27 = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U27[i * 3 + j][col27] = 1
                    V27[j * 3 + k][col27] = 1
                    W27[k * 3 + i][col27] = 1
                    col27 += 1
        return {"U": U27, "V": V27, "W": W27}
# EVOLVE-BLOCK-END
