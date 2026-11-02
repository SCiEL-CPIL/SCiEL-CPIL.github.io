"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's rank-7 algorithm applied independently to each of the 5
    # disjoint column-pairs of the 2x10 matrix. The 2x2 A-factors are shared
    # across blocks; the B/C supports are disjoint, so the tensor rank is the
    # sum of the per-block ranks = 5 * 7 = 35 (optimal for this tensor).
    #
    # Indexing: U[i*m+j, r] for A[i,j]; V[k*m+l, r] for B[k,l]; W[l*n+i, r] for C[i,l]
    # With n=2, m=2, p=10: U is 4x35, V is 20x35, W is 20x35
    # A indices: A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B indices: B0k=V[k], B1k=V[10+k] for k=0..9
    # C indices: C0k=W[2k], C1k=W[2k+1] for k=0..9

    for block in range(5):
        k1 = 2 * block
        k2 = 2 * block + 1

        # M1: (A00+A11)(B00+B11) -> contributes to C0k1, C0k2, C1k1, C1k2
        r = 7 * block + 0
        U[0][r] += 1
        U[3][r] += 1
        V[k1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] += 1
        W[2 * k2 + 1][r] += 1

        # M2: (A10+A11)B00 -> contributes to C1k1, C1k2
        r = 7 * block + 1
        U[2][r] += 1
        U[3][r] += 1
        V[k1][r] += 1
        W[2 * k1 + 1][r] += 1
        W[2 * k2 + 1][r] -= 1

        # M3: A00(B01-B11) -> contributes to C0k2
        r = 7 * block + 2
        U[0][r] += 1
        V[k2][r] += 1
        V[10 + k2][r] -= 1
        W[2 * k2][r] += 1
        W[2 * k2 + 1][r] += 1

        # M4: A11(B10-B00) -> contributes to C1k1
        r = 7 * block + 3
        U[3][r] += 1
        V[10 + k1][r] += 1
        V[k1][r] -= 1
        W[2 * k1][r] += 1
        W[2 * k1 + 1][r] += 1

        # M5: (A00+A01)B11 -> contributes to C0k1, C0k2
        r = 7 * block + 4
        U[0][r] += 1
        U[1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] -= 1
        W[2 * k2][r] += 1

        # M6: (A10-A00)(B00+B01) -> contributes to C1k2
        r = 7 * block + 5
        U[2][r] += 1
        U[0][r] -= 1
        V[k1][r] += 1
        V[k2][r] += 1
        W[2 * k2 + 1][r] += 1

        # M7: (A01-A11)(B10+B11) -> contributes to C0k1
        r = 7 * block + 6
        U[1][r] += 1
        U[3][r] -= 1
        V[10 + k1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
