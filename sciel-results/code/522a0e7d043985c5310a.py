"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    _deadline = time.time() + 140

    # We decompose 2x3x8 as:
    #   For the 2x2x2 sub-blocks (j in {0,1}, k in {0,1},{2,3},{4,5},{6,7}): Strassen rank 7 each -> 28
    #   For the 2x1x4 sub-blocks (j=2, k in {0..3}, {4..7}): schoolbook rank 8 each -> 16
    # Total rank = 28 + 16 = 44

    rank = 44

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def emit_strassen(i0, j0, k0):
        nonlocal col
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U[(i0+i)*m + (j0+j)], V[(j0+j)*p + (k0+k)], W[(k0+k)*n + (i0+i)]

        # m1 = (A00+A11)(B00+B11)
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # m2 = (A10+A11)B00
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1

        # m3 = A00(B10-B11)
        U[(i0+0)*m + (j0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # m4 = A11(B01-B00)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += -1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        col += 1

        # m5 = (A00+A01)B11
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        col += 1

        # m6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+0)][col] += -1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # m7 = (A01-A11)(B10+B11)
        U[(i0+0)*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        col += 1

    def emit_schoolbook_2x1x4(i0, j0, k0):
        nonlocal col
        for ii in range(2):
            for kk in range(4):
                U[(i0+ii)*m + (j0+0)][col] = 1
                V[(j0+0)*p + (k0+kk)][col] = 1
                W[(k0+kk)*n + (i0+ii)][col] = 1
                col += 1

    # Four Strassen blocks for j in {0,1}
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 0, 4)
    emit_strassen(0, 0, 6)

    # Two schoolbook blocks for j=2
    emit_schoolbook_2x1x4(0, 2, 0)
    emit_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
