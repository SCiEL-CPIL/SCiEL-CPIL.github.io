"""Matrix multiplication 2,4,6 — rank-42 decomposition via Strassen tiling."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = 42
    U = np.zeros((n * m, R), dtype=np.int32)
    V = np.zeros((m * p, R), dtype=np.int32)
    W = np.zeros((p * n, R), dtype=np.int32)

    def _emit(i0, j0, k0, off):
        # Seven Strassen multipliers for 2x2 x 2x2 block
        # M1 = (A00+A11)(B00+B11)
        c = off
        U[i0*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # M2 = (A10+A11)B00
        c = off + 1
        U[(i0+1)*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] -= 1

        # M3 = A00(B01-B11)
        c = off + 2
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] -= 1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # M4 = A11(B10-B00)
        c = off + 3
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] -= 1
        W[k0*n + i0, c] += 1
        W[k0*n + (i0+1), c] += 1

        # M5 = (A00+A01)B11
        c = off + 4
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] -= 1
        W[(k0+1)*n + i0, c] += 1

        # M6 = (A10-A00)(B00+B01)
        c = off + 5
        U[(i0+1)*m + j0, c] += 1
        U[i0*m + j0, c] -= 1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # M7 = (A01-A11)(B10+B11)
        c = off + 6
        U[i0*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+1), c] -= 1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1

    # Two column-pairs of A (size 2 each) x three column-pairs of B (size 2 each)
    # A columns: [0,1] and [2,3]; B columns: [0,1], [2,3], [4,5]
    for ac in range(2):
        for bc in range(3):
            _emit(0, 2*ac, 2*bc, ac*21 + bc*7)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
