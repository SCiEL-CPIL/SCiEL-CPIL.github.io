import time

_DIMS = (2, 4, 6)
_T0 = time.time()

def construct():
    n, m, p = _DIMS
    R = 42
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def blk(i0, j0, k0, off):
        U[i0*m + j0][off] += 1
        U[(i0+1)*m + (j0+1)][off] += 1
        V[j0*p + k0][off] += 1
        V[(j0+1)*p + (k0+1)][off] += 1
        W[k0*n + i0][off] += 1
        W[(k0+1)*n + (i0+1)][off] += 1

        c = off + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] -= 1

        c = off + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = off + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] -= 1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = off + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] -= 1
        W[(k0+1)*n + i0][c] += 1

        c = off + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] -= 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = off + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] -= 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    blk(0, 0, 0, 0)
    blk(0, 0, 2, 7)
    blk(0, 0, 4, 14)
    blk(0, 2, 0, 21)
    blk(0, 2, 2, 28)
    blk(0, 2, 4, 35)

    return {"U": U, "V": V, "W": W}
