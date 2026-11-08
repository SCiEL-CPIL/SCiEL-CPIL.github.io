import time
import random

DIMENSIONS = [2, 4, 6]
DEADLINE = 105.0

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    R = 42
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def strassen(i0, j0, k0, c0):
        # Local->global: row i = i0+dr, col j = j0+dj, out k = k0+dc (dr,dj,dc in {0,1})
        # M1 = a*(h-f): A: (i0,j0) ; B: (j0+1,k0+1)+, (j0,k0+1)- ; W: (k0,i0)+, (k0+1,i0+1)+
        U[i0 * m + j0][c0 + 0] += 1
        V[(j0 + 1) * p + (k0 + 1)][c0 + 0] += 1
        V[j0 * p + (k0 + 1)][c0 + 0] += -1
        W[k0 * n + i0][c0 + 0] += 1
        W[(k0 + 1) * n + (i0 + 1)][c0 + 0] += 1
        # M2 = (c+d)*e: A: (i0+1,j0)+,(i0+1,j0+1)+ ; B: (j0,k0)+ ; W: (k0,i0+1)+
        U[(i0 + 1) * m + j0][c0 + 1] += 1
        U[(i0 + 1) * m + (j0 + 1)][c0 + 1] += 1
        V[j0 * p + k0][c0 + 1] += 1
        W[k0 * n + (i0 + 1)][c0 + 1] += 1
        # M3 = b*(g-h): A: (i0,j0+1)+ ; B: (j0+1,k0)+,(j0+1,k0+1)- ; W: (k0+1,i0)+
        U[i0 * m + (j0 + 1)][c0 + 2] += 1
        V[(j0 + 1) * p + k0][c0 + 2] += 1
        V[(j0 + 1) * p + (k0 + 1)][c0 + 2] += -1
        W[(k0 + 1) * n + i0][c0 + 2] += 1
        # M4 = d*(e-g): A: (i0+1,j0+1)+ ; B: (j0,k0)+,(j0+1,k0)- ; W: (k0,i0)+,(k0,i0+1)+
        U[(i0 + 1) * m + (j0 + 1)][c0 + 3] += 1
        V[j0 * p + k0][c0 + 3] += 1
        V[(j0 + 1) * p + k0][c0 + 3] += -1
        W[k0 * n + i0][c0 + 3] += 1
        W[k0 * n + (i0 + 1)][c0 + 3] += 1
        # M5 = (a+d)*(f+g): A: (i0,j0)+,(i0+1,j0+1)+ ; B: (j0,k0+1)+,(j0+1,k0)+ ; W: (k0,i0)-,(k0+1,i0)+
        U[i0 * m + j0][c0 + 4] += 1
        U[(i0 + 1) * m + (j0 + 1)][c0 + 4] += 1
        V[j0 * p + (k0 + 1)][c0 + 4] += 1
        V[(j0 + 1) * p + k0][c0 + 4] += 1
        W[k0 * n + i0][c0 + 4] += -1
        W[(k0 + 1) * n + i0][c0 + 4] += 1
        # M6 = (c-a)*(e+f): A: (i0+1,j0)+,(i0,j0)- ; B: (j0,k0)+,(j0,k0+1)+ ; W: (k0+1,i0+1)+
        U[(i0 + 1) * m + j0][c0 + 5] += 1
        U[i0 * m + j0][c0 + 5] += -1
        V[j0 * p + k0][c0 + 5] += 1
        V[j0 * p + (k0 + 1)][c0 + 5] += 1
        W[(k0 + 1) * n + (i0 + 1)][c0 + 5] += 1
        # M7 = (b-d)*(g+h): A: (i0,j0+1)+,(i0+1,j0+1)- ; B: (j0+1,k0)+,(j0+1,k0+1)+ ; W: (k0,i0)-,(k0+1,i0+1)+
        U[i0 * m + (j0 + 1)][c0 + 6] += 1
        U[(i0 + 1) * m + (j0 + 1)][c0 + 6] += -1
        V[(j0 + 1) * p + k0][c0 + 6] += 1
        V[(j0 + 1) * p + (k0 + 1)][c0 + 6] += 1
        W[k0 * n + i0][c0 + 6] += -1
        W[(k0 + 1) * n + (i0 + 1)][c0 + 6] += 1

    tile = 0
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                strassen(i0, j0, k0, tile * 7)
                tile += 1
    assert tile == 6, "expected 6 Strassen tiles"

    # Exact integer verification: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k).
    for i in range(n):
        for j in range(m):
            uc = U[i * m + j]
            for k in range(p):
                vc = V[j * p + k]
                wc = W[k * n + i]
                s = 0
                for c in range(R):
                    s += uc[c] * vc[c] * wc[c]
                if s != 1:
                    return _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
