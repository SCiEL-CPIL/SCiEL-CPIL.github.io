"""Matrix multiplication 2,3,5 — rank-28 exact tensor decomposition.

Uses two Strassen 2x2x2 blocks (j in {0,1} × k in {0,1} and j in {0,1} × k in {2,3})
plus 14 schoolbook terms for the remaining (j,k) pairs.
Total rank = 7 + 7 + 14 = 28.
All entries are integers, so the decomposition is exact.
"""
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

def construct():
    """Return {'U','V','W'} with shapes (6,28),(15,28),(10,28) for [2,3,5] matmul."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    c = 0

    def add_strassen_block(j_off, k_off):
        """Add 7 Strassen terms for a 2x2 sub-block: rows i={0,1}, cols j={j_off,j_off+1}, k={k_off,k_off+1}."""
        nonlocal c
        # m1 = (a[0][j0]+a[1][j1])*(b[j0][k0]+b[j1][k1])
        #   -> c[0][k0]+=m1, c[1][k1]+=m1
        U[0*m + j_off][c] = 1
        U[1*m + j_off+1][c] = 1
        V[j_off*p + k_off][c] = 1
        V[(j_off+1)*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1

        # m2 = (a[1][j0]+a[1][j1])*b[j0][k0]
        #   -> c[1][k0]+=m2, c[1][k1]-=m2
        U[1*m + j_off][c] = 1
        U[1*m + j_off+1][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1
        W[(k_off+1)*n + 1][c] = -1
        c += 1

        # m3 = a[0][j0]*(b[j0][k1]-b[j1][k1])
        #   -> c[0][k1]+=m3, c[1][k1]+=m3
        U[0*m + j_off][c] = 1
        V[j_off*p + k_off+1][c] = 1
        V[(j_off+1)*p + k_off+1][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1

        # m4 = a[1][j1]*(b[j1][k0]-b[j0][k0])
        #   -> c[0][k0]+=m4, c[1][k0]+=m4
        U[1*m + j_off+1][c] = 1
        V[(j_off+1)*p + k_off][c] = 1
        V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1
        W[k_off*n + 1][c] = 1
        c += 1

        # m5 = (a[0][j0]+a[0][j1])*b[j1][k1]
        #   -> c[0][k0]-=m5, c[0][k1]+=m5
        U[0*m + j_off][c] = 1
        U[0*m + j_off+1][c] = 1
        V[(j_off+1)*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        c += 1

        # m6 = (a[1][j0]-a[0][j0])*(b[j0][k0]+b[j0][k1])
        #   -> c[1][k1]+=m6
        U[1*m + j_off][c] = 1
        U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1
        V[j_off*p + k_off+1][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1

        # m7 = (a[0][j1]-a[1][j1])*(b[j1][k0]+b[j1][k1])
        #   -> c[0][k0]+=m7
        U[0*m + j_off+1][c] = 1
        U[1*m + j_off+1][c] = -1
        V[(j_off+1)*p + k_off][c] = 1
        V[(j_off+1)*p + k_off+1][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    # Two disjoint Strassen blocks
    add_strassen_block(0, 0)   # j in {0,1}, k in {0,1}
    add_strassen_block(0, 2)   # j in {0,1}, k in {2,3}

    # Schoolbook for remaining (j,k) pairs: j in {0,1} with k=4, and j=2 with all k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j <= 1 and k <= 3:
                    continue  # covered by Strassen blocks
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1

    assert c == rank, f"Expected rank {rank}, got {c}"
    assert time.time() < _DEADLINE, "Time budget exceeded"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
