import numpy as np
import time

def construct():
    # Deadline guard
    t0 = time.time()
    t_deadline = t0 + 120.0

    n, m, p = 2, 3, 5
    R = 28

    # Allocate factor matrices
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    col = 0

    # Helper: write a single rank-1 term
    def emit(u_rows, v_rows, w_rows, val=1):
        nonlocal col
        assert col < R, "rank overflow"
        for r in u_rows:
            U[r, col] += val
        for r in v_rows:
            V[r, col] += val
        for r in w_rows:
            W[r, col] += val
        col += 1

    # Helper: 2x2 by 2x2 Strassen (7 terms)
    # i0,i1: rows of A block; j0,j1: cols of A block / rows of B block; k0,k1: cols of B block
    def strassen_222(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        # M1 = (A00+A11)*(B00+B11)
        emit([i0*m+j0, i1*m+j1], [j0*p+k0, j1*p+k1], [k0*n+i0, k1*n+i1])
        # M2 = A10*(B00+B11)
        emit([i1*m+j0], [j0*p+k0, j1*p+k1], [k0*n+i1, k1*n+i1])
        # M3 = A00*(B11-B01)
        emit([i0*m+j0], [j0*p+k1, j1*p+k1], [k1*n+i0, k1*n+i1])
        W[k1*n+i0, col-1] -= 1
        # M4 = A11*(B10-B00)
        emit([i1*m+j1], [j1*p+k0, j0*p+k0], [k0*n+i0, k0*n+i1])
        W[k0*n+i0, col-1] -= 1
        # M5 = (A00+A01)*B11
        emit([i0*m+j0, i0*m+j1], [j1*p+k1], [k0*n+i0, k1*n+i0])
        W[k0*n+i0, col-1] -= 1
        # M6 = (A10-A00)*(B00+B01)
        emit([i1*m+j0, i0*m+j0], [j0*p+k0, j0*p+k1], [k1*n+i1])
        U[i0*m+j0, col-1] -= 1
        # M7 = (A01-A11)*(B10+B11)
        emit([i0*m+j1, i1*m+j1], [j1*p+k0, j1*p+k1], [k0*n+i0])
        U[i1*m+j1, col-1] -= 1

    # Helper: 2x2 by 2x1 schoolbook (4 terms)
    def schoolbook_221(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        for ii in range(2):
            for jj in range(2):
                emit([ (i0+ii)*m + (j0+jj) ], [ (j0+jj)*p + k0 ], [ k0*n + (i0+ii) ])

    # Block 1: A[:,0] (2x1) * B[0,:] (1x5) -> 2x5, rank 10
    for ii in range(n):
        for kk in range(p):
            emit([ii*m + 0], [0*p + kk], [kk*n + ii])

    # Block 2: A[:,1:3] (2x2) * B[1:3,:] (2x5) -> 2x5, rank 18
    # Split B[1:3,:] into B[1:3,0:2] (2x2), B[1:3,2:4] (2x2), B[1:3,4] (2x1)
    strassen_222(0, 1, 0)   # A' * B[1:3,0:2]
    strassen_222(0, 1, 2)   # A' * B[1:3,2:4]
    schoolbook_221(0, 1, 4) # A' * B[1:3,4]

    assert col == R, f"Column count mismatch: {col} != {R}"
    assert time.time() < t_deadline

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
