import time
import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Block structure:
    - j=0: schoolbook 2x1x5, rank 10
    - j in {1,2}, k in {0,1}: Strassen 2x2x2, rank 7
    - j in {1,2}, k=2: schoolbook 2x2x1, rank 4
    - j in {1,2}, k in {3,4}: Strassen 2x2x2, rank 7
    Total: 28
    """
    n, m, p = 2, 3, 5
    R = 28

    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)

    # Segment 1: column indices 0-9 for j=0 schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0, i * p + k] = 1
            V[k, i * p + k] = 1
            W[k * n + i, i * p + k] = 1

    # Strassen subroutine for 2x2x2 blocks at j0, k0
    def emit_strassen(j0, k0, c0):
        # Term 0
        U[0*m + j0, c0] = 1
        U[1*m + j0+1, c0] = 1
        V[j0*p + k0, c0] = 1
        V[(j0+1)*p + k0+1, c0] = 1
        W[k0*n, c0] = 1
        W[(k0+1)*n + 1, c0] = 1
        # Term 1
        U[1*m + j0, c0+1] = 1
        U[1*m + j0+1, c0+1] = 1
        V[j0*p + k0, c0+1] = 1
        W[k0*n + 1, c0+1] = 1
        W[(k0+1)*n + 1, c0+1] = -1
        # Term 2
        U[0*m + j0, c0+2] = 1
        V[j0*p + k0+1, c0+2] = 1
        V[(j0+1)*p + k0+1, c0+2] = -1
        W[(k0+1)*n, c0+2] = 1
        W[(k0+1)*n + 1, c0+2] = 1
        # Term 3
        U[1*m + j0+1, c0+3] = 1
        V[(j0+1)*p + k0, c0+3] = 1
        V[j0*p + k0, c0+3] = -1
        W[k0*n, c0+3] = 1
        W[k0*n + 1, c0+3] = 1
        # Term 4
        U[0*m + j0, c0+4] = 1
        U[0*m + j0+1, c0+4] = 1
        V[(j0+1)*p + k0+1, c0+4] = 1
        W[k0*n, c0+4] = -1
        W[(k0+1)*n, c0+4] = 1
        # Term 5
        U[1*m + j0, c0+5] = 1
        U[0*m + j0, c0+5] = -1
        V[j0*p + k0, c0+5] = 1
        V[j0*p + k0+1, c0+5] = 1
        W[(k0+1)*n + 1, c0+5] = 1
        # Term 6
        U[0*m + j0+1, c0+6] = 1
        U[1*m + j0+1, c0+6] = -1
        V[(j0+1)*p + k0, c0+6] = 1
        V[(j0+1)*p + k0+1, c0+6] = 1
        W[k0*n, c0+6] = 1

    emit_strassen(1, 0, 10)
    # Segment 3: columns 17-20 for j in {1,2}, k=2
    for idx, (i, j) in enumerate([(0,1),(0,2),(1,1),(1,2)]):
        U[i*m + j, 17 + idx] = 1
        V[j*p + 2, 17 + idx] = 1
        W[2*n + i, 17 + idx] = 1

    emit_strassen(1, 3, 21)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
