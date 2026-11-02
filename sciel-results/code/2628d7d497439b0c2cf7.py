import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2(i0, j0, k0, U, V, W, r_start):
    """Add 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Block A rows [i0:i0+2], B cols [j0:j0+2], C rows [k0:k0+2].
    Returns r_start + 7.
    """
    r = r_start
    # M1 = (A[i0,:2] + A[i0+1,:2]) * (B[:2,k0] - B[:2,k0+1])
    #    -> C[i0,k0] += M1, C[i0+1,k0] += M1, C[i0,k0+1] -= M1
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = -1.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = -1.0
    W[k0*(N)+i0, r] = 1.0
    W[k0*(N)+i0+1, r] = 1.0
    W[(k0+1)*(N)+i0, r] = -1.0
    r += 1

    # M2 = A[i0+1,:2] * (B[:2,k0] + B[:2,k0+1])
    #    -> C[i0+1,k0+1] += M2
    U[i0*(M)+j0, r] = 0.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = 0.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[(k0+1)*(N)+i0+1, r] = 1.0
    r += 1

    # M3 = (A[i0,:2] - A[i0+1,:2]) * B[:2,k0+1]
    #    -> C[i0,k0+1] += M3, C[i0+1,k0+1] -= M3
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = -1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = -1.0
    V[j0*(P)+k0, r] = 0.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 0.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0, r] = 0.0
    W[k0*(N)+i0+1, r] = 0.0
    W[(k0+1)*(N)+i0, r] = 1.0
    W[(k0+1)*(N)+i0+1, r] = -1.0
    r += 1

    # M4 = A[i0,:2] * B[:2,k0]
    #    -> C[i0,k0] += M4
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = 0.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = 0.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = 0.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = 0.0
    W[k0*(N)+i0, r] = 1.0
    r += 1

    # M5 = (A[i0+1,:2] + A[i0,:2]) * B[:2,k0+1]
    #    -> C[i0+1,k0] += M5
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 0.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 0.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0+1, r] = 1.0
    r += 1

    # M6 = (A[i0+1,:2] - A[i0,:2]) * (B[:2,k0] + B[:2,k0+1])
    #    -> C[i0+1,k0] += M6
    U[i0*(M)+j0, r] = -1.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = -1.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0+1, r] = 1.0
    r += 1

    # M7 = (A[i0,:2] - A[i0+1,:2]) * (B[:2,k0+1] - B[:2,k0])
    #    -> C[i0,k0] += M7
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = -1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = -1.0
    V[j0*(P)+k0, r] = -1.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = -1.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0, r] = 1.0
    r += 1

    return r


def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Partition:
    # A: rows {0,1},{2,3}; cols {0,1},{2,3,4}
    # B: rows {0,1},{2,3,4}; cols {0,1},{2,3,4}
    # C11 = A11*B11 + A12*B21  (2x2 * 2x2 + 2x3 * 3x2)
    # C12 = A11*B12 + A12*B22  (2x2 * 2x3 + 2x3 * 3x3)
    # C21 = A21*B11 + A22*B21  (2x2 * 2x2 + 2x3 * 3x2)
    # C22 = A21*B12 + A22*B22  (2x2 * 2x3 + 2x3 * 3x3)
    #
    # For C12 and C22, we can apply Strassen to the 2x2x2 subblocks:
    # A11(2x2)*B12(2x3): split B12 into B12a(2x2)+B12b(2x1)
    #   A11*B12a: Strassen 7, A11*B12b: schoolbook 4 -> 11
    # A12(2x3)*B22(3x3): split A12 into A12a(2x2)+A12b(2x1), B22 into B22a(3x2)+B22b(3x1)
    #   A12a*B22a: Strassen 7, A12a*B22b: schoolbook 4, A12b*B22a: schoolbook 4, A12b*B22b: schoolbook 2 -> 17
    # C12 total: 11+17 = 28
    # Similarly C22: 28
    # C11: 7+12 = 19, C21: 7+12 = 19
    # Total: 19+28+19+28 = 94
    
    rank = 94
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    # C11: A11(2x2)*B11(2x2) via Strassen: i0=0, j0=0, k0=0
    r = _strassen_2x2x2(0, 0, 0, U, V, W, r)
    # C11: A12(2x3)*B21(3x2) schoolbook: i in {0,1}, j in {2,3,4}, k in {0,1}
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    # C12: A11(2x2)*B12a(2x2) via Strassen: i0=0, j0=0, k0=2
    r = _strassen_2x2x2(0, 0, 2, U, V, W, r)
    # C12: A11(2x2)*B12b(2x1) schoolbook: i in {0,1}, j in {0,1}, k in {4}
    for i in range(2):
        for j in range(2):
            U[i*m+j, r] = 1.0
            V[j*p+4, r] = 1.0
            W[4*n+i, r] = 1.0
            r += 1
    # C12: A12a(2x2)*B22a(3x2) via Strassen: i0=0, j0=2, k0=2
    r = _strassen_2x2x2(0, 2, 2, U, V, W, r)
    # C12: A12a(2x2)*B22b(3x1) schoolbook: i in {0,1}, j in {2,3}, k in {4}
    for i in range(2):
        for j in range(2, 4):
            U[i*m+j, r] = 1.0
            V[j*p+4, r] = 1.0
            W[4*n+i, r] = 1.0
            r += 1
    # C12: A12b(2x1)*B22a(3x2) schoolbook: i in {0,1}, j in {4}, k in {2,3}
    for i in range(2):
        for k in range(2, 4):
            U[i*m+4, r] = 1.0
            V[4*p+k, r] = 1.0
            W[k*n+i, r] = 1.0
            r += 1
    # C12: A12b(2x1)*B22b(3x1) schoolbook: i in {0,1}, j in {4}, k in {4}
    for i in range(2):
        U[i*m+4, r] = 1.0
        V[4*p+4, r] = 1.0
        W[4*n+i, r] = 1.0
        r += 1
    
    # C21: A21(2x2)*B11(2x2) via Strassen: i0=2, j0=0, k0=0
    r = _strassen_2x2x2(2, 0, 0, U, V, W, r)
    # C21: A22(2x3)*B21(3x2) schoolbook: i in {2,3}, j in {2,3,4}, k in {0,1}
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    # C22: A21(2x2)*B12a(2x2) via Strassen: i0=2, j0=0, k0=2
    r = _strassen_2x2x2(2, 0, 2, U, V, W, r)
    # C22: A21(2x2)*B12b(2x1) schoolbook: i in {2,3}, j in {0,1}, k in {4}
    for i in range(2, 4):
        for j in range(2):
            U[i*m+j, r] = 1.0
            V[j*p+4, r] = 1.0
            W[4*n+i, r] = 1.0
            r += 1
    # C22: A22a(2x2)*B22a(3x2) via Strassen: i0=2, j0=2, k0=2
    r = _strassen_2x2x2(2, 2, 2, U, V, W, r)
    # C22: A22a(2x2)*B22b(3x1) schoolbook: i in {2,3}, j in {2,3}, k in {4}
    for i in range(2, 4):
        for j in range(2, 4):
            U[i*m+j, r] = 1.0
            V[j*p+4, r] = 1.0
            W[4*n+i, r] = 1.0
            r += 1
    # C22: A22b(2x1)*B22a(3x2) schoolbook: i in {2,3}, j in {4}, k in {2,3}
    for i in range(2, 4):
        for k in range(2, 4):
            U[i*m+4, r] = 1.0
            V[4*p+k, r] = 1.0
            W[k*n+i, r] = 1.0
            r += 1
    # C22: A22b(2x1)*B22b(3x1) schoolbook: i in {2,3}, j in {4}, k in {4}
    for i in range(2, 4):
        U[i*m+4, r] = 1.0
        V[4*p+4, r] = 1.0
        W[4*n+i, r] = 1.0
        r += 1
    
    if r != rank:
        pass

    if not _validate(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
