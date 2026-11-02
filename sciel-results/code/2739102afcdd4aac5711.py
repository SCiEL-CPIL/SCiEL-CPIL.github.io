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


def _schoolbook(n, m, p):
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
    return U, V, W


def _build_decomp_98():
    """Build the rank-98 decomposition: Strassen on 2x2x2 blocks + schoolbook for the rest."""
    n, m, p = N, M, P
    rank = 98
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
    
    # C12: A11(2x2)*B12(2x3) schoolbook: i in {0,1}, j in {0,1}, k in {2,3,4}
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    # C12: A12(2x3)*B22(3x3) schoolbook: i in {0,1}, j in {2,3,4}, k in {2,3,4}
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
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
    
    # C22: A21(2x2)*B12(2x3) schoolbook: i in {2,3}, j in {0,1}, k in {2,3,4}
    for i in range(2, 4):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    # C22: A22(2x3)*B22(3x3) schoolbook: i in {2,3}, j in {2,3,4}, k in {2,3,4}
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    return U, V, W


def _build_decomp_96():
    """Try a slightly different partition to see if we can get rank 96.
    Split A cols 0-1, 2-4; B rows 0-1, 2-4; B cols 0-1, 2-4.
    But use Strassen on all four 2x2x2 sub-blocks where possible.
    Actually, let's try: A rows 0-1,2-3; A cols 0-1,2-4; B rows 0-1,2-4; B cols 0-1,2-4.
    C11(2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    C12(2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    C21(2x2) = A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    C22(2x3) = A21(2x2)*B12(2x3) + A22(2x3)*B22(3x3)
    
    For C11: Strassen(2x2x2)=7 + schoolbook(2x3x2)=12 => 19
    For C12: schoolbook(2x2x3)=12 + schoolbook(2x3x3)=18 => 30
    For C21: Strassen(2x2x2)=7 + schoolbook(2x3x2)=12 => 19
    For C22: schoolbook(2x2x3)=12 + schoolbook(2x3x3)=18 => 30
    Total: 19+30+19+30 = 98. Same as before.
    
    Let's try a different split: A rows 0-2,3; A cols 0-2,3-4; B rows 0-2,3-4; B cols 0-2,3-4.
    This gives 3x2, 2x2, etc. blocks.
    C11(3x3) = A11(3x3)*B11(3x3) + A12(3x2)*B21(2x3)
    C12(3x2) = A11(3x3)*B12(3x2) + A12(3x2)*B22(2x2)
    C21(1x3) = A21(1x3)*B11(3x3) + A22(1x2)*B21(2x3)
    C22(1x2) = A21(1x3)*B12(3x2) + A22(1x2)*B22(2x2)
    
    Rank for 3x3x3 is at most 27 (schoolbook) or maybe less with Strassen-like? 
    Standard Strassen is for 2x2x2. For 3x3x3, the best known is 23 or so, but we don't have that.
    Let's stick to 2x2x2 Strassen blocks.
    
    Another idea: Use Strassen on the 2x2x2 part of C11, C21, and also try to use Strassen-like 
    tricks on the 2x3x2 or 3x3x2 blocks? No, Strassen is specifically for 2x2x2.
    
    Let's try to combine the schoolbook parts more efficiently.
    Actually, the current 98 is quite good. Let's just verify it and return it.
    """
    # For now, return the same as 98. We can try to optimize later if time permits.
    return _build_decomp_98()


def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Build the rank-98 decomposition
    U, V, W = _build_decomp_98()
    
    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
