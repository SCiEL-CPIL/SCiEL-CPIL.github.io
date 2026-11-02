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
    # In tensor: U[i*m+j], V[j*p+k], W[k*n+i]
    # A indices: i in {i0, i0+1}, j in {j0, j0+1}
    # B indices: j in {j0, j0+1}, k in {k0, k0+1}
    # C indices: k in {k0, k0+1}, i in {i0, i0+1}
    
    # M1: (A[i0,:]+A[i0+1,:]) * (B[:,k0]-B[:,k0+1])
    # U: rows i0,j0 and i0+1,j0 -> 1; rows i0,j0+1 and i0+1,j0+1 -> 1
    # V: cols j0,k0 -> 1; j0,k0+1 -> -1; j0+1,k0 -> 1; j0+1,k0+1 -> -1
    # W: k0,i0 -> 1; k0,i0+1 -> 1; k0+1,i0 -> -1
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
    
    # Block partitioning:
    # A (4x5): rows 0-1, 2-3; cols 0-1, 2-4
    # B (5x5): rows 0-1, 2-4; cols 0-1, 2-4
    # C (4x5): rows 0-1, 2-3; cols 0-1, 2-4
    #
    # C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    # C12 (2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    # C21 (2x2) = A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    # C22 (2x3) = A21(2x2)*B12(2x3) + A22(2x3)*B22(3x3)
    #
    # Rank counts:
    # C11: Strassen 2x2x2 = 7 (for A11*B11) + schoolbook 2*3*2=12 (for A12*B21) = 19
    # C12: schoolbook 2*2*3=12 (A11*B12) + 2*3*3=18 (A12*B22) = 30
    # C21: Strassen 2x2x2 = 7 (for A21*B11) + schoolbook 2*3*2=12 (for A22*B21) = 19
    # C22: schoolbook 2*2*3=12 (A21*B12) + 2*3*3=18 (A22*B22) = 30
    # Total: 19+30+19+30 = 98
    
    # Wait, that's still high. Let me reconsider.
    # Actually for C11: A11*B11 is 2x2*2x2->2x2 (Strassen rank 7)
    #                 A12*B21 is 2x3*3x2->2x2 (schoolbook rank 2*3*2=12)
    # So C11 needs 7+12=19 terms.
    
    # Total: 98. Still better than 100 but not great.
    # Let me try a different split: A cols 2+3, B rows 2+3, B cols 2+3
    # Actually the issue is the 2x3 and 3x3 blocks are expensive.
    
    # Alternative: split A cols 3+2, B rows 3+2, B cols 2+3
    # A11(2x3), A12(2x2), A21(2x3), A22(2x2)
    # B11(3x2), B12(3x3), B21(2x2), B22(2x3)
    # C11(2x2)=A11*B11+A12*B21: 2x3*3x2 (18) + 2x2*2x2 (Strassen 7) = 25
    # Not better.
    
    # Let me just go with the 98 construction.
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
    
    # Validate
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
