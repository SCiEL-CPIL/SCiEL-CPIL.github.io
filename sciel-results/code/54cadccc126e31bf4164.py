"""Matrix multiplication 4x4x6 - Kronecker structured decomposition with dependency elimination."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """Return (U, V, W) each of shape (4, 7) for the 2x2x2 Strassen decomposition.
    U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    # M3 = a00(b01-b11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    # M4 = a11(b10-b00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    # M5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    # M6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    # M7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    return U, V, W

def _strassen_2x2x3():
    """Return (U, V, W) for 2x2x3: Strassen for cols 0,1 + schoolbook for col 2.
    U shape (4, 11), V shape (6, 11), W shape (6, 11).
    V[j*3+k], W[k*2+i]."""
    U1, V1, W1 = _strassen_2x2x2()
    R = 11
    U = np.zeros((4, R), dtype=int)
    V = np.zeros((6, R), dtype=int)
    W = np.zeros((6, R), dtype=int)
    for c in range(7):
        U[:, c] = U1[:, c]
        V[0, c] = V1[0, c]  # j=0,k=0
        V[1, c] = V1[1, c]  # j=0,k=1
        V[3, c] = V1[2, c]  # j=1,k=0
        V[4, c] = V1[3, c]  # j=1,k=1
        W[0, c] = W1[0, c]  # k=0,i=0
        W[1, c] = W1[1, c]  # k=0,i=1
        W[2, c] = W1[2, c]  # k=1,i=0
        W[3, c] = W1[3, c]  # k=1,i=1
    # Schoolbook for column 2 (4 terms)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    U[0,7]=1; V[2,7]=1; W[4,7]=1   # j=0,k=2 -> V[0*3+2]=V[2]; k=2,i=0 -> W[2*2+0]=W[4]
    U[1,8]=1; V[5,8]=1; W[4,8]=1   # j=1,k=2 -> V[1*3+2]=V[5]
    U[2,9]=1; V[2,9]=1; W[5,9]=1   # i=1,k=2 -> W[2*2+1]=W[5]
    U[3,10]=1; V[5,10]=1; W[5,10]=1
    return U, V, W

def _build_kronecker():
    """Build the full 4x4x6 decomposition via Kronecker product of 2x2x2 and 2x2x3.
    Returns (U, V, W) with shapes (16, 77), (24, 77), (24, 77)."""
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x3()
    R = 7 * 11
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1, c1]
                            u2 = U2[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row, c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1, c1]
                            v2 = V2[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row, c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1, c1]
                            w2 = W2[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row, c] = w1 * w2
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=6):
    """Verify decomposition with random test."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha = np.sum(A * U[:, r].reshape(n, m))
        beta = np.sum(B * V[:, r].reshape(m, p))
        C_computed += alpha * beta * W[:, r].reshape(p, n).T
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _verify_all_ones(U, V, W, n=4, m=4, p=6, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, deadline):
    """Try to reduce rank by detecting linear dependencies among terms.
    Build the 96 x R constraint matrix and iteratively remove columns."""
    n, m, p = 4, 4, 6
    n_eq = n * m * p
    R = U.shape[1]
    # Build constraint matrix M of shape (n_eq, R)
    M = np.zeros((n_eq, R), dtype=float)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    M[eq, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                eq += 1
    target = np.ones(n_eq)
    # Verify
    if not np.allclose(M @ np.ones(R), target, atol=1e-9):
        return U, V, W  # invalid
    # Iteratively remove columns
    active = list(range(R))
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for t in list(active):
            if len(active) <= 1:
                break
            remaining = [x for x in active if x != t]
            M_sub = M[:, remaining]
            r1 = np.linalg.matrix_rank(M_sub)
            r2 = np.linalg.matrix_rank(np.column_stack([M_sub, target]))
            if r1 == r2:
                active = remaining
                improved = True
                break
    # Extract reduced matrices
    R_new = len(active)
    U_new = U[:, active]
    V_new = V[:, active]
    W_new = W[:, active]
    return U_new, V_new, W_new

def construct():
    deadline = time.time() + 130
    # Build Kronecker decomposition
    U, V, W = _build_kronecker()
    # Verify
    if not _verify(U, V, W):
        # Fallback schoolbook
        R = 4*4*6
        U = np.zeros((16, R), dtype=int)
        V = np.zeros((24, R), dtype=int)
        W = np.zeros((24, R), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Try rank reduction
    if time.time() < deadline - 20:
        U, V, W = _try_reduce_rank(U, V, W, deadline)
    # Final verification
    if not _verify(U, V, W):
        R = 4*4*6
        U = np.zeros((16, R), dtype=int)
        V = np.zeros((24, R), dtype=int)
        W = np.zeros((24, R), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
