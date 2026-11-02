"""Matrix multiplication 4x4 x 4x6 tensor decomposition using Kronecker product."""
import numpy as np
import random
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W (each 4x7) for 2x2 x 2x2 matrix multiplication (Strassen)."""
    # Strassen's 7 rank-1 terms for M_{2,2,2}
    # Tensor T[i,j,k] = 1 (the matmul tensor)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] where n=m=p=2
    # Convention: W[k*n + i, r]
    
    # Strassen formulas:
    # M1 = (A11 - A22)(B12 + B22)
    # M2 = (A11 + A22)B22
    # M3 = A11(B11 - B21)
    # M4 = A22(B21 + B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    
    # U indices: i*2 + j (A_{ij})
    # V indices: j*2 + k (B_{jk})
    # W indices: k*2 + i (C_{ki} in trace convention W[k*n+i])
    
    # M1: A11-A22 times B12+B22 -> C11 + C22 (with signs)
    U[0, 0] = 1   # A11
    U[3, 0] = -1  # A22
    V[1, 0] = 1   # B12
    V[3, 0] = 1   # B22
    W[0, 0] = 1   # C11 (k=0,i=0)
    W[3, 0] = 1   # C22 (k=1,i=1)
    
    # M2: (A11+A22)*B22 -> C21 + C22 (with sign)
    U[0, 1] = 1
    U[3, 1] = 1
    V[3, 1] = 1
    W[1, 1] = 1   # C21 (k=0,i=1)
    W[2, 1] = -1  # C12 (k=1,i=0)
    
    # M3: A11*(B11-B21) -> C11 - C12
    U[0, 2] = 1
    V[0, 2] = 1
    V[2, 2] = -1
    W[0, 2] = 1   # C11
    W[1, 2] = -1  # C12 (k=1,i=0)
    
    # M4: A22*(B21+B11) -> C11 + C21
    U[3, 3] = 1
    V[0, 3] = 1
    V[2, 3] = 1
    W[0, 3] = 1   # C11
    W[1, 3] = 1   # C21 (k=0,i=1)
    
    # M5: (A11+A12)*B22 -> C11 - C12
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C11
    W[1, 4] = 1   # C12 (k=1,i=0)
    
    # M6: (A21-A11)*(B11+B12) -> C22 + C21
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[2, 5] = 1   # C22 (k=1,i=1)
    W[3, 5] = 1   # C21 (k=0,i=1)
    
    # M7: (A12-A22)*(B21+B22) -> C11 + C22
    U[1, 6] = 1
    U[3, 6] = -1
    V[2, 6] = 1
    V[3, 6] = 1
    W[0, 6] = 1   # C11
    W[3, 6] = 1   # C22 (k=1,i=1)
    
    return U, V, W


def _matmul_2x2x3():
    """Return U, V, W for 2x2 x 2x3 matrix multiplication (schoolbook, rank 12)."""
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W


def _kron_product(U1, V1, W1, U2, V2, W2):
    """
    Compute Kronecker product of two tensor decompositions.
    If T1 = sum_r U1_r ⊗ V1_r ⊗ W1_r and T2 = sum_s U2_s ⊗ V2_s ⊗ W2_s,
    then T1 ⊗ T2 = sum_{r,s} (U1_r ⊗ U2_s) ⊗ (V1_r ⊗ V2_s) ⊗ (W1_r ⊗ W2_s)
    
    U1 is (a, r), U2 is (b, s) -> result U is (a*b, r*s)
    V1 is (c, r), V2 is (d, s) -> result V is (c*d, r*s)
    W1 is (e, r), W2 is (f, s) -> result W is (e*f, r*s)
    """
    r1, r2 = U1.shape[1], U2.shape[1]
    rank = r1 * r2
    
    U = np.zeros((U1.shape[0] * U2.shape[0], rank), dtype=float)
    V = np.zeros((V1.shape[0] * V2.shape[0], rank), dtype=float)
    W = np.zeros((W1.shape[0] * W2.shape[0], rank), dtype=float)
    
    for r in range(r1):
        for s in range(r2):
            idx = r * r2 + s
            U[:, idx] = np.kron(U1[:, r], U2[:, s])
            V[:, idx] = np.kron(V1[:, r], V2[:, s])
            W[:, idx] = np.kron(W1[:, r], W2[:, s])
    
    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of M_{n,m,p}."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        for i in range(n):
            Wc_col_i = Wc[:, i]
            M = Vc * Wc_col_i[None, :]
            T[i, :, :] += Uc[i, :][:, None] * M
    return np.allclose(T, 1.0, atol=1e-10)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W


def _try_remove_terms(U, V, W, n, m, p, deadline):
    """Try to remove rank terms one at a time if the decomposition remains valid."""
    rank = U.shape[1]
    for _ in range(rank):  # at most 'rank' removal attempts
        if time.time() > deadline:
            break
        removed = False
        for c in range(rank - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if this term is zero
            if np.allclose(U[:, c], 0) or np.allclose(V[:, c], 0) or np.allclose(W[:, c], 0):
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                rank -= 1
                removed = True
                break
        if not removed:
            # Try removing each non-zero term
            for c in range(rank - 1, -1, -1):
                if time.time() > deadline:
                    break
                U_test = np.delete(U, c, axis=1)
                V_test = np.delete(V, c, axis=1)
                W_test = np.delete(W, c, axis=1)
                if _validate(U_test, V_test, W_test, n, m, p):
                    U = U_test
                    V = V_test
                    W = W_test
                    rank -= 1
                    removed = True
                    break
            if not removed:
                break
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Strategy: Kronecker product of Strassen (2x2x2, rank 7) and schoolbook (2x2x3, rank 12)
    # This gives M_{4,4,6} with rank 7*12 = 84, beating schoolbook rank 96.
    
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _matmul_2x2x3()
    
    U, V, W = _kron_product(U1, V1, W1, U2, V2, W2)
    
    # Verify
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank by removing redundant terms
    U, V, W = _try_remove_terms(U, V, W, n, m, p, deadline)
    
    # Convert to lists with half-integer precision
    def to_list(arr):
        arr = np.round(arr * 2) / 2.0
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
