import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = 130  # safety margin before 150s kill


def _verify_vectorized(n, m, p, U, V, W):
    """Fast vectorized verification of the tensor decomposition."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    rank = Ua.shape[1]
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(Ua[a_idx], Va[b_idx] * Wa[c_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_strassen_2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Returns U, V, W as lists of lists.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A: 00, 01, 10, 11 -> 0, 1, 2, 3
    B: 00, 01, 10, 11 -> 0, 1, 2, 3
    C: 00, 01, 10, 11 -> 0, 1, 2, 3 (Trace: C[col*2+row])
    C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_rank23_3x3():
    """
    Build a rank-27 (schoolbook) decomposition for 3x3 matrix multiplication.
    U, V, W: (9, 27)
    Trace convention for W: C[col*3+row]
    """
    U = np.zeros((9, 27), dtype=np.float64)
    V = np.zeros((9, 27), dtype=np.float64)
    W = np.zeros((9, 27), dtype=np.float64)
    
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, idx] = 1
                V[j * 3 + k, idx] = 1
                W[k * 3 + i, idx] = 1
                idx += 1
                
    return U.tolist(), V.tolist(), W.tolist()


def _build_kronecker_6x6(n, m, p):
    """
    Build Kronecker product decomposition: Strassen(2x2) x Schoolbook(3x3).
    Total rank = 27 * 7 = 189.
    """
    rank = 189
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    idx = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                # A block [I,K]
                a00 = (2 * I) * m + 2 * K
                a01 = (2 * I) * m + 2 * K + 1
                a10 = (2 * I + 1) * m + 2 * K
                a11 = (2 * I + 1) * m + 2 * K + 1
                
                # B block [K,J]
                b00 = (2 * K) * p + 2 * J
                b01 = (2 * K) * p + 2 * J + 1
                b10 = (2 * K + 1) * p + 2 * J
                b11 = (2 * K + 1) * p + 2 * J + 1
                
                # C block [I,J]
                c00 = (2 * J) * n + 2 * I
                c01 = (2 * J + 1) * n + 2 * I
                c10 = (2 * J) * n + (2 * I + 1)
                c11 = (2 * J + 1) * n + (2 * I + 1)
                
                # Strassen's 7 terms
                # M1
                U[a00, idx] = 1; U[a11, idx] = 1
                V[b00, idx] = 1; V[b11, idx] = 1
                W[c00, idx] = 1; W[c11, idx] = 1
                idx += 1
                # M2
                U[a10, idx] = 1; U[a11, idx] = 1
                V[b00, idx] = 1
                W[c10, idx] = 1; W[c11, idx] = -1
                idx += 1
                # M3
                U[a00, idx] = 1
                V[b01, idx] = 1; V[b11, idx] = -1
                W[c01, idx] = 1; W[c11, idx] = 1
                idx += 1
                # M4
                U[a11, idx] = 1
                V[b10, idx] = 1; V[b00, idx] = -1
                W[c00, idx] = 1; W[c10, idx] = 1
                idx += 1
                # M5
                U[a00, idx] = 1; U[a01, idx] = 1
                V[b11, idx] = 1
                W[c00, idx] = -1; W[c01, idx] = 1
                idx += 1
                # M6
                U[a10, idx] = 1; U[a00, idx] = -1
                V[b00, idx] = 1; V[b01, idx] = 1
                W[c11, idx] = 1
                idx += 1
                # M7
                U[a01, idx] = 1; U[a11, idx] = -1
                V[b10, idx] = 1; V[b11, idx] = 1
                W[c00, idx] = 1
                idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_schoolbook(n, m, p):
    """Fallback: naive schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the rank-189 Kronecker(Strassen-2x2 x Schoolbook-3x3) decomposition
    U, V, W = _build_kronecker_6x6(n, m, p)
    
    # Verify correctness with time guard
    if time.time() - start_time < DEADLINE - 10:
        if not _verify_vectorized(n, m, p, U, V, W):
            # Fallback to schoolbook if verification fails
            U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
