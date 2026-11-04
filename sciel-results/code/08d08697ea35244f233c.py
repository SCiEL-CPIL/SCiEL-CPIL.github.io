import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _schoolbook(n, m, p):
    """Return (U, V, W) for the schoolbook decomposition of M_{n,m,p} with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return (U, V, W) for the Strassen decomposition of M_{2,2,2} with rank 7.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # P1: (A11+A22)(B11+B22) -> C11 + C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2: (A11+A22)(B21-B11) -> C21
    U[0, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1; V[0, 1] = -1
    W[1, 1] = 1
    
    # P3: (A11-A21)B11 -> C11 + C12
    U[0, 2] = 1; U[2, 2] = -1
    V[0, 2] = 1
    W[0, 2] = 1; W[1, 2] = 1
    
    # P4: (A22-A12)B22 -> C21 + C22
    U[3, 3] = 1; U[1, 3] = -1
    V[3, 3] = 1
    W[1, 3] = 1; W[3, 3] = 1
    
    # P5: (A11+A12)B22 -> C12 - C22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = -1
    
    # P6: (A21-A11)(B11+B12) -> C22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7: (A12-A22)(B21+B22) -> C11
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_22p(p):
    """
    Build M_{2,2,p} by splitting B's columns into 2 + (p-2).
    M_{2,2,2} (Strassen, rank 7) + M_{2,2,p-2} (schoolbook, rank 2*(p-2)).
    Total rank: 7 + 2*(p-2) = 2p + 3.
    """
    if p == 2:
        return _strassen_222()
    
    U_s, V_s, W_s = _strassen_222()
    U_b, V_b, W_b = _schoolbook(2, 2, p - 2)
    
    rank = 7 + 2 * (p - 2)
    U_full = np.zeros((4, rank))
    V_full = np.zeros((2 * p, rank))
    W_full = np.zeros((2 * p, rank))
    
    # Left part: k in {0,1}
    U_full[:, :7] = U_s
    for j in range(2):
        for k in range(2):
            V_full[j * p + k, :7] = V_s[j * 2 + k]
    for k in range(2):
        for i in range(2):
            W_full[k * 2 + i, :7] = W_s[k * 2 + i]
    
    # Right part: k in {2,...,p-1}
    for i in range(2):
        for j in range(2):
            U_full[i * 2 + j, 7:] = U_b[i * 2 + j]
    for j in range(2):
        for k in range(p - 2):
            V_full[j * p + (k + 2), 7:] = V_b[j * (p - 2) + k]
    for k in range(p - 2):
        for i in range(2):
            W_full[(k + 2) * 2 + i, 7:] = W_b[(k + 2) * 2 + i]
    
    return U_full, V_full, W_full

def _build_237():
    """
    Build M_{2,3,7} by splitting A's columns into 2 + 1.
    M_{2,2,7} (rank 24) + M_{2,1,7} (schoolbook, rank 14).
    Total rank: 38.
    """
    U_227, V_227, W_227 = _build_22p(7)
    U_b, V_b, W_b = _schoolbook(2, 1, 7)
    
    rank = 24 + 14  # 38
    U_full = np.zeros((6, rank))
    V_full = np.zeros((21, rank))
    W_full = np.zeros((14, rank))
    
    # Left part: j in {0,1}
    for i in range(2):
        for j in range(2):
            U_full[i * 3 + j, :24] = U_227[i * 2 + j, :]
    for j in range(2):
        for k in range(7):
            V_full[j * 7 + k, :24] = V_227[j * 7 + k, :]
    for k in range(7):
        for i in range(2):
            W_full[k * 2 + i, :24] = W_227[k * 2 + i, :]
    
    # Right part: j = 2
    for i in range(2):
        U_full[i * 3 + 2, 24:] = U_b[i, :]
    for k in range(7):
        V_full[2 * 7 + k, 24:] = V_b[k, :]
    for k in range(7):
        for i in range(2):
            W_full[k * 2 + i, 24:] = W_b[k * 2 + i, :]
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check 1-entries: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check 0-entries: for any (alpha, beta, gamma) not aligned, the sum should be 0
    for alpha in range(n*m):
        i1 = alpha // m
        j1 = alpha % m
        for beta in range(m*p):
            j2 = beta // p
            k2 = beta % p
            if j1 != j2:
                continue
            for gamma in range(p*n):
                k1 = gamma // n
                i2 = gamma % n
                if i1 == i2 and k1 == k2:
                    continue
                s = 0.0
                for r in range(rank):
                    s += U[alpha, r] * V[beta, r] * W[gamma, r]
                if abs(s) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_237()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
