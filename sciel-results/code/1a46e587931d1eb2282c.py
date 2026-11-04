import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _generate_target_tensor(n, m, p):
    """Generate the target tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    T = _generate_target_tensor(n, m, p)
    return np.allclose(T_hat, T, atol=1e-9, rtol=1e-9)

def _build_strassen_2x2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 = 2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7)
    where U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Multiplier 0: m1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # Multiplier 1: m2 = (a10+a11)*b00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # Multiplier 2: m3 = a00*(b01-b11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # Multiplier 3: m4 = a11*(b10-b00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # Multiplier 4: m5 = (a00+a01)*b11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # Multiplier 5: m6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # Multiplier 6: m7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W

def _build_2x3x7_decomposition():
    """
    Build a rank-39 decomposition for 2x3 x 3x7 = 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen for two 2x2x2 blocks: (i in {0,1}, j in {0,1}, k in {0,1}) and (i in {0,1}, j in {0,1}, k in {2,3})
    - This replaces 16 schoolbook terms with 14 Strassen terms (save 2)
    - Remaining terms use schoolbook
    
    Total rank: 42 - 2 = 40... wait, let me recount.
    Actually: 42 total schoolbook terms. Two Strassen blocks each save 1 term (8->7).
    So rank = 42 - 2 = 40.
    
    But we can do better by using a different grouping. Let me try:
    - Use Strassen for (i in {0,1}, j in {0,1}, k in {0,1}) and (i in {0,1}, j in {2,2}, k in {0,1})?
    No, j=2 is only 1 column, can't form 2x2.
    
    Let me try another approach:
    - Partition j into {0,1} and {2}
    - Partition k into {0,1}, {2,3}, {4,5}, {6}
    
    For j in {0,1}, k in {0,1}: 2x2*2x2, Strassen rank 7
    For j in {0,1}, k in {2,3}: 2x2*2x2, Strassen rank 7
    For j in {0,1}, k in {4,5}: 2x2*2x2, Strassen rank 7
    For j in {0,1}, k in {6}: 2x2*2x1, rank 4 (schoolbook)
    For j in {2}, k in {0,1,2,3,4,5,6}: 2x1*1x7, rank 7 (schoolbook)
    
    Total: 7+7+7+4+7 = 32? That seems too low. Let me verify.
    
    Actually, 2x2*2x1: the tensor has 2*2*1=4 nonzero entries, rank is at least max(4,2,2)=4.
    Schoolbook gives rank 4, which is optimal.
    
    2x1*1x7: tensor has 2*1*7=14 nonzero entries... wait no.
    For 2x1 * 1x7: A is 2x1, B is 1x7, C is 2x7.
    T[i*1+0, 0*7+k, k*2+i] = 1 for i in {0,1}, k in {0,...,6}
    That's 2*7=14 terms. Rank lower bound: max(2, 7, 14) = 14.
    Schoolbook rank is 14. So rank is 14.
    
    Total: 7+7+7+4+14 = 39.
    
    Let me implement this.
    """
    n, m, p = 2, 3, 7
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    col = 0
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*3+j, col] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V[j*7+k, col] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[k*2+i, col] = W_s[k*2+i, r]
        col += 1
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*3+j, col] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V[j*7+(k+2), col] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[(k+2)*2+i, col] = W_s[k*2+i, r]
        col += 1
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> Strassen
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U[i*3+j, col] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V[j*7+(k+4), col] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W[(k+4)*2+i, col] = W_s[k*2+i, r]
        col += 1
    
    # Block 4: i in {0,1}, j in {0,1}, k in {6} -> schoolbook 2x2*2x1, rank 4
    for i in range(2):
        for j in range(2):
            U[i*3+j, col] = 1.0
            V[j*7+6, col] = 1.0
            W[6*2+i, col] = 1.0
            col += 1
    
    # Block 5: i in {0,1}, j in {2}, k in {0,...,6} -> schoolbook 2x1*1x7, rank 14
    for i in range(2):
        for k in range(7):
            U[i*3+2, col] = 1.0
            V[2*7+k, col] = 1.0
            W[k*2+i, col] = 1.0
            col += 1
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return _build_schoolbook(n, m, p)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        U, V, W = _build_2x3x7_decomposition()
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
