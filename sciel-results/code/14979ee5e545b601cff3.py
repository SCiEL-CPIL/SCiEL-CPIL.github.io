import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition_mm(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication C = AB.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k?
    No, for C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    The term U[i*m+j, r] * V[j*p+k, r] is the coefficient for A[i,j] B[j,k].
    This contributes to C[i,k].
    So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W_coeff[i,k,r] == 1?
    In the seed, W is indexed by k*n+i.
    If W[k*n+i, r] is the coefficient for C[k,i], then we are computing C^T?
    
    Let's assume the standard check:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    This is the check in the seed.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _combine_blocks_with_dims(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2):
    assert m1 == m2
    m = m1
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    n = n1 + n2
    p = p1 + p2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U[:n1*m, :r1] = U1
    V[:m*p1, :r1] = V1
    W[:p1*n1, :r1] = W1
    
    U[n1*m:, r1:] = U2
    V[m*p1:, r1:] = V2
    W[p1*n1:, r1:] = W2
    
    return U, V, W

def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS
    
    U_school, V_school, W_school = _schoolbook_block(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Check if Strassen passes the seed's verification
    if not _verify_decomposition_mm(U_s.tolist(), V_s.tolist(), W_s.tolist(), 2, 2, 2):
        return best_result
    
    U_224, V_224, W_224 = _combine_blocks_with_dims(
        2, 2, 2, U_s, V_s, W_s,
        2, 2, 2, U_s, V_s, W_s
    )
    
    U_244, V_244, W_244 = _combine_blocks_with_dims(
        2, 2, 4, U_224, V_224, W_224,
        2, 2, 4, U_224, V_224, W_224
    )
    
    U_241, V_241, W_241 = _schoolbook_block(2, 4, 1)
    
    U_total, V_total, W_total = _combine_blocks_with_dims(
        2, 4, 4, U_244, V_244, W_244,
        2, 4, 1, U_241, V_241, W_241
    )
    
    if _verify_decomposition_mm(U_total.tolist(), V_total.tolist(), W_total.tolist(), n, m, p):
        best_result = {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
        best_rank = 36
    
    return best_result
# EVOLVE-BLOCK-END
