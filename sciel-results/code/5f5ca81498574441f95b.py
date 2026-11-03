"""
Rank-77 exact tensor decomposition for 4x4 x 4x6 matrix multiplication.
Uses Kronecker product of Strassen 2x2x2 (rank 7) and 2x2x3 (rank 11).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition (rank 7).
    U: 4x7, indices row = i*2 + j (A[i,j])
    V: 4x7, indices row = j*2 + k (B[j,k])  
    W: 4x7, indices row = k*2 + i (C[k,i] -> trace convention W[k*n+i])
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_2x2x3():
    """
    Returns U, V, W for 2x2x3 multiplication.
    A is 2x2, B is 2x3, C is 2x3.
    U: 4x11, row = i*2 + j (A[i,j])
    V: 6x11, row = j*3 + k (B[j,k])
    W: 6x11, row = k*2 + i (C[k,i] -> trace convention)
    """
    U1, V1, W1 = _get_strassen_2x2()
    
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)
    W2 = np.zeros((6, 11), dtype=float)
    
    # Copy Strassen part for B columns 0,1
    # B indices: j*3+k. For k=0,1: j=0->0,1; j=1->3,4
    # W indices: k*2+i. For k=0,1: i=0->0,1; i=1->2,3
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]  # j=0,k=0
        V2[1, c] = V1[1, c]  # j=0,k=1
        V2[3, c] = V1[2, c]  # j=1,k=0
        V2[4, c] = V1[3, c]  # j=1,k=1
        W2[0, c] = W1[0, c]  # k=0,i=0
        W2[1, c] = W1[1, c]  # k=0,i=1
        W2[2, c] = W1[2, c]  # k=1,i=0
        W2[3, c] = W1[3, c]  # k=1,i=1
    
    # Schoolbook for B column 2 (k=2)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # W indices for k=2: k*2+i -> 4+i
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U2[0, 7] = 1
    V2[2, 7] = 1   # j=0,k=2 -> 0*3+2=2
    W2[4, 7] = 1   # k=2,i=0 -> 2*2+0=4
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U2[1, 8] = 1
    V2[5, 8] = 1   # j=1,k=2 -> 1*3+2=5
    W2[4, 8] = 1   # k=2,i=0 -> 2*2+0=4
    
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U2[2, 9] = 1
    V2[2, 9] = 1   # j=0,k=2 -> 0*3+2=2
    W2[5, 9] = 1   # k=2,i=1 -> 2*2+1=5
    
    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U2[3, 10] = 1
    V2[5, 10] = 1  # j=1,k=2 -> 1*3+2=5
    W2[5, 10] = 1  # k=2,i=1 -> 2*2+1=5
    
    return U2, V2, W2

def _kron_decomposition():
    """
    Constructs the full 4x4x6 decomposition using Kronecker products.
    """
    U1, V1, W1 = _get_strassen_2x2()
    U2, V2, W2 = _get_2x2x3()
    
    rank1 = U1.shape[1]  # 7
    rank2 = U2.shape[1]  # 11
    total_rank = rank1 * rank2  # 77
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)
    W = np.zeros((24, total_rank), dtype=float)
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            
            # U: A[i,j] where i=2*i1+i2, j=2*j1+j2
            # U1 index: i1*2+j1, U2 index: i2*2+j2
            nz_u1 = np.nonzero(U1[:, c1])[0]
            nz_u2 = np.nonzero(U2[:, c2])[0]
            for idx1 in nz_u1:
                val1 = U1[idx1, c1]
                i1, j1 = divmod(idx1, 2)
                for idx2 in nz_u2:
                    val2 = U2[idx2, c2]
                    i2, j2 = divmod(idx2, 2)
                    row = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    U[row, c] = val1 * val2
            
            # V: B[j,k] where j=2*j1+j2, k=3*k1+k2
            # V1 index: j1*2+k1, V2 index: j2*3+k2
            nz_v1 = np.nonzero(V1[:, c1])[0]
            nz_v2 = np.nonzero(V2[:, c2])[0]
            for idx1 in nz_v1:
                val1 = V1[idx1, c1]
                j1, k1 = divmod(idx1, 2)
                for idx2 in nz_v2:
                    val2 = V2[idx2, c2]
                    j2, k2 = divmod(idx2, 3)
                    row = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
                    V[row, c] = val1 * val2
            
            # W: C[i,k] where i=2*i1+i2, k=3*k1+k2
            # W1 index: k1*2+i1, W2 index: k2*2+i2
            nz_w1 = np.nonzero(W1[:, c1])[0]
            nz_w2 = np.nonzero(W2[:, c2])[0]
            for idx1 in nz_w1:
                val1 = W1[idx1, c1]
                k1, i1 = divmod(idx1, 2)
                for idx2 in nz_w2:
                    val2 = W2[idx2, c2]
                    k2, i2 = divmod(idx2, 2)
                    row = (k1 * 3 + k2) * 4 + (i1 * 2 + i2)
                    W[row, c] = val1 * val2
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """
    Verifies the decomposition using random test cases.
    n=4, m=4, p=6
    """
    n, m, p = 4, 4, 6
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            alpha = np.sum(A * U[:, r].reshape(n, m))
            beta = np.sum(B * V[:, r].reshape(m, p))
            if alpha != 0 and beta != 0:
                C_computed += alpha * beta * W[:, r].reshape(p, n).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    # Also test with integer matrices
    A_int = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=float)
    B_int = np.array([[1,0,0,1,0,1],[0,1,0,0,1,0],[1,0,1,0,0,1],[0,1,0,1,0,0]], dtype=float)
    C_expected = A_int @ B_int
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha = np.sum(A_int * U[:, r].reshape(n, m))
        beta = np.sum(B_int * V[:, r].reshape(m, p))
        if alpha != 0 and beta != 0:
            C_computed += alpha * beta * W[:, r].reshape(p, n).T
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

def _schoolbook_fallback():
    """Returns schoolbook rank n*m*p decomposition as fallback."""
    n, m, p = 4, 4, 6
    rank = n * m * p  # 96
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    try:
        U, V, W = _kron_decomposition()
        
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded during construction")
            
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            return _schoolbook_fallback()
    except Exception:
        return _schoolbook_fallback()
# EVOLVE-BLOCK-END
