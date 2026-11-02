import time
import numpy as np

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

def _time_remaining():
    return _DEADLINE - time.time()

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indexing:
    # A[i,j] -> U[i*m + j] = U[2*i+j]
    #   A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B[j,k] -> V[j*p + k] = V[7*j+k]
    # C[i,k] -> W[k*n + i] = W[2*k+i]  (trace convention)
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        k0 = k_start
        k1 = k_start + 1
        
        # M1 = (A00 + A11) * (B0,k0 + B1,k1)
        # C00 += M1, C11 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[0*p + k0, col_start+0] += 1
        V[1*p + k1, col_start+0] += 1
        W[k0*n + 0, col_start+0] += 1   # C[0,k0]
        W[k1*n + 1, col_start+0] += 1   # C[1,k1]
        
        # M2 = (A10 + A11) * B0,k0
        # C10 += M2, C11 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[0*p + k0, col_start+1] += 1
        W[k0*n + 1, col_start+1] += 1   # C[1,k0]
        W[k1*n + 1, col_start+1] -= 1   # C[1,k1]
        
        # M3 = A00 * (B0,k1 - B1,k1)
        # C01 += M3, C11 += M3
        U[0, col_start+2] += 1
        V[0*p + k1, col_start+2] += 1
        V[1*p + k1, col_start+2] -= 1
        W[k1*n + 0, col_start+2] += 1   # C[0,k1]
        W[k1*n + 1, col_start+2] += 1   # C[1,k1]
        
        # M4 = A11 * (B1,k0 - B0,k0)
        # C00 += M4, C10 += M4
        U[3, col_start+3] += 1
        V[1*p + k0, col_start+3] += 1
        V[0*p + k0, col_start+3] -= 1
        W[k0*n + 0, col_start+3] += 1   # C[0,k0]
        W[k0*n + 1, col_start+3] += 1   # C[1,k0]
        
        # M5 = (A00 + A01) * B1,k1
        # C00 -= M5, C01 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[1*p + k1, col_start+4] += 1
        W[k0*n + 0, col_start+4] -= 1   # C[0,k0]
        W[k1*n + 0, col_start+4] += 1   # C[0,k1]
        
        # M6 = (A10 - A00) * (B0,k0 + B0,k1)
        # C11 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[0*p + k0, col_start+5] += 1
        V[0*p + k1, col_start+5] += 1
        W[k1*n + 1, col_start+5] += 1   # C[1,k1]
        
        # M7 = (A01 - A11) * (B1,k0 + B1,k1)
        # C00 += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[1*p + k0, col_start+6] += 1
        V[1*p + k1, col_start+6] += 1
        W[k0*n + 0, col_start+6] += 1   # C[0,k0]
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    # Schoolbook for column 6
    col_start = 21
    k0 = 6
    
    # C[0,6] = A00*B0,6 + A01*B1,6
    U[0, col_start+0] += 1
    V[0*p + 6, col_start+0] += 1
    W[6*n + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[1*p + 6, col_start+1] += 1
    W[6*n + 0, col_start+1] += 1
    
    # C[1,6] = A10*B0,6 + A11*B1,6
    U[2, col_start+2] += 1
    V[0*p + 6, col_start+2] += 1
    W[6*n + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[1*p + 6, col_start+3] += 1
    W[6*n + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
