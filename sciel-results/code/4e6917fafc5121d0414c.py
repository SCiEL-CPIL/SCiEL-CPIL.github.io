import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s kill

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
    
    Indexing convention:
    - A[i,j] -> U[i*2+j] (row-major: A00=0, A01=1, A10=2, A11=3)
    - B[j,k] -> V[j*7+k] (row-major: B0,0=0 ... B1,6=13)
    - C[i,k] -> W[k*2+i] (trace convention as specified)
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))  # (4, 25)
    V = np.zeros((m*p, rank))  # (14, 25)
    W = np.zeros((p*n, rank))  # (14, 25)
    
    def add_strassen(k0, k1, col_offset):
        """
        Add Strassen rank-7 terms for 2x2 block at columns k0, k1.
        
        Standard Strassen for C = A * B (all 2x2):
        M1 = (A00+A11) * (B00+B11)
        M2 = (A10+A11) * B00
        M3 = A00 * (B01-B11)
        M4 = A11 * (B10-B00)
        M5 = (A00+A01) * B11
        M6 = (A10-A00) * (B00+B01)
        M7 = (A01-A11) * (B10+B11)
        
        C00 = M1 - M4 + M5 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # M1 = (A00+A11) * (B0,k0 + B1,k1)
        U[0, col_offset+0] += 1   # A00
        U[3, col_offset+0] += 1   # A11
        V[7*0 + k0, col_offset+0] += 1   # B0,k0
        V[7*1 + k1, col_offset+0] += 1   # B1,k1
        # C00 += M1, C11 += M1
        W[2*k0 + 0, col_offset+0] += 1   # C[0,k0]
        W[2*k1 + 1, col_offset+0] += 1   # C[1,k1]
        
        # M2 = (A10+A11) * B0,k0
        U[2, col_offset+1] += 1   # A10
        U[3, col_offset+1] += 1   # A11
        V[7*0 + k0, col_offset+1] += 1   # B0,k0
        # C10 += M2, C11 -= M2
        W[2*k0 + 1, col_offset+1] += 1   # C[1,k0]
        W[2*k1 + 1, col_offset+1] -= 1   # C[1,k1]
        
        # M3 = A00 * (B0,k1 - B1,k1)
        U[0, col_offset+2] += 1   # A00
        V[7*0 + k1, col_offset+2] += 1   # B0,k1
        V[7*1 + k1, col_offset+2] -= 1   # B1,k1
        # C01 += M3, C11 += M3
        W[2*k1 + 0, col_offset+2] += 1   # C[0,k1]
        W[2*k1 + 1, col_offset+2] += 1   # C[1,k1]
        
        # M4 = A11 * (B1,k0 - B0,k0)
        U[3, col_offset+3] += 1   # A11
        V[7*1 + k0, col_offset+3] += 1   # B1,k0
        V[7*0 + k0, col_offset+3] -= 1   # B0,k0
        # C00 -= M4, C10 += M4
        W[2*k0 + 0, col_offset+3] -= 1   # C[0,k0]
        W[2*k0 + 1, col_offset+3] += 1   # C[1,k0]
        
        # M5 = (A00+A01) * B1,k1
        U[0, col_offset+4] += 1   # A00
        U[1, col_offset+4] += 1   # A01
        V[7*1 + k1, col_offset+4] += 1   # B1,k1
        # C00 += M5, C01 += M5
        W[2*k0 + 0, col_offset+4] += 1   # C[0,k0]
        W[2*k1 + 0, col_offset+4] += 1   # C[0,k1]
        
        # M6 = (A10-A00) * (B0,k0 + B0,k1)
        U[2, col_offset+5] += 1   # A10
        U[0, col_offset+5] -= 1   # A00
        V[7*0 + k0, col_offset+5] += 1   # B0,k0
        V[7*0 + k1, col_offset+5] += 1   # B0,k1
        # C11 += M6
        W[2*k1 + 1, col_offset+5] += 1   # C[1,k1]
        
        # M7 = (A01-A11) * (B1,k0 + B1,k1)
        U[1, col_offset+6] += 1   # A01
        U[3, col_offset+6] -= 1   # A11
        V[7*1 + k0, col_offset+6] += 1   # B1,k0
        V[7*1 + k1, col_offset+6] += 1   # B1,k1
        # C00 -= M7
        W[2*k0 + 0, col_offset+6] -= 1   # C[0,k0]
        
    # Apply Strassen to column pairs (0,1), (2,3), (4,5)
    add_strassen(0, 1, 0)
    add_strassen(2, 3, 7)
    add_strassen(4, 5, 14)
    
    # Schoolbook for column 6 (4 terms)
    col_start = 21
    # C[0,6] = A00*B0,6 + A01*B1,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # C[1,6] = A10*B0,6 + A11*B1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p  # 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
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
    
    # Verify with random matrices
    np.random.seed(42)
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    # Reconstruct C from tensor decomposition
    # C[i,k] = sum_r (sum_{i',j} U[i',j,r] * A[i',j]) * (sum_{j',k'} V[j',k',r] * B[j',k']) * W[k*2+i, r]
    # Wait, that's not right. The tensor decomposition means:
    # The bilinear form for C[i,k] is: sum_r U_r(A) * V_r(B) * W_r[i,k]
    # where U_r(A) = sum_{i,j} U[i,j,r] * A[i,j]
    # and V_r(B) = sum_{j,k} V[j,k,r] * B[j,k]
    # and W_r[i,k] = W[k*2+i, r]
    
    C_reconstructed = np.zeros((2, 7))
    for r in range(U.shape[1]):
        u_val = np.dot(U[:, r], A.flatten())  # sum over A indices
        v_val = np.dot(V[:, r], B.flatten())  # sum over B indices
        for k in range(7):
            for i in range(2):
                C_reconstructed[i, k] += W[k*2 + i, r] * u_val * v_val
    
    if np.allclose(C_reconstructed, C_expected, atol=1e-10):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print("Reconstructed:")
        print(C_reconstructed)
        print("Expected:")
        print(C_expected)
        print("Max diff:", np.max(np.abs(C_reconstructed - C_expected)))

