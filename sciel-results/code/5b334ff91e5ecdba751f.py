import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

def _time_remaining():
    return _DEADLINE - time.time()

def _build_target_tensor():
    """Build the (2,2,7) matrix multiplication tensor T[i,j,k]."""
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T=None):
    """Verify U, V, W decompose the target tensor exactly."""
    if T is None:
        T = _build_target_tensor()
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-10)

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
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

def _construct_2x2_2x7_decomposition():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7.
    Uses Strassen (rank 7) for column pairs (0,1), (2,3), (4,5)
    and schoolbook (rank 4) for column 6.
    """
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def _add_strassen_block(col_start, base_idx):
        """Add a Strassen decomposition for columns col_start and col_start+1."""
        b00 = col_start
        b01 = col_start + 1
        b10 = 7 + col_start
        b11 = 7 + col_start + 1
        c00 = col_start * 2
        c01 = (col_start + 1) * 2
        c10 = col_start * 2 + 1
        c11 = (col_start + 1) * 2 + 1
        
        # M1 = (a00+a11)*(b00+b11)
        U[0, base_idx+0] = 1; U[3, base_idx+0] = 1
        V[b00, base_idx+0] = 1; V[b11, base_idx+0] = 1
        W[c00, base_idx+0] = 1; W[c11, base_idx+0] = 1
        
        # M2 = (a10+a11)*b00
        U[2, base_idx+1] = 1; U[3, base_idx+1] = 1
        V[b00, base_idx+1] = 1
        W[c10, base_idx+1] = 1; W[c11, base_idx+1] = -1
        
        # M3 = a00*(b01-b11)
        U[0, base_idx+2] = 1
        V[b01, base_idx+2] = 1; V[b11, base_idx+2] = -1
        W[c01, base_idx+2] = 1; W[c11, base_idx+2] = 1
        
        # M4 = a11*(b10-b00)
        U[3, base_idx+3] = 1
        V[b10, base_idx+3] = 1; V[b00, base_idx+3] = -1
        W[c00, base_idx+3] = 1; W[c10, base_idx+3] = 1
        
        # M5 = (a00+a01)*b11
        U[0, base_idx+4] = 1; U[1, base_idx+4] = 1
        V[b11, base_idx+4] = 1
        W[c00, base_idx+4] = -1; W[c01, base_idx+4] = 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[2, base_idx+5] = 1; U[0, base_idx+5] = -1
        V[b00, base_idx+5] = 1; V[b01, base_idx+5] = 1
        W[c11, base_idx+5] = 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[1, base_idx+6] = 1; U[3, base_idx+6] = -1
        V[b10, base_idx+6] = 1; V[b11, base_idx+6] = 1
        W[c00, base_idx+6] = 1

    # Add Strassen blocks for column pairs (0,1), (2,3), (4,5)
    _add_strassen_block(0, 0)
    _add_strassen_block(2, 7)
    _add_strassen_block(4, 14)
    
    # Schoolbook for column 6
    b0_6 = 6
    b1_6 = 13
    c0_6 = 12
    c1_6 = 13
    
    # Term 21: a00 * b0_6 -> c0_6
    U[0, 21] = 1
    V[b0_6, 21] = 1
    W[c0_6, 21] = 1
    
    # Term 22: a01 * b1_6 -> c0_6
    U[1, 22] = 1
    V[b1_6, 22] = 1
    W[c0_6, 22] = 1
    
    # Term 23: a10 * b0_6 -> c1_6
    U[2, 23] = 1
    V[b0_6, 23] = 1
    W[c1_6, 23] = 1
    
    # Term 24: a11 * b1_6 -> c1_6
    U[3, 24] = 1
    V[b1_6, 24] = 1
    W[c1_6, 24] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Construct a minimal-rank exact tensor decomposition for 2x2 * 2x7.
    Returns rank 25 decomposition.
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    result = _construct_2x2_2x7_decomposition()
    
    # Quick verification to ensure correctness
    T = _build_target_tensor()
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    if _verify_decomposition(U, V, W, T):
        return result
    else:
        return _schoolbook_decomposition()

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
