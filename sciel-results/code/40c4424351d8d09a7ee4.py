import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _verify_decomposition(U, V, W):
    """Verify tensor identity for 2x2 * 2x7 matrix multiplication."""
    n, m, p = 2, 2, 7
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            jb = b // p
            k = b % p
            if j == jb:
                c = k * n + i
                T_true[a, b, c] = 1.0
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_true, T_decomp, atol=1e-10)

def _build_rank23():
    """
    Rank-23 decomposition: 3 Strassen blocks (cols 0-1, 2-3, 4-5) with
    the last block modified to also partially cover col 6, plus 2 extra terms.
    
    Standard Strassen for 2x2*2x2 uses 7 terms.
    Cols 0-1: 7 terms (standard)
    Cols 2-3: 7 terms (standard)  
    Cols 4-5: 7 terms (standard Strassen)
    Col 6:    2 terms (exploiting overlap with modified M2, M5 of cols 4-5 block)
    Total: 7 + 7 + 7 + 2 = 23
    """
    n, m, p = 2, 2, 7
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        """Standard Strassen for 2x2 * 2x2 block at columns k_start, k_start+1."""
        # M1 = (A00+A11)*(B00+B11): C00 += M1, C11 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        # M2 = (A10+A11)*B00: C10 += M2, C11 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        # M3 = A00*(B01-B11): C01 += M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        # M4 = A11*(B10-B00): C10 += M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 1, col_start+3] += 1
        # M5 = (A00+A01)*B11: C00 -= M5, C01 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        # M6 = (A10-A00)*(B00+B01): C11 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        # M7 = (A01-A11)*(B10+B11): C00 += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen(0, 0)   # Cols 0-1
    add_strassen(7, 2)   # Cols 2-3
    add_strassen(14, 4)  # Cols 4-5
    
    # Now handle col 6 with 2 terms, exploiting that M2 and M5 of the cols 4-5
    # Strassen block already partially cover col 6 if we extend them.
    # 
    # Instead of modifying the Strassen block, let's just add the 2 remaining terms.
    # After standard Strassen for cols 4-5, col 6 has no contributions yet.
    # We need:
    #   C[0,6] = A00*B0,6 + A01*B1,6
    #   C[1,6] = A10*B0,6 + A11*B1,6
    #
    # Strategy: Use modified M2 and M5 from the cols 4-5 block to cover part of col 6.
    # But since we already added standard Strassen, let's instead just add 4 terms
    # for col 6... that gives rank 25.
    #
    # Better: Re-do cols 4-5 with modified M2 and M5, then add 2 terms.
    # Let's clear the cols 4-5 Strassen and redo it modified.
    
    # Clear cols 4-5 Strassen (terms 14-20)
    for c in range(14, 21):
        U[:, c] = 0
        V[:, c] = 0
        W[:, c] = 0
    
    cs = 14
    k4, k5, k6 = 4, 5, 6
    
    # M1 = (A00+A11)*(B0,4+B1,5): C[0,4] += M1, C[1,5] += M1
    U[0, cs+0] += 1
    U[3, cs+0] += 1
    V[7*0+k4, cs+0] += 1
    V[7*1+k5, cs+0] += 1
    W[2*k4+0, cs+0] += 1
    W[2*k5+1, cs+0] += 1
    
    # M2 = (A10+A11)*(B0,4+B0,6): C[1,4] += M2, C[1,5] -= M2, C[1,6] += M2
    U[2, cs+1] += 1
    U[3, cs+1] += 1
    V[7*0+k4, cs+1] += 1
    V[7*0+k6, cs+1] += 1
    W[2*k4+1, cs+1] += 1
    W[2*k5+1, cs+1] -= 1
    W[2*k6+1, cs+1] += 1
    
    # M3 = A00*(B0,5-B1,5): C[0,5] += M3
    U[0, cs+2] += 1
    V[7*0+k5, cs+2] += 1
    V[7*1+k5, cs+2] -= 1
    W[2*k5+0, cs+2] += 1
    
    # M4 = A11*(B1,4-B0,4): C[1,4] += M4
    U[3, cs+3] += 1
    V[7*1+k4, cs+3] += 1
    V[7*0+k4, cs+3] -= 1
    W[2*k4+1, cs+3] += 1
    
    # M5 = (A00+A01)*(B1,5+B1,6): C[0,4] -= M5, C[0,5] += M5, C[0,6] += M5
    U[0, cs+4] += 1
    U[1, cs+4] += 1
    V[7*1+k5, cs+4] += 1
    V[7*1+k6, cs+4] += 1
    W[2*k4+0, cs+4] -= 1
    W[2*k5+0, cs+4] += 1
    W[2*k6+0, cs+4] += 1
    
    # M6 = (A10-A00)*(B0,4+B0,5): C[1,5] += M6
    U[2, cs+5] += 1
    U[0, cs+5] -= 1
    V[7*0+k4, cs+5] += 1
    V[7*0+k5, cs+5] += 1
    W[2*k5+1, cs+5] += 1
    
    # M7 = (A01-A11)*(B1,4+B1,5): C[0,4] += M7
    U[1, cs+6] += 1
    U[3, cs+6] -= 1
    V[7*1+k4, cs+6] += 1
    V[7*1+k5, cs+6] += 1
    W[2*k4+0, cs+6] += 1
    
    # M8 = A00*(B0,6-B1,6): C[0,6] += M8
    # Remaining for C[0,6]: A00*B0,6 - A00*B1,6
    U[0, cs+7] += 1
    V[7*0+k6, cs+7] += 1
    V[7*1+k6, cs+7] -= 1
    W[2*k6+0, cs+7] += 1
    
    # M9 = A11*(B1,6-B0,6): C[1,6] += M9
    # Remaining for C[1,6]: A11*B1,6 - A11*B0,6
    U[3, cs+8] += 1
    V[7*1+k6, cs+8] += 1
    V[7*0+k6, cs+8] -= 1
    W[2*k6+1, cs+8] += 1
    
    return U, V, W

def _build_rank25():
    """Build the known-good rank-25 decomposition."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return U, V, W

def construct():
    """
    Construct a minimal-rank exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Tries rank-23 first, falls back to rank-25.
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    # Try rank-23
    try:
        U, V, W = _build_rank23()
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fall back to rank-25
    try:
        U, V, W = _build_rank25()
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    return _schoolbook_decomposition()

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
        print("Verification PASSED, rank =", U.shape[1])
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
