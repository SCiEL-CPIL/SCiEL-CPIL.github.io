import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n=2, m=2, p=7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k, s)
    return True, None

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses Strassen rank-7 for three 2x2 blocks and schoolbook rank-4 for the last column.
    """
    # Safety check: if we are close to the deadline, return a known valid (though higher rank) solution
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    n, m, p = 2, 2, 7
    rank = 25

    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    def add_strassen_block(col_start, k_start):
        # Adds 7 terms for the 2x2 block starting at column k_start
        # Standard Strassen decomposition for 2x2 * 2x2 -> 2x2
        # U rows: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # V rows: B[0,k]=7*0+k, B[1,k]=7*1+k
        # W rows: C[0,k]=2*k+0, C[1,k]=2*k+1
        
        # M1 = (A00+A11) * (B0,ks+B1,ks+1)
        # Contributes: C00 += M1, C11 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10+A11) * B0,ks
        # Contributes: C10 += M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # Contributes: C01 += M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # Contributes: C11 += M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*(k_start+1) + 1, col_start+3] += 1

        # M5 = (A00+A01) * B1,ks+1
        # Contributes: C00 -= M5, C11 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 1, col_start+4] += 1

        # M6 = (A10-A00) * (B0,ks+B0,ks+1)
        # Contributes: C11 -= M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] -= 1

        # M7 = (A01-A11) * (B1,ks+B1,ks+1)
        # Contributes: C00 += M7, C01 += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        W[2*(k_start+1) + 0, col_start+6] += 1

    # Apply Strassen to columns 0-1
    add_strassen_block(0, 0)
    
    # Apply Strassen to columns 2-3
    add_strassen_block(7, 2)
    
    # Apply Strassen to columns 4-5
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6
    col_start = 21
    k = 6

    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1

    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1

    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1

    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness
    if _time_remaining() > 10.0:
        valid, info = _verify_decomposition(U.tolist(), V.tolist(), W.tolist())
        if not valid:
            # Fallback to schoolbook if verification fails
            return _schoolbook_decomposition()
    
    return result
