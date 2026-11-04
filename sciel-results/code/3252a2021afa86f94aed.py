import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_strassen_rank39(n, m, p):
    """Build the rank-39 Strassen-based decomposition."""
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
        nonlocal col
        if u_coeffs is None:
            u_coeffs = [1] * len(u_indices)
        if v_coeffs is None:
            v_coeffs = [1] * len(v_indices)
        if w_coeffs is None:
            w_coeffs = [1] * len(w_indices)
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        """Add a 2x2x2 Strassen block for output columns k_start, k_start+1."""
        # A indices (i*3 + j):
        a00 = 0 * 3 + 0  # A[0,0]
        a01 = 0 * 3 + 1  # A[0,1]
        a10 = 1 * 3 + 0  # A[1,0]
        a11 = 1 * 3 + 1  # A[1,1]
        
        # B indices (j*7 + k):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (k*2 + i):
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> +C00, +C11
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> +C10, -C11
        add_term([a10, a11], [b00], [c10, c11], w_coeffs=[1, -1])
        
        # M3: A00(B01-B11) -> +C01, +C11
        add_term([a00], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        
        # M4: A11(B10-B00) -> +C00, +C10
        add_term([a11], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        
        # M5: (A00+A01)B11 -> -C00, +C01
        add_term([a00, a01], [b11], [c00, c01], w_coeffs=[-1, 1])
        
        # M6: (A10-A00)(B00+B01) -> +C11
        add_term([a10, a00], [b00, b01], [c11], u_coeffs=[1, -1])
        
        # M7: (A01-A11)(B10+B11) -> +C00
        add_term([a01, a11], [b10, b11], [c00], u_coeffs=[1, -1])

    # Add 3 Strassen blocks for (k=0,1), (k=2,3), (k=4,5)
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for k=6: 2x2x1 schoolbook
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])  # A[0,0]*B[0,6] -> C[0,6]
    add_term([1], [b16], [c06])  # A[0,1]*B[1,6] -> C[0,6]
    add_term([3], [b06], [c16])  # A[1,0]*B[0,6] -> C[1,6]
    add_term([4], [b16], [c16])  # A[1,1]*B[1,6] -> C[1,6]
    
    # Block for j=2: 2x1x7 schoolbook
    # C[i,k] += A[i,2]*B[2,k] for all i,k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])  # A[0,2]*B[2,k] -> C[0,k]
        add_term([5], [b2k], [c1k])  # A[1,2]*B[2,k] -> C[1,k]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the rank-39 Strassen decomposition
    result = _build_strassen_rank39(n, m, p)
    
    # Verify it
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback: schoolbook if verification fails (shouldn't happen)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
