"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    This implementation uses a known rank-26 decomposition.
    It combines a 7-term Strassen decomposition for the 2x2 sub-block 
    with 19 explicit rank-1 terms for the remaining border elements.
    
    The decomposition T = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] 
    must hold for the multiplication tensor.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    R = 26
    
    # Use int8 to avoid any potential overflow issues in intermediate steps, 
    # though int8 is sufficient for 0/1/-1 values. 
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)
    
    # Helper to get flattened indices
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
        
    # --- Strassen Part (7 terms) ---
    # These handle the 2x2 multiplication of the top-left block (indices 0,1).
    
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[idx_u(0, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # c00 += m1
    W[idx_w(1, 1), r] = 1  # c11 += m1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[idx_u(1, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1  # c10 += m2
    W[idx_w(1, 1), r] = -1 # c11 -= m2
    
    # m3 = a00(b01 - b11)
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1  # c01 += m3
    W[idx_w(1, 1), r] = 1  # c11 += m3
    
    # m4 = a11(b10 - b00)
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1
    V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1  # c00 += m4
    W[idx_w(0, 1), r] = 1  # c10 += m4
    
    # m5 = (a00 + a01)b11
    r = 4
    U[idx_u(0, 0), r] = 1
    U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1 # c00 -= m5
    W[idx_w(1, 0), r] = 1  # c01 += m5
    
    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[idx_u(1, 0), r] = 1
    U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1  # c11 += m6
    
    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[idx_u(0, 1), r] = 1
    U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # c00 += m7
    
    # --- Standard Part (19 terms) ---
    # Explicitly define the 19 border terms where at least one index is 2.
    # Format: (i, j, k)
    border_indices = [
        # i = 2 (9 terms)
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        
        # j = 2, i != 2 (6 terms)
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        
        # k = 2, i != 2, j != 2 (4 terms)
        (0, 0, 2), (0, 1, 2),
        (1, 0, 2), (1, 1, 2)
    ]
    
    assert len(border_indices) == 19
    
    col = 7
    for i, j, k in border_indices:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
        
    # col should be 26
    assert col == R, f"Rank mismatch: {col} vs {R}"

    # Verify the decomposition is exact before returning
    # Build the full tensor T from U, V, W
    T_decomp = np.zeros((n, m, p), dtype=np.int64)
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The multiplication tensor has T[i,j,k] = 1 if i==k else 0
    T_expected = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i == k:
                    T_expected[i, j, k] = 1
    
    if not np.array_equal(T_decomp, T_expected):
        # Fallback: return rank 27 (trivial) if something went wrong
        R27 = 27
        U27 = np.zeros((n * m, R27), dtype=np.int8)
        V27 = np.zeros((m * p, R27), dtype=np.int8)
        W27 = np.zeros((p * n, R27), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i == k:
                        U27[i*m+j, col] = 1
                        V27[j*p+k, col] = 1
                        W27[k*n+i, col] = 1
                        col += 1
        assert col == R27
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
