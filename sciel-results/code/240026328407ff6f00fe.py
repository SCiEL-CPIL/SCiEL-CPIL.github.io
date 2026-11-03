"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    
    This implementation uses a rank-26 decomposition constructed by combining
    the 7-multiplication Strassen algorithm for the 2x2 sub-block with 19 standard
    multiplications for the remaining elements.
    
    The tensor T is defined by T[i, j, k] = 1 for all i, j, k in {0, 1, 2}.
    The decomposition T = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] must hold.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Rank 26 decomposition
    R = 26
    
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
    # These handle all (i, j, k) where at least one index is 2.
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if all indices are in {0, 1}
                if i < 2 and j < 2 and k < 2:
                    continue
                
                # Add a rank-1 term: e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
                
    # col should be 26
    assert col == R, f"Rank mismatch: {col} vs {R}"

    # Quick verification to ensure no coding errors in indices
    # This is fast (27 * 26 operations)
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    # Check all 27 positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_f[i*m+j, :], V_f[j*p+k, :] * W_f[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): sum={s}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
