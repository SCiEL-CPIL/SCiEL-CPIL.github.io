import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Strategy: Split j into {0,1} and {2}.
    # Part 1: j in {0,1}. Dimensions 2x2x5. Use Schoolbook (Rank 20).
    # Part 2: j=2. Dimensions 2x1x5. Use Rank 1 (Outer product).
    # Total Rank = 20 + 1 = 21.
    
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 2: j=2 (Rank 1) ---
    r = 0
    # U indices: i*m + j = i*3 + 2 for i in {0,1}
    # V indices: j*p + k = 2*5 + k = 10+k for k in {0..4}
    # W indices: k*n + i = k*2 + i for k in {0..4}, i in {0,1}
    
    for i in range(n):
        U[i * m + 2, r] = 1
    for k in range(p):
        V[2 * p + k, r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1
            
    # --- Part 1: j in {0,1} (Rank 20, Schoolbook) ---
    # For each i in {0,1}, j in {0,1}, k in {0..4}
    # This is a 2x2x5 multiplication.
    # Schoolbook rank is 2*2*5 = 20.
    
    r_offset = 1
    idx = 0
    for i in range(n):
        for j in range(2):  # j is 0, 1
            for k in range(p):
                r = r_offset + idx
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                idx += 1
                
    # Verify
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1
                
    if not np.allclose(T_reconstructed, T_expected, atol=1e-10):
        # Fallback to naive rank 30 if something went wrong
        rank_naive = n * m * p
        U_naive = np.zeros((n * m, rank_naive))
        V_naive = np.zeros((m * p, rank_naive))
        W_naive = np.zeros((p * n, rank_naive))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_naive[ii * m + jj, col] = 1
                    V_naive[jj * p + kk, col] = 1
                    W_naive[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
