"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks of 2 columns each
    rank = 7 * num_blocks  # 56

    # Initialize U, V, W as lists of lists of integers/floats
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen factors for 2x2x2 block multiplication.
    # We decompose C = A * B where A is 2x2, B is 2x2, C is 2x2.
    # The 7 products are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Recovery:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Define entries for U, V, W for a single block.
    # U indices: i*m + j (0..3)
    # V indices: j*p + k (0..31), but we use local k (0..1) and add block offset
    # W indices: k*n + i (0..31), but we use local k (0..1) and add block offset
    
    # Format: (list_of_u_entries, list_of_v_entries, list_of_w_entries)
    # Entry: (local_index, coeff)
    # For U: local_index is i*m + j
    # For V: local_index is j*2 + k_local (relative to block start in k) -> Actually V row is j*p + k_global.
    #        k_global = block_start + k_local.
    #        So V_row = j*p + block_start + k_local.
    #        We can store (j, k_local, coeff) and compute row later, or precompute base row for block 0.
    #        Let's store (j, k_local, coeff) to be safe and clear.
    # For W: local_index is k_local*n + i.
    #        W_row = k_global*n + i = (block_start + k_local)*n + i.
    #        Let's store (k_local, i, coeff).

    factors = [
        # s=0: M1
        {
            'u': [(0, 1), (3, 1)],  # A00, A11
            'v': [(0, 0, 1), (1, 1, 1)],  # B00, B11
            'w': [(0, 0, 1), (1, 1, 1)]   # C00, C11 (k_local, i, coeff)
        },
        # s=1: M2
        {
            'u': [(2, 1), (3, 1)],  # A10, A11
            'v': [(0, 0, 1)],       # B00
            'w': [(0, 1, 1), (1, 1, -1)]  # C10, C11
        },
        # s=2: M3
        {
            'u': [(0, 1)],          # A00
            'v': [(0, 1, 1), (1, 1, -1)],  # B01, B11
            'w': [(1, 0, 1), (1, 1, 1)]    # C01, C11
        },
        # s=3: M4
        {
            'u': [(3, 1)],          # A11
            'v': [(1, 0, 1), (0, 0, -1)],  # B10, B00
            'w': [(0, 0, 1), (0, 1, 1)]    # C00, C10
        },
        # s=4: M5
        {
            'u': [(0, 1), (1, 1)],  # A00, A01
            'v': [(1, 1, 1)],       # B11
            'w': [(0, 0, -1), (1, 0, 1)]  # C00, C01
        },
        # s=5: M6
        {
            'u': [(2, 1), (0, -1)], # A10, A00
            'v': [(0, 0, 1), (0, 1, 1)],  # B00, B01
            'w': [(1, 1, 1)]            # C11
        },
        # s=6: M7
        {
            'u': [(1, 1), (3, -1)],  # A01, A11
            'v': [(1, 0, 1), (1, 1, 1)],  # B10, B11
            'w': [(0, 0, 1)]            # C00
        }
    ]

    for t in range(num_blocks):
        block_start_k = 2 * t
        for s in range(7):
            col = t * 7 + s
            f = factors[s]
            
            # Fill U
            for u_idx, coeff in f['u']:
                U[u_idx][col] = coeff
                
            # Fill V
            # V row index = j * p + k_global
            # k_global = block_start_k + k_local
            for j, k_local, coeff in f['v']:
                v_row = j * p + block_start_k + k_local
                V[v_row][col] = coeff
                
            # Fill W
            # W row index = k_global * n + i
            # k_global = block_start_k + k_local
            for k_local, i, coeff in f['w']:
                w_row = (block_start_k + k_local) * n + i
                W[w_row][col] = coeff

    # Verify correctness
    if time.time() > deadline:
        # If we ran out of time, return what we have (should be complete)
        return {"U": U, "V": V, "W": W}

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute the tensor S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    S = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    # Check required entries
    # Iterate over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = S[a_idx, b_idx, c_idx]
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2 * m + j2][col] = 1
                                V_fb[j2 * p + k2][col] = 1
                                W_fb[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
