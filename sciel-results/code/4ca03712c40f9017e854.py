import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 6x6x6 tensor decomposition.
    
    Strategy:
    - Outer: 2x2 block Strassen (rank 7)
    - Inner: 3x3 schoolbook (rank 27)
    - Total rank: 189
    
    The output W follows the convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    dim = n  # 6

    block_size = 3
    rank_inner = block_size ** 3  # 27
    rank_outer = 7  # Strassen rank for 2x2
    total_rank = rank_inner * rank_outer  # 189

    # Allocate factor matrices
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]

    col = 0

    # Strassen steps for 2x2 matrix multiplication
    steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00 * (B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11 * (B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00) * (B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    for A_coeffs, B_coeffs, C_coeffs in steps:
        for u in range(block_size):
            for w in range(block_size):
                for v in range(block_size):
                    # Update U (left factor)
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca

                    # Update V (right factor)
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb

                    # Update W (output factor)
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc

                    col += 1

    # Verify the decomposition with a quick check
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        import random
        random.seed(42)
        valid = True
        for _ in range(500):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            val = 0.0
            for r in range(total_rank):
                val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            if abs(val - 1.0) > 1e-10:
                valid = False
                break

        if valid:
            return {"U": U, "V": V, "W": W}

    # Fall back to schoolbook if verification fails or time is tight
    rank_fallback = n * m * p  # 216
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
