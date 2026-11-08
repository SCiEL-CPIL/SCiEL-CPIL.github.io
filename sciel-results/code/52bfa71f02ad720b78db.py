"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Use 3x3 blocks of 2x2 Strassen.
    # 6 = 3 * 2, so we have 3x3 = 9 blocks of size 2x2.
    # Each 2x2 block multiplication uses rank 7 (Strassen).
    # Total rank = 9 * 7 = 63.
    #
    # The tensor decomposition: C[i,j] = sum_r W[j*n+i, r] * (U_r : A) * (V_r : B)
    # where U_r : A = sum_{a,b} U[a*m+b, r] * A[a,b]
    # and V_r : B = sum_{a,b} V[a*p+b, r] * B[a,b]
    #
    # For the block structure:
    # A is 6x6, partitioned into 3x3 blocks of size 2x2.
    # B is 6x6, partitioned into 3x3 blocks of size 2x2.
    # C is 6x6, partitioned into 3x3 blocks of size 2x2.
    #
    # Block (I,J) of C = sum_K Block(I,K) of A * Block(K,J) of B
    # For each triple (I,K,J), we compute the 2x2 product using Strassen (7 terms).
    #
    # Strassen 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B11-B01)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    block_size = 2
    blocks_per_dim = 3  # 6 / 2

    rank = (blocks_per_dim ** 3) * 7  # 9 * 7 = 63

    # Initialize factor matrices as numpy arrays for efficient construction
    U_arr = np.zeros((n * m, rank), dtype=int)
    V_arr = np.zeros((m * p, rank), dtype=int)
    W_arr = np.zeros((p * n, rank), dtype=int)

    # Strassen patterns: (U_coeffs, V_coeffs, W_coeffs)
    # Each is a dict mapping (local_row, local_col) -> coefficient
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B11-B01) -> C01 += 1, C11 += 1
        ({(0,0): 1}, {(0,1): -1, (1,1): 1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00) -> C00 += 1, C10 += 1
        ({(1,1): 1}, {(0,0): -1, (1,0): 1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01) -> C11 += 1
        ({(0,0): -1, (1,0): 1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11) -> C00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    col_idx = 0
    for I in range(blocks_per_dim):
        for K in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                if time.time() > deadline:
                    break
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # U: A block at (I, K), local indices (pi, pj)
                    # Global A row = I*block_size + pi, col = K*block_size + pj
                    # U index = row * m + col = (I*2+pi) * 6 + (K*2+pj)
                    for (pi, pj), val in u_coeffs.items():
                        r_global = I * block_size + pi
                        c_global = K * block_size + pj
                        U_arr[r_global * m + c_global, col_idx] += val

                    # V: B block at (K, J), local indices (pi, pj)
                    # Global B row = K*block_size + pi, col = J*block_size + pj
                    # V index = row * p + col = (K*2+pi) * 6 + (J*2+pj)
                    for (pi, pj), val in v_coeffs.items():
                        r_global = K * block_size + pi
                        c_global = J * block_size + pj
                        V_arr[r_global * p + c_global, col_idx] += val

                    # W: C block at (I, J), local indices (pi, pj)
                    # C[i,j] where i = I*2+pi, j = J*2+pj
                    # W index = j * n + i = (J*2+pj) * 6 + (I*2+pi)
                    for (pi, pj), val in w_coeffs.items():
                        i_global = I * block_size + pi
                        j_global = J * block_size + pj
                        W_arr[j_global * n + i_global, col_idx] += val

                    col_idx += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Verify correctness with a random test
    try:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, (n, m))
        B_test = np.random.randint(-5, 6, (m, p))
        C_expected = A_test @ B_test

        A_flat = A_test.flatten()  # row-major: A[i*m+k]
        B_flat = B_test.flatten()  # row-major: B[k*p+j]

        M_A = U_arr @ A_flat  # shape (rank,)
        M_B = V_arr @ B_flat  # shape (rank,)

        # C[i,j] = sum_r W[j*n+i, r] * M_A[r] * M_B[r]
        # W_arr has shape (p*n, rank), W_arr[j*n+i, r]
        W_reshaped = W_arr.reshape(p, n, rank)
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                C_computed[i, j] = np.dot(W_reshaped[j, i, :], M_A * M_B)

        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-6:
            # Fallback: use naive rank-216 decomposition
            rank_naive = n * m * p
            U_naive = np.zeros((n * m, rank_naive), dtype=int)
            V_naive = np.zeros((m * p, rank_naive), dtype=int)
            W_naive = np.zeros((p * n, rank_naive), dtype=int)
            col = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_naive[i * m + k, col] = 1
                        V_naive[k * p + j, col] = 1
                        W_naive[j * n + i, col] = 1
                        col += 1
            return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    except Exception:
        pass

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
# EVOLVE-BLOCK-END
