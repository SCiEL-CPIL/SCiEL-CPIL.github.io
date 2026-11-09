"""Rewritten solution for matrix multiplication 4,4,4."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5 second safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Element-level Strassen data for 2x2 matrices
    # Indexing: [elem00, elem01, elem10, elem11]
    U_elem = [
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1],  # t=6: a01-a11
    ]
    V_elem = [
        [1, 0, 0, 1],   # t=0: b00+b11
        [1, 0, 0, 0],   # t=1: b00
        [0, 1, 0, -1],  # t=2: b01-b11
        [-1, 0, 1, 0],  # t=3: b10-b00
        [0, 0, 0, 1],   # t=4: b11
        [1, 1, 0, 0],   # t=5: b00+b01
        [0, 0, 1, 1],   # t=6: b10+b11
    ]
    W_elem = [
        [[1, 0], [0, 1]],   # t=0
        [[0, 0], [1, -1]],  # t=1
        [[0, 1], [0, 1]],   # t=2
        [[1, 0], [1, 0]],   # t=3
        [[-1, 1], [0, 0]],  # t=4
        [[0, 0], [0, 1]],   # t=5
        [[1, 0], [0, 0]],   # t=6
    ]

    # Block-level Strassen data: which 2x2 blocks are used and with what coefficient
    # Format: list of (block_row, block_col, coefficient)
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: A00+A11
        [(1,0,1), (1,1,1)],      # s=1: A10+A11
        [(0,0,1)],               # s=2: A00
        [(1,1,1)],               # s=3: A11
        [(0,0,1), (0,1,1)],      # s=4: A00+A01
        [(1,0,1), (0,0,-1)],     # s=5: A10-A00
        [(0,1,1), (1,1,-1)],     # s=6: A01-A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: B00+B11
        [(0,0,1)],               # s=1: B00
        [(0,1,1), (1,1,-1)],     # s=2: B01-B11
        [(1,0,1), (0,0,-1)],     # s=3: B10-B00
        [(1,1,1)],               # s=4: B11
        [(0,0,1), (0,1,1)],      # s=5: B00+B01
        [(1,0,1), (1,1,1)],      # s=6: B10+B11
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: C00+=M1, C11+=M1
        [(1,0,1), (1,1,-1)],     # s=1: C10+=M2, C11-=M2
        [(0,1,1), (1,1,1)],      # s=2: C01+=M3, C11+=M3
        [(0,0,1), (1,0,1)],      # s=3: C00+=M4, C10+=M4
        [(0,0,-1), (0,1,1)],     # s=4: C00-=M5, C01+=M5
        [(1,1,1)],               # s=5: C11+=M6
        [(0,0,1)],               # s=6: C00+=M7
    ]

    # Build U, V, W matrices
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            # Build U for this term
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2 * bi + ei) * 4 + (2 * bj + ej)
                        U[row_a, col] += c_a * U_elem[t][ei * 2 + ej]

            # Build V for this term
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2 * bj + ej) * 4 + (2 * bk + ek)
                        V[row_b, col] += c_b * V_elem[t][ej * 2 + ek]

            # Build W for this term
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        # W index: k*n + i where n=4
                        row_c = (2 * bk + ek) * 4 + (2 * bi + ei)
                        W[row_c, col] += c_c * W_elem[t][ei][ek]

    # Verify correctness with robust random tests
    try:
        if time.time() > deadline:
            # Out of time, return what we have
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        import random
        # Use multiple random seeds with varied value ranges for robust verification
        all_correct = True
        test_configs = [
            (42, 5), (123, 10), (999, 3), (777, 7), (555, 20),
            (111, 1), (222, 50), (333, 100), (444, 10), (888, 5)
        ]
        for seed_val, max_val in test_configs:
            if time.time() > deadline:
                break
            random.seed(seed_val)
            A_test = np.array([[random.randint(-max_val, max_val) for _ in range(4)] for _ in range(4)], dtype=int)
            B_test = np.array([[random.randint(-max_val, max_val) for _ in range(4)] for _ in range(4)], dtype=int)

            # Compute A*B using schoolbook
            AB = A_test @ B_test

            # Compute using decomposition with vectorized operations
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            # u_vals[r] = sum_{i,j} U[i*4+j, r] * A[i,j]
            # v_vals[r] = sum_{j,k} V[j*4+k, r] * B[j,k]
            u_vals = U @ A_flat  # shape (rank,)
            v_vals = V @ B_flat  # shape (rank,)
            scalars = u_vals * v_vals  # shape (rank,)
            
            # W has shape (16, rank), W[k*4+i, r] corresponds to C[i,k]
            # AB_decomp[i,k] = sum_r W[k*4+i, r] * scalars[r]
            # Reshape W from (16, rank) to (4, 4, rank) where index k*4+i -> [i, k, :]
            W_reshaped = W.reshape(4, 4, rank)  # W[k*4+i, r] -> W_reshaped[i, k, r]
            # AB_decomp[i,k] = sum_r W_reshaped[i, k, r] * scalars[r]
            AB_decomp = np.tensordot(W_reshaped, scalars, axes=([2], [0]))  # shape (4, 4)
            
            # AB_decomp is C^T due to W[k*4+i] convention
            if not np.array_equal(AB.T, AB_decomp):
                all_correct = False
                break

        if not all_correct:
            # If mismatch, fall back to schoolbook
            rank_fb = 64
            U_fb = np.zeros((16, rank_fb), dtype=int)
            V_fb = np.zeros((16, rank_fb), dtype=int)
            W_fb = np.zeros((16, rank_fb), dtype=int)
            col_fb = 0
            for i2 in range(4):
                for j2 in range(4):
                    for k2 in range(4):
                        U_fb[i2 * 4 + j2, col_fb] = 1
                        V_fb[j2 * 4 + k2, col_fb] = 1
                        W_fb[k2 * 4 + i2, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
