"""
Solution for 4x4 matrix multiplication with rank 49.

Uses a 2-level Strassen decomposition (Kronecker product of two 2x2 Strassen decompositions).
The rank is 7 * 7 = 49.

Strategy: Clean structural implementation with fast vectorized verification.
"""
import time
import numpy as np

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

    # Fast verification using tensor contraction
    # The decomposition says: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # We check this against the true multiplication tensor T[i,j,k] = 1 if i==k else 0... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (the coefficient of A[i,j]*B[j,k] in C[i,k] is 1)
    
    # Verify by checking a few random integer matrices
    try:
        if time.time() - start_time > deadline - 2:
            raise TimeoutError("skip verification")
        
        rng = np.random.RandomState(42)
        A_test = rng.randint(-3, 3, size=(4, 4)).astype(int)
        B_test = rng.randint(-3, 3, size=(4, 4)).astype(int)
        
        # True product
        AB_true = A_test @ B_test
        
        # Decomposed product
        # u_vals[r] = sum_{i,j} U[i*4+j, r] * A[i,j]
        # v_vals[r] = sum_{j,k} V[j*4+k, r] * B[j,k]
        # AB_decomp[i,k] = sum_r W[k*4+i, r] * u_vals[r] * v_vals[r]
        
        A_flat = A_test.flatten()  # (16,)
        B_flat = B_test.flatten()  # (16,)
        
        # U is (16, 49), so u_vals = U.T @ A_flat
        u_vals = U.T @ A_flat  # (49,)
        v_vals = V.T @ B_flat  # (49,)
        
        # AB_decomp[k*4+i] = sum_r W[k*4+i, r] * u_vals[r] * v_vals[r]
        # = W @ (u_vals * v_vals)
        # W is (16, 49), u_vals*v_vals is (49,)
        AB_decomp_flat = W @ (u_vals * v_vals)  # (16,)
        AB_decomp = AB_decomp_flat.reshape((4, 4)).T  # transpose to get [i,k]
        
        if not np.array_equal(AB_true, AB_decomp):
            raise ValueError("Verification failed")
    except Exception:
        # Fallback to schoolbook rank 64
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
