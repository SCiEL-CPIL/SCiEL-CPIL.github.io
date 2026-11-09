"""Rewritten solution for matrix multiplication 4,4,4."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 4, 4, 4
    
    # Use the known optimal rank-49 decomposition (Strassen-based)
    rank = 49
    
    # Pre-computed Strassen 2x2 element patterns
    # U_elem[t][ei*2+ej], V_elem[t][ej*2+ek], W_elem[t][ei][ek]
    U_elem = np.array([
        [1, 0, 0, 1],   # t=0
        [0, 0, 1, 1],   # t=1
        [1, 0, 0, 0],   # t=2
        [0, 0, 0, 1],   # t=3
        [1, 1, 0, 0],   # t=4
        [-1, 0, 1, 0],  # t=5
        [0, 1, 0, -1],  # t=6
    ], dtype=np.int64)
    
    V_elem = np.array([
        [1, 0, 0, 1],   # t=0
        [1, 0, 0, 0],   # t=1
        [0, 1, 0, -1],  # t=2
        [-1, 0, 1, 0],  # t=3
        [0, 0, 0, 1],   # t=4
        [1, 1, 0, 0],   # t=5
        [0, 0, 1, 1],   # t=6
    ], dtype=np.int64)
    
    W_elem = np.array([
        [[1, 0], [0, 1]],   # t=0
        [[0, 0], [1, -1]],  # t=1
        [[0, 1], [0, 1]],   # t=2
        [[1, 0], [1, 0]],   # t=3
        [[-1, 1], [0, 0]],  # t=4
        [[0, 0], [0, 1]],   # t=5
        [[1, 0], [0, 0]],   # t=6
    ], dtype=np.int64)
    
    # Block-level Strassen patterns: list of (block_row, block_col, coefficient)
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0
        [(1,0,1), (1,1,1)],      # s=1
        [(0,0,1)],               # s=2
        [(1,1,1)],               # s=3
        [(0,0,1), (0,1,1)],      # s=4
        [(1,0,1), (0,0,-1)],     # s=5
        [(0,1,1), (1,1,-1)],     # s=6
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0
        [(0,0,1)],               # s=1
        [(0,1,1), (1,1,-1)],     # s=2
        [(1,0,1), (0,0,-1)],     # s=3
        [(1,1,1)],               # s=4
        [(0,0,1), (0,1,1)],      # s=5
        [(1,0,1), (1,1,1)],      # s=6
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0
        [(1,0,1), (1,1,-1)],     # s=1
        [(0,1,1), (1,1,1)],      # s=2
        [(0,0,1), (1,0,1)],      # s=3
        [(0,0,-1), (0,1,1)],     # s=4
        [(1,1,1)],               # s=5
        [(0,0,1)],               # s=6
    ]
    
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Build U column
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2 * bi + ei) * 4 + (2 * bj + ej)
                        U[row_a, col] += c_a * U_elem[t, ei * 2 + ej]
            
            # Build V column
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2 * bj + ej) * 4 + (2 * bk + ek)
                        V[row_b, col] += c_b * V_elem[t, ej * 2 + ek]
            
            # Build W column: W[k*4+i, r]
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        row_c = (2 * bk + ek) * 4 + (2 * bi + ei)
                        W[row_c, col] += c_c * W_elem[t, ei, ek]
    
    # Quick verification with a few random tests
    rng = np.random.default_rng(42)
    all_correct = True
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        A_test = rng.integers(-10, 10, size=(4, 4), dtype=np.int64)
        B_test = rng.integers(-10, 10, size=(4, 4), dtype=np.int64)
        AB_expected = A_test @ B_test
        
        # Compute using decomposition
        u_vals = U @ A_test.flatten()
        v_vals = V @ B_test.flatten()
        scalars = u_vals * v_vals
        
        # W is indexed as W[k*4+i, r], so we need to be careful
        # AB_decomp[i,k] = sum_r W[k*4+i, r] * scalars[r]
        W_reshaped = W.reshape(4, 4, rank)  # W_reshaped[k, i, r]
        AB_decomp = np.einsum('kir,r->ik', W_reshaped, scalars)
        
        if not np.array_equal(AB_expected, AB_decomp):
            all_correct = False
            break
    
    if not all_correct:
        # Fallback to schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=np.int64)
        V_fb = np.zeros((16, rank_fb), dtype=np.int64)
        W_fb = np.zeros((16, rank_fb), dtype=np.int64)
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
# EVOLVE-BLOCK-END
