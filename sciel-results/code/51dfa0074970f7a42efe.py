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
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)

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

    # Verify correctness with a random test
    try:
        import random
        random.seed(42)
        A_test = np.array([[random.randint(-5, 5) for _ in range(4)] for _ in range(4)], dtype=int)
        B_test = np.array([[random.randint(-5, 5) for _ in range(4)] for _ in range(4)], dtype=int)

        # Compute A*B using schoolbook
        AB = A_test @ B_test

        # Compute using decomposition
        AB_decomp = np.zeros((4, 4), dtype=float)
        for r in range(rank):
            u_val = np.sum(U[:, r] * A_test.flatten())
            v_val = np.sum(V[:, r] * B_test.flatten())
            scalar = u_val * v_val
            AB_decomp += W[:, r].reshape(4, 4) * scalar

        # Check if they match
        if not np.allclose(AB, AB_decomp, atol=1e-9):
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

    # Try to reduce rank by finding linear dependencies
    # The tensor identity is: for all i,j,k, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,k} (for matmul it's A[i,j]*B[j,k])
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta... no.
    # The tensor is T_{ijk} such that C_{ik} = sum_j A_{ij} T_{ijk} B_{jk}... 
    
    # Let me think differently. The decomposition satisfies:
    # For all A, B: A*B = sum_r (U_r^T * A) * (V_r^T * B) * W_r
    # where U_r is a column of U, etc.
    
    # Two rank-1 terms r1 and r2 are redundant if there exists alpha such that
    # U_r1 + alpha*U_r2 = 0, V_r1 + alpha*V_r2 = 0, W_r1 + alpha*W_r2 = 0
    # But that's too strict. More generally, if the set of (U_r, V_r, W_r) triples has a linear dependency
    # in the tensor space, we can reduce rank.
    
    # A simpler approach: check if any column of W is in the span of others, 
    # and if so, whether we can adjust U and V to compensate.
    
    # For now, let's try a greedy approach: try removing each term and see if the remaining
    # can still represent the tensor (this is expensive, so skip for now).
    
    # Alternative: use the fact that 4x4x4 matmul rank is known to be 49 (from 2-level Strassen).
    # The best known upper bound is 49, and lower bound is 49 (actually, the lower bound is 49? 
    # Let me recall: for 4x4x4, the rank is between 49 and 49? No, the lower bound is higher.
    # Actually, the rank of 4x4x4 matrix multiplication is known to be exactly 49? 
    # I'm not sure. Let me just return the 49-rank decomposition.
    
    # Convert to integer if possible
    U_int = np.round(U).astype(int)
    V_int = np.round(V).astype(int)
    W_int = np.round(W).astype(int)
    
    # Verify integer version
    try:
        import random
        random.seed(123)
        A_test = np.array([[random.randint(-3, 3) for _ in range(4)] for _ in range(4)], dtype=int)
        B_test = np.array([[random.randint(-3, 3) for _ in range(4)] for _ in range(4)], dtype=int)
        AB = A_test @ B_test
        AB_decomp = np.zeros((4, 4), dtype=int)
        for r in range(rank):
            u_val = np.sum(U_int[:, r] * A_test.flatten())
            v_val = np.sum(V_int[:, r] * B_test.flatten())
            scalar = u_val * v_val
            AB_decomp += W_int[:, r].reshape(4, 4) * scalar
        if not np.array_equal(AB, AB_decomp):
            # Use float version
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
# EVOLVE-BLOCK-END
