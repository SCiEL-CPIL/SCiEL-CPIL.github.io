import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE

    n, m, p = 3, 4, 8

    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: schoolbook rank 96
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Part 1: i=0 (1x4x8 schoolbook)
    # A[0, :] * B[:, :] -> C[0, :]
    for j in range(m):
        for k in range(p):
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1

    # Part 2: i=1,2 (2x4x8 Strassen)
    # A[1:3, :] * B[:, :] -> C[1:3, :]
    # Split A[1:3, :] into two 1x2 blocks? No, 2x4.
    # Split B[:, :] into four 4x2 blocks? No, 4x8.
    # We treat A[1:3, :] as 2x2 blocks of columns.
    # We treat B[:, :] as 2x2 blocks of rows and columns.
    
    # Strassen matrices for 2x2x2
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    # A_sub is 2x4. Rows 1,2 of A.
    # B_sub is 4x8. All rows of B.
    # C_sub is 2x8. Rows 1,2 of C.
    
    # j_block: 0, 1 (columns of A_sub, size 2 each)
    # k_block: 0, 1, 2, 3 (columns of B_sub, size 2 each)
    # For each j_block, k_block, we compute a 2x2 block of C_sub.
    # The 2x2 block of A_sub is A[1:3, 2*j_block:2*j_block+2]
    # The 2x2 block of B_sub is B[2*j_block:2*j_block+2, 2*k_block:2*k_block+2]
    # Wait, B is 4x8. The rows of B correspond to the columns of A.
    # So for j_block=0, we use B rows 0,1.
    # For j_block=1, we use B rows 2,3.
    # The columns of B are 0..7.
    # For k_block=0, we use B cols 0,1.
    # For k_block=1, we use B cols 2,3.
    # etc.
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            # The 2x2 sub-matrix of A (rows 1,2; cols j_start, j_start+1)
            # The 2x2 sub-matrix of B (rows j_start, j_start+1; cols k_start, k_start+1)
            # Result 2x2 sub-matrix of C (rows 1,2; cols k_start, k_start+1)
            
            for t in range(7):
                # U part: A coefficients
                # S_A[t, :] corresponds to A elements [A00, A01, A10, A11]
                # A00 = A[1, j_start], A01 = A[1, j_start+1]
                # A10 = A[2, j_start], A11 = A[2, j_start+1]
                
                # V part: B coefficients
                # S_B[t, :] corresponds to B elements [B00, B01, B10, B11]
                # B00 = B[j_start, k_start], B01 = B[j_start, k_start+1]
                # B10 = B[j_start+1, k_start], B11 = B[j_start+1, k_start+1]
                
                # W part: C coefficients
                # S_C[t, :] corresponds to C elements [C00, C01, C10, C11]
                # C00 = C[1, k_start], C01 = C[1, k_start+1]
                # C10 = C[2, k_start], C11 = C[2, k_start+1]
                
                # Fill U
                for l in range(4):
                    coeff = S_A[t, l]
                    if coeff != 0:
                        # l=0: (0,0), l=1: (0,1), l=2: (1,0), l=3: (1,1)
                        loc_i = l // 2
                        loc_j = l % 2
                        g_i = 1 + loc_i
                        g_j = j_start + loc_j
                        U[g_i * m + g_j, col] += coeff

                # Fill V
                for l in range(4):
                    coeff = S_B[t, l]
                    if coeff != 0:
                        loc_j = l // 2
                        loc_k = l % 2
                        g_j = j_start + loc_j
                        g_k = k_start + loc_k
                        V[g_j * p + g_k, col] += coeff

                # Fill W
                for l in range(4):
                    coeff = S_C[t, l]
                    if coeff != 0:
                        loc_i = l // 2
                        loc_k = l % 2
                        g_i = 1 + loc_i
                        g_k = k_start + loc_k
                        W[g_k * n + g_i, col] += coeff

                col += 1

    assert col == rank, f"Rank mismatch: {col} != {rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
