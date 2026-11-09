import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition matrices
    # U, V, W are 4x7
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # Tensor product of Strassen
    U1 = np.kron(S_U, S_U) # 16x49
    V1 = np.kron(S_V, S_V) # 16x49 -> We need 24x49, but only first 16 rows (j in 0..3, k in 0..3) are non-zero?
    # Wait, V1 shape in 4x4x4 is (m*p, rank) = (4*4, 49) = (16, 49).
    # But the full V matrix is (m*p, total_rank) = (4*6, total_rank) = (24, total_rank).
    # For the 4x4x4 part, V only depends on B[0:4, 0:4].
    # So V1_full is 24x49, with rows 16-23 being zero.
    
    V1_full = np.zeros((m * p, 49))
    V1_full[:16, :] = V1
    
    W1 = np.kron(S_W, S_W) # 16x49 -> W shape is (p*n, rank) = (4*4, 49) = (16, 49).
    # Full W is 24x49. Rows 16-23 correspond to k in 4,5 which are zero in this part.
    W1_full = np.zeros((p * n, 49))
    W1_full[:16, :] = W1
    
    # Part 2: 4x4x2 decomposition (Rank 28)
    # Tiling approach
    # U2: 16x28
    # V2: 24x28 (non-zero in rows corresponding to j in 0..3, k in 4..5)
    # W2: 24x28 (non-zero in rows corresponding to k in 4..5, i in 0..3)
    
    U2 = np.zeros((n * m, 28))
    V2 = np.zeros((m * p, 28))
    W2 = np.zeros((p * n, 28))
    
    # The 4x4x2 part involves A (4x4) and B_right (4x2).
    # B_right corresponds to columns 4,5 of B.
    # We split A into four 2x2 blocks: A00, A01, A10, A11.
    # We split B_right into two 2x2 blocks: B_top (rows 0,1), B_bot (rows 2,3).
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    
    # Indices:
    # A block (bi, bj) corresponds to rows i in [2*bi, 2*bi+1] and cols j in [2*bj, 2*bj+1].
    # B_right block (dk) corresponds to rows k in [2*dk, 2*dk+1] of the 4x2 matrix.
    # In the full B matrix, these are rows 2*dk, 2*dk+1.
    # The columns of B_right are 4, 5.
    # So in V2, the non-zero rows are for j in 0..3 and k in 4..5?
    # No, V index is j * p + k.
    # j is the column of A (0..3). k is the row of B (0..5).
    # In the 4x4x2 part, B is 4x2.
    # The "k" index in the 4x4x2 sub-problem corresponds to the row of B_right.
    # Let's map carefully.
    
    # Sub-problem 1: A00 * B_top
    # A00: i in {0,1}, j in {0,1}.
    # B_top: rows k in {0,1} of B_right. In full B, these are rows 0,1.
    # Wait, B_right is the right part of B.
    # B is 4x6.
    # B_left is 4x4 (cols 0-3).
    # B_right is 4x2 (cols 4-5).
    # The multiplication is A (4x4) * B_right (4x2).
    # The input to V is B.
    # The terms in V2 should pick up B[j, 4] and B[j, 5].
    # So the "k" index in the 4x4x2 tensor corresponds to the column index of B_right (0 or 1).
    # But in the full tensor, the index is (j, k_full) where k_full is 0..5.
    # The multiplication C[i, k_full] = sum_j A[i, j] B[j, k_full].
    # For k_full in {4, 5}, we use the 4x4x2 decomposition.
    # So the "k" in the 4x4x2 decomposition maps to k_full - 4.
    
    # Let's re-index.
    # The 4x4x2 tensor has indices (i, j, k') where i in 0..3, j in 0..3, k' in 0..1.
    # k' = 0 corresponds to k_full = 4.
    # k' = 1 corresponds to k_full = 5.
    
    # Strassen for 2x2 * 2x2.
    # We need to compute A_block (2x2) * B_block (2x2).
    # B_block is formed from B_right.
    # B_right is 4x2.
    # B_top (rows 0,1 of B_right) is 2x2.
    # B_bot (rows 2,3 of B_right) is 2x2.
    
    # Term 1: A00 * B_top
    # A00: i in {0,1}, j in {0,1}.
    # B_top: j in {0,1} (cols of A), k' in {0,1} (rows of B_right).
    # Wait, B_top is rows 0,1 of B_right.
    # B_right[j, k'] = B_full[j, 4+k'].
    # So B_top[j, k'] = B_full[j, 4+k'] for j in {0,1}.
    
    # In the 4x4x2 decomposition:
    # U2 picks up A[i, j].
    # V2 picks up B[j, 4+k'].
    # W2 picks up C[i, 4+k'].
    
    # We use 4 Strassen instances.
    # Instance 1: A00 * B_top
    # A00: i in {0,1}, j in {0,1}.
    # B_top: j in {0,1}, k' in {0,1}.
    # Result C_top_left: i in {0,1}, k' in {0,1}.
    
    # Let's map Strassen indices.
    # Strassen U: 4x7. Rows correspond to (i_local, j_local) in 0..1.
    # Strassen V: 4x7. Rows correspond to (j_local, k_local) in 0..1.
    # Strassen W: 4x7. Rows correspond to (k_local, i_local) in 0..1.
    
    # For Instance 1 (A00 * B_top):
    # i_local = i, j_local = j. (i,j in 0,1)
    # j_local = j, k_local = k'. (j in 0,1, k' in 0,1)
    # k_local = k', i_local = i.
    
    # U2 row (i, j) gets S_U[i, j, c].
    # V2 row (j, 4+k') gets S_V[j, k', c].
    # W2 row (4+k', i) gets S_W[k', i, c].
    
    # We have 4 such blocks:
    # 1. A00 * B_top: i in 0,1; j in 0,1; k' in 0,1.
    # 2. A01 * B_bot: i in 0,1; j in 2,3; k' in 0,1.
    #    B_bot is rows 2,3 of B_right.
    #    B_bot[j, k'] = B_full[j, 4+k'] for j in 2,3.
    # 3. A10 * B_top: i in 2,3; j in 0,1; k' in 0,1.
    # 4. A11 * B_bot: i in 2,3; j in 2,3; k' in 0,1.
    
    # Total rank 4 * 7 = 28.
    
    for bi in range(2): # A row block
        for bj in range(2): # A col block
            # Determine which B block to use
            # A00 (bi=0, bj=0) -> B_top (j in 0,1)
            # A01 (bi=0, bj=1) -> B_bot (j in 2,3)
            # A10 (bi=1, bj=0) -> B_top (j in 0,1)
            # A11 (bi=1, bj=1) -> B_bot (j in 2,3)
            # Wait, the formula was:
            # C_top = A00 B_top + A01 B_bot
            # C_bot = A10 B_top + A11 B_bot
            
            # So:
            # If bj == 0, use B_top (j in 0,1).
            # If bj == 1, use B_bot (j in 2,3).
            
            if bj == 0:
                j_offset = 0
            else:
                j_offset = 2
                
            for c in range(7):
                col = (bi * 2 + bj) * 7 + c # 0..27
                
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U2[i * m + j, col] = S_U[di * 2 + dj, c]
                        
                        # V index: j (col of A) and k' (row of B_right)
                        # j is fixed by bj, dj.
                        # k' is 0 or 1.
                        # In V2, row is j * p + k_full.
                        # k_full = 4 + k'.
                        for dk in range(2):
                            k_prime = dk
                            k_full = 4 + k_prime
                            # j in V2 corresponds to the column of A, which is j.
                            # But Strassen V uses j_local and k_local.
                            # j_local = dj? No.
                            # In Strassen, V row is (j_local, k_local).
                            # Here j_local corresponds to the column index within the 2x2 block of A.
                            # But A's columns are j.
                            # For A00, j is 0,1. j_local = j.
                            # For A01, j is 2,3. j_local = j - 2.
                            # So j_local = dj.
                            # k_local = k_prime.
                            
                            # Wait, the B block is B_top or B_bot.
                            # B_top uses rows j in 0,1 of B.
                            # B_bot uses rows j in 2,3 of B.
                            # So the "j" in the B matrix is the same as the "j" in the A matrix?
                            # Yes, because it's A * B.
                            # A[i, j] * B[j, k].
                            # So the index j matches.
                            
                            # So for a given A block (bi, bj), the j values are fixed.
                            # And the B block must use the same j values.
                            # If bj=0, j in 0,1. B_top uses j in 0,1. Correct.
                            # If bj=1, j in 2,3. B_bot uses j in 2,3. Correct.
                            
                            V2[j * p + k_full, col] = S_V[dj * 2 + k_prime, c]
                            
                            # W index: k_full and i.
                            # W row is k * n + i.
                            # k_local = k_prime.
                            # i_local = di.
                            W2[k_full * n + i, col] = S_W[k_prime * 2 + di, c]

    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1_full, V2])
    W = np.hstack([W1_full, W2])
    
    # Remove zero columns
    mask = np.any(U != 0, axis=0) | np.any(V != 0, axis=0) | np.any(W != 0, axis=0)
    U_final = U[:, mask]
    V_final = V[:, mask]
    W_final = W[:, mask]
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
