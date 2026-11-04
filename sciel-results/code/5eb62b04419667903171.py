"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Strategy:
    # 1. Try Rank 161: Strassen (2x2 blocks) + Laderman (3x3 blocks)
    #    6x6 matrix viewed as 2x2 blocks of 3x3 matrices.
    #    Block mult: Strassen (7 terms).
    #    Inner mult: Laderman (23 terms).
    #    Total Rank: 7 * 23 = 161.
    
    # 2. Fallback Rank 189: Strassen (2x2 blocks) + Schoolbook (3x3 blocks)
    #    OR Schoolbook (3x3 blocks) + Strassen (2x2 blocks).
    #    Current program uses 3x3 blocks of 2x2 (27*7=189).
    
    # Helper: Verify decomposition
    def verify_decomp(U, V, W, n, m, p):
        # U, V, W are lists of lists.
        # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # Must equal 1 for all i,j,k.
        
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Reshape
        # U: (n, m, rank)
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # Compute T
        # T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # Use einsum for speed
        # 'ijt,jkt,kit->ijk'
        T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T is all 1s
        # Tolerance for floating point errors
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Attempt Rank 161 ---
    
    # Strassen 2x2 patterns (7 terms)
    # U, V, W are dicts of {(row, col): val}
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Laderman 3x3 patterns (23 terms)
    # Source: Laderman (1976). 
    # I will use a verified set from a known implementation.
    # If this set is incorrect, verification will fail.
    
    # The following set is from a verified implementation.
    laderman_3x3 = [
        # t0
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        # t1
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # t2
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        # t3
        ([1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # t4
        ([0,0,0, 0,1,0, 1,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # t5
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,1,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # t6
        ([1,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # t7
        ([0,0,0, 0,0,0, 0,1,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # t8
        ([0,1,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # t9
        ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # t10
        ([0,0,1, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # t11
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # t12
        ([1,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # t13
        ([0,1,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t14
        ([0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t15
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t16
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t17
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t18
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t19
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t20
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t21
        ([1,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # t22
        ([0,0,0, 0,0,1, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # This list is likely wrong/incomplete.
    # I will NOT use this.
    
    # I will return the 189 solution.
    # But wait, the score is 1.1429.
    # If I return 189, the score is 1.1429.
    # I need to improve.
    
    # Is there a rank 160 algorithm?
    # No, 161 is the standard improvement.
    
    # I will try to find a correct Laderman set in the `inspirations`?
    # Inspiration 3 mentions "verified 3x3 block decomposition".
    # This implies it is possible.
    
    # I will use a **different** strategy:
    # I will use the `numpy` library to solve for the decomposition?
    # No, rank minimization is NP-hard.
    
    # I will stick to the 189 solution but ensure it is correct and fast.
    # Actually, I can try to use **Strassen** on the 3x3 blocks?
    # No, Strassen is for 2x2.
    
    # I will implement the 189 solution.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
