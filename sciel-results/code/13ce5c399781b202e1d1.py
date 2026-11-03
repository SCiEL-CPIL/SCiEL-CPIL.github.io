import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We use a different structural approach: instead of splitting into 2x2*2x2 blocks,
    # we use a direct rank-7 decomposition for 2x4*4x8 by treating it as two independent
    # 2x2*2x8 multiplications and using a rank-14 decomposition for each 2x2*2x8.
    #
    # Actually, let me think more carefully. The key insight is that 2x4 * 4x8 can be
    # seen as computing C = A * B where A is 2x4 and B is 4x8.
    #
    # Alternative approach: Use the fact that we can write this as a sum of rank-1 terms
    # more efficiently by exploiting the structure.
    #
    # Let's try: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can group the j index. For each pair of j values, we have a 2x2 * 2x8 subproblem.
    # But instead of using Strassen on 2x2*2x2 blocks, let's use a direct approach.
    #
    # Actually, the current approach of rank 56 is likely close to optimal for this structure.
    # Let me try a different decomposition that might achieve lower rank.
    #
    # Key idea: Instead of 8 independent 2x2*2x2 Strassen decompositions (rank 7 each = 56),
    # can we share some of the intermediate products?
    #
    # Consider: C = A * B where A is 2x4, B is 4x8.
    # Write A = [A0 A1] where A0, A1 are 2x2.
    # Write B = [B0; B1] where B0, B1 are 2x8.
    # Then C = A0*B0 + A1*B1.
    #
    # Each A_i * B_i is a 2x2 * 2x8 multiplication.
    # For 2x2 * 2x8, we can use the fact that B_i has 8 columns, so we can view it as 4 blocks of 2x2.
    # Each 2x2 * 2x2 can use Strassen with rank 7, so total 4*7 = 28 for each A_i*B_i.
    # Total: 2*28 = 56. Same as before.
    #
    # Can we do better? What if we use a different decomposition for 2x2 * 2x8?
    # The rank of 2x2 * 2x8 is at most 2*2*8 = 32 (naive) but can be reduced.
    # Actually, for 2x2 * 2x8, the minimum rank is not well-known, but we can try to find
    # a decomposition with rank less than 28.
    #
    # Let me try a different approach: use the Winograd scheme or other known fast matrix multiplication.
    #
    # Actually, let me just stick with the rank-56 approach but implement it more cleanly.
    # The current implementation seems correct, so I'll keep it.
    
    # Strassen components for 2x2 matrix multiplication
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # r=2 (M3)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # r=3 (M4)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0
                a_map[1] = 1
                a_map[2] = 4
                a_map[3] = 5
            else:
                a_map[0] = 2
                a_map[1] = 3
                a_map[2] = 6
                a_map[3] = 7
                
            b_map = [0]*4
            if s == 0:
                b_map[0] = 2*j
                b_map[1] = 2*j+1
                b_map[2] = 8 + 2*j
                b_map[3] = 8 + 2*j+1
            else:
                b_map[0] = 16 + 2*j
                b_map[1] = 16 + 2*j+1
                b_map[2] = 24 + 2*j
                b_map[3] = 24 + 2*j+1
                
            c_map = [0]*4
            c_map[0] = 2*j
            c_map[1] = 2*j+1
            c_map[2] = 2 + 2*j
            c_map[3] = 3 + 2*j

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc], r] += val
                        
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
