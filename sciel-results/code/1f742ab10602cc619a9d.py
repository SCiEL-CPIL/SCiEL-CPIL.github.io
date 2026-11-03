"""Matrix multiplication 2x4 by 4x8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 64
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to construct rank 56 using Strassen-like blocks
    # 8 blocks of 2x2x2, each rank 7 -> 56 total
    target_rank = 56
    U_try = np.zeros((n*m, target_rank))
    V_try = np.zeros((m*p, target_rank))
    W_try = np.zeros((p*n, target_rank))
    
    col = 0
    for block_type in range(2): # 0: A1/Btop, 1: A2/Bbot
        for block_idx in range(4): # 4 column blocks in B
        
            # Define global indices for A entries
            if block_type == 0:
                # A1: Cols 0, 1
                a_00 = (0, 0)
                a_01 = (0, 1)
                a_10 = (1, 0)
                a_11 = (1, 1)
            else:
                # A2: Cols 2, 3
                a_00 = (0, 2)
                a_01 = (0, 3)
                a_10 = (1, 2)
                a_11 = (1, 3)
                
            # Define global indices for B entries
            b_row_0 = 0 if block_type == 0 else 2
            b_row_1 = 1 if block_type == 0 else 3
            
            b_00 = (b_row_0, 2*block_idx)
            b_01 = (b_row_0, 2*block_idx+1)
            b_10 = (b_row_1, 2*block_idx)
            b_11 = (b_row_1, 2*block_idx+1)
            
            # The 7 products P1..P7
            terms = []
            
            # P1
            u1 = {a_00: 1, a_11: 1}
            v1 = {b_00: 1, b_11: 1}
            w1 = {(0, 2*block_idx): 1, (1, 2*block_idx+1): 1}
            terms.append((u1, v1, w1))
            
            # P2
            u2 = {a_10: 1, a_11: 1}
            v2 = {b_00: 1}
            w2 = {(1, 2*block_idx): 1, (1, 2*block_idx+1): -1}
            terms.append((u2, v2, w2))
            
            # P3
            u3 = {a_00: 1}
            v3 = {b_01: 1, b_11: -1}
            w3 = {(0, 2*block_idx+1): 1, (1, 2*block_idx+1): 1}
            terms.append((u3, v3, w3))
            
            # P4
            u4 = {a_11: 1}
            v4 = {b_10: 1, b_00: -1}
            w4 = {(0, 2*block_idx): 1, (1, 2*block_idx): 1}
            terms.append((u4, v4, w4))
            
            # P5
            u5 = {a_00: 1, a_01: 1}
            v5 = {b_11: 1}
            w5 = {(0, 2*block_idx): -1, (0, 2*block_idx+1): 1}
            terms.append((u5, v5, w5))
            
            # P6
            u6 = {a_10: 1, a_00: -1}
            v6 = {b_00: 1, b_01: 1}
            w6 = {(1, 2*block_idx+1): 1}
            terms.append((u6, v6, w6))
            
            # P7
            u7 = {a_01: 1, a_11: -1}
            v7 = {b_10: 1, b_11: 1}
            w7 = {(0, 2*block_idx): 1}
            terms.append((u7, v7, w7))
            
            for u_dict, v_dict, w_dict in terms:
                # Fill U
                for (i, j), val in u_dict.items():
                    U_try[i*m+j, col] = val
                # Fill V
                for (j, k), val in v_dict.items():
                    V_try[j*p+k, col] = val
                # Fill W
                for (i, k), val in w_dict.items():
                    W_try[k*n+i, col] = val
                col += 1
    
    # Validate the rank 56 solution
    def validate(U_try, V_try, W_try, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if validate(U_try, V_try, W_try, target_rank):
        best_U, best_V, best_W = U_try, V_try, W_try
        best_rank = target_rank
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        pass

    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
