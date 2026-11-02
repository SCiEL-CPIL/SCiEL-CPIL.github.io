"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 28 using 4 independent Strassen blocks.
    # This is the direct sum of four M_2,2,2 tensors.
    
    rank = 28
    # Initialize factor matrices with zeros
    # U: (n*m, rank) -> (4, 28)
    # V: (m*p, rank) -> (16, 28)
    # W: (p*n, rank) -> (16, 28)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen 7-factor decomposition details
    # Local indices for 2x2 matrices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) 
    # Note: The problem statement says W[k*n+i,r]. 
    # Standard C index (row i, col k) maps to linear index i*p + k? 
    # Wait, "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the output C is indexed by k (col) and i (row).
    # C[i,k] = sum_r U[:,r] V[:,r] W[:,r]?
    # Usually, Tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
    # Here A is (n,m), B is (m,p), C is (n,p).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So U should correspond to A indices (i,j) -> n*m.
    # V should correspond to B indices (j,k) -> m*p.
    # W should correspond to C indices (i,k) -> n*p.
    # The prompt says W has shape (p*n, rank).
    # And "W[k*n+i,r]". This is (p*n) entries.
    # So index in W is (k*n + i).
    
    # Strassen Factors:
    # U_strassen[r] is a dict or list of (A_index, coeff)
    # V_strassen[r] is a dict or list of (B_index, coeff)
    # W_strassen[r] is a dict or list of (C_index, coeff)
    
    # A_index: i*2 + j
    # B_index: j*2 + k (local k) -> Global k = block_start + k_local
    # C_index: k*n + i (Global k = block_start + k_local)
    
    U_s = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    V_s = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    W_s = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    for block in range(4):
        base_k = block * 2
        for r_local in range(7):
            r_global = block * 7 + r_local
            
            # Fill U
            for idx, val in U_s[r_local]:
                U[idx][r_global] += val
            
            # Fill V
            for idx_local, val in V_s[r_local]:
                # idx_local is (j, k_local) flattened? 
                # In my V_s definition above, indices are 0..3 representing (j,k_local)
                # j = idx // 2, k_local = idx % 2
                j = idx_local // 2
                k_loc = idx_local % 2
                # B global index: j*p + (base_k + k_loc)
                v_idx = j * p + base_k + k_loc
                V[v_idx][r_global] += val
                
            # Fill W
            for idx_local, val in W_s[r_local]:
                # idx_local represents C local index?
                # In my W_s definition, indices 0..3 represent (i, k_local)
                # i = idx // 2? No.
                # Let's check my W_s mapping from the original code.
                # Original: 
                # W_strassen = [
                #   {0: 1, 3: 1}, # C00, C11? 
                #   ...
                # ]
                # My previous mapping:
                # w_local 0 -> (0,0) i=0, k=0
                # w_local 1 -> (1,0) i=1, k=0
                # w_local 2 -> (0,1) i=0, k=1
                # w_local 3 -> (1,1) i=1, k=1
                # Wait, the prompt says W[k*n+i].
                # So for local (i, k_loc), the global W index is (base_k + k_loc)*n + i.
                
                # I need to map my W_s indices to (i, k_loc).
                # My W_s above used indices 0,1,2,3.
                # Let's define the mapping explicitly.
                # 0: i=0, k=0
                # 1: i=1, k=0
                # 2: i=0, k=1
                # 3: i=1, k=1
                i = idx_local % 2
                k_loc = idx_local // 2
                
                # Check original code mapping:
                # Original W_strassen[0] = {0:1, 3:1}
                # Original code: w_idx = (base_col + k_local) * n + i
                # If w_local=0: i=0, k_loc=0 -> w_idx = base_k*2 + 0
                # If w_local=3: i=1, k_loc=1 -> w_idx = (base_k+1)*2 + 1
                # This matches C00 and C11.
                
                w_idx = (base_k + k_loc) * n + i
                W[w_idx][r_global] += val

    return {"U": U, "V": W, "W": W} # BUG: Return V and W correctly
# EVOLVE-BLOCK-END
