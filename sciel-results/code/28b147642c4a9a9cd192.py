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
    
    # We use a block Strassen decomposition.
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B into B_top (2x8) and B_bottom (2x8).
    # C = A_left * B_top + A_right * B_bottom.
    # Each term is a 2x2 * 2x8 multiplication.
    # A 2x2 * 2x8 multiplication can be done with 4 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 uses Strassen (rank 7).
    # So 2x2 * 2x8 uses rank 28.
    # Total rank = 2 * 28 = 56.
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Strassen components for 2x2 matrix multiplication
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U_local, V_local, W_local are 4x7 matrices
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # We have 8 blocks: 2 j-groups (top/bottom of B) x 4 k-groups (columns of C)
    # Actually, the split is:
    # j-group 0: A_left, B_top. Contributes to C.
    # j-group 1: A_right, B_bottom. Contributes to C.
    # For each j-group, we have 4 k-groups (pairs of columns in C).
    
    # Mapping for U (A indices)
    # A is 2x4. Index = i*4 + j.
    # For j-group 0 (A_left): j in {0,1}. Indices: i*4 + 0, i*4 + 1.
    # For j-group 1 (A_right): j in {2,3}. Indices: i*4 + 2, i*4 + 3.
    
    # Mapping for V (B indices)
    # B is 4x8. Index = j*8 + k.
    # For j-group 0 (B_top): j in {0,1}. Indices: j*8 + k.
    # For j-group 1 (B_bottom): j in {2,3}. Indices: j*8 + k.
    
    # Mapping for W (C indices)
    # C is 2x8. Index = k*2 + i (per task convention W[k*n+i]).
    # k-groups: 0:{0,1}, 1:{2,3}, 2:{4,5}, 3:{6,7}.
    
    for jg in range(2):
        for kg in range(4):
            base_r = (jg * 4 + kg) * 7
            
            # A indices for this j-group
            # Local A index 0 -> i=0, j=2*jg
            # Local A index 1 -> i=0, j=2*jg+1
            # Local A index 2 -> i=1, j=2*jg
            # Local A index 3 -> i=1, j=2*jg+1
            a_map = [
                0*4 + (2*jg),
                0*4 + (2*jg+1),
                1*4 + (2*jg),
                1*4 + (2*jg+1)
            ]
            
            # B indices for this j-group and k-group
            # Local B index 0 -> j=2*jg, k=2*kg
            # Local B index 1 -> j=2*jg, k=2*kg+1
            # Local B index 2 -> j=2*jg+1, k=2*kg
            # Local B index 3 -> j=2*jg+1, k=2*kg+1
            b_map = [
                (2*jg)*8 + (2*kg),
                (2*jg)*8 + (2*kg+1),
                (2*jg+1)*8 + (2*kg),
                (2*jg+1)*8 + (2*kg+1)
            ]
            
            # C indices for this k-group
            # Local C index 0 -> i=0, k=2*kg
            # Local C index 1 -> i=0, k=2*kg+1
            # Local C index 2 -> i=1, k=2*kg
            # Local C index 3 -> i=1, k=2*kg+1
            # W index = k*2 + i
            c_map = [
                (2*kg)*2 + 0,
                (2*kg+1)*2 + 0,
                (2*kg)*2 + 1,
                (2*kg+1)*2 + 1
            ]
            
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
