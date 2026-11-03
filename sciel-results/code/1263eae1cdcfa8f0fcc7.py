import time
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
    
    # We use 4 independent 2x2 Strassen blocks.
    # Block (h, s) where h is half of middle dim (0,1) and s is half of output dim (0,1)
    # h=0 corresponds to j in {0,1}, h=1 corresponds to j in {2,3}
    # s corresponds to k in {2s, 2s+1, 2s+2, 2s+3}
    
    total_rank = 28
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Strassen coefficients for 2x2 multiplication
    # Each tuple is (u_coeffs, v_coeffs, w_coeffs)
    # where each is a list of (local_idx, value)
    # Local indices for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Term 0: M1 = (A11 + A22) * (B11 + B22)
    # Term 1: M2 = (A21 + A22) * B11
    # Term 2: M3 = A11 * (B21 - B22)
    # Term 3: M4 = A22 * (B12 - B11)
    # Term 4: M5 = (A11 + A12) * B22
    # Term 5: M6 = (A21 - A11) * (B11 + B12)
    # Term 6: M7 = (A12 - A22) * (B21 + B22)
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    # Iterate over the 4 blocks
    # Block index b from 0 to 3
    # h = b // 2 (0 or 1, corresponding to j offset 0 or 2)
    # s = b % 2 (0 or 1, corresponding to k offset 0 or 2)
    
    for b in range(4):
        h = b // 2
        s = b % 2
        
        # Base indices for this block
        # A indices: row i (0,1), col j (2h, 2h+1)
        # Flattened A index: i * 4 + j
        # B indices: row j (2h, 2h+1), col k (2s, 2s+1, 2s+2, 2s+3)
        # Flattened B index: j * 8 + k
        # C indices: row k (2s, 2s+1, 2s+2, 2s+3), col i (0,1)
        # Flattened C index: k * 2 + i
        
        # Map local 2x2 indices (0,1,2,3) to global indices
        # Local 0: (0,0) -> Global A: 0*4 + (2h), B: (2h)*8 + (2s), C: (2s)*2 + 0
        # Local 1: (0,1) -> Global A: 0*4 + (2h+1), B: (2h)*8 + (2s+1), C: (2s+1)*2 + 0
        # Local 2: (1,0) -> Global A: 1*4 + (2h), B: (2h+1)*8 + (2s), C: (2s)*2 + 1
        # Local 3: (1,1) -> Global A: 1*4 + (2h+1), B: (2h+1)*8 + (2s+1), C: (2s+1)*2 + 1
        
        a_global = [
            0 * 4 + (2 * h),
            0 * 4 + (2 * h + 1),
            1 * 4 + (2 * h),
            1 * 4 + (2 * h + 1)
        ]
        b_global = [
            (2 * h) * 8 + (2 * s),
            (2 * h) * 8 + (2 * s + 1),
            (2 * h + 1) * 8 + (2 * s),
            (2 * h + 1) * 8 + (2 * s + 1)
        ]
        c_global = [
            (2 * s) * 2 + 0,
            (2 * s + 1) * 2 + 0,
            (2 * s) * 2 + 1,
            (2 * s + 1) * 2 + 1
        ]
        
        # Fill the rank slots for this block
        base_rank = b * 7
        
        for t in range(7):
            r = base_rank + t
            u_list, v_list, w_list = strassen_terms[t]
            
            for idx, val in u_list:
                U[a_global[idx], r] += val
            for idx, val in v_list:
                V[b_global[idx], r] += val
            for idx, val in w_list:
                W[c_global[idx], r] += val
        
        # Check time
        if time.time() - start_time > 140:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
