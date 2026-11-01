import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Rank 42 solution using Strassen blocks.
    # Decomposition: M_{2,4,6} = M_{2,2,2} \otimes M_{1,2,2} is NOT used directly as a Kronecker product of tensors in the code,
    # but rather by tiling Strassen 2x2 blocks.
    # We split m=4 into two blocks of 2 (j=0,1 and j=2,3).
    # We split p=6 into three blocks of 2 (k=0,1; k=2,3; k=4,5).
    # For each (j_block, k_block) pair, we apply Strassen 2x2 @ 2x2 (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen 2x2 @ 2x2 decomposition coefficients
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # U coefficients (A part)
    strassen_U = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # V coefficients (B part)
    strassen_V = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1}        # M7: B10 + B11
    ]
    
    # W coefficients (C part)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W = [
        {0: 1, 3: 1},       # M1: C00 + C11
        {2: 1, 3: -1},      # M2: C10 - C11
        {1: 1, 3: 1},       # M3: C01 + C11
        {0: 1, 2: 1},       # M4: C00 + C10
        {0: -1, 1: 1},      # M5: -C00 + C01
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
    ]
    
    col = 0
    # j_block: 0 (j=0,1), 1 (j=2,3)
    for j_block in range(2):
        # k_block: 0 (k=0,1), 1 (k=2,3), 2 (k=4,5)
        for k_block in range(3):
            for prod_idx in range(7):
                r = col + prod_idx
                
                # Fill U: A[i, j]
                # i in {0,1}, j in {j_block*2, j_block*2+1}
                for local_idx, sign in strassen_U[prod_idx].items():
                    gi = local_idx // 2
                    gj = local_idx % 2
                    global_j = j_block * 2 + gj
                    U[gi * m + global_j, r] = sign
                
                # Fill V: B[j, k]
                # j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                for local_idx, sign in strassen_V[prod_idx].items():
                    gj = local_idx // 2
                    gk = local_idx % 2
                    global_j = j_block * 2 + gj
                    global_k = k_block * 2 + gk
                    V[global_j * p + global_k, r] = sign
                
                # Fill W: C[i, k]
                # i in {0,1}, k in {k_block*2, k_block*2+1}
                # Output format: W[k*n + i, r]
                for local_idx, sign in strassen_W[prod_idx].items():
                    gi = local_idx // 2
                    gk = local_idx % 2
                    global_k = k_block * 2 + gk
                    W[global_k * n + gi, r] = sign
            
            col += 7
            
    # Time guard
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
