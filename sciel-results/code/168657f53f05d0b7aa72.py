import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    
    # Rank 18 decomposition
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Helper to set Strassen block for given offsets
    # i_offset: 0 or 1? No, i is global 0..1.
    # j_offset: 0 or 1 (for the 2x2 part)
    # k_offset: 0, 1, 2, 3, 4, 5
    # rank_offset: 0..8 for first block, 9..17 for second
    
    def add_strassen_block(j_base, k_base, rank_base):
        # j_base is 0 (so j=0,1)
        # k_base is 0 (so k=0,1) or 3 (so k=3,4)
        # We map local i, j, k to global
        # Local i is 0,1 -> Global i is 0,1
        # Local j is 0,1 -> Global j is j_base + local_j
        # Local k is 0,1 -> Global k is k_base + local_k
        
        # M1
        r = rank_base + 0
        U[0 * m + 0, r] += 1.0
        U[1 * m + 1, r] += 1.0
        V[0 * p + 0, r] += 1.0
        V[1 * p + 1, r] += 1.0
        W[0 * n + 0, r] += 1.0
        W[1 * n + 1, r] += 1.0
        
        # M2
        r = rank_base + 1
        U[1 * m + 0, r] += 1.0
        U[1 * m + 1, r] += 1.0
        V[0 * p + 0, r] += 1.0
        W[0 * n + 1, r] += 1.0
        W[1 * n + 1, r] += -1.0
        
        # M3
        r = rank_base + 2
        U[0 * m + 0, r] += 1.0
        V[0 * p + 1, r] += 1.0
        V[1 * p + 1, r] += -1.0
        W[1 * n + 0, r] += 1.0
        W[1 * n + 1, r] += 1.0
        
        # M4
        r = rank_base + 3
        U[1 * m + 1, r] += 1.0
        V[1 * p + 0, r] += -1.0
        V[0 * p + 0, r] += 1.0
        W[0 * n + 0, r] += 1.0
        W[0 * n + 1, r] += 1.0
        
        # M5
        r = rank_base + 4
        U[0 * m + 0, r] += 1.0
        U[0 * m + 1, r] += 1.0
        V[1 * p + 1, r] += 1.0
        W[0 * n + 0, r] += -1.0
        W[1 * n + 0, r] += 1.0
        
        # M6
        r = rank_base + 5
        U[1 * m + 0, r] += 1.0
        U[0 * m + 0, r] += -1.0
        V[0 * p + 0, r] += 1.0
        V[0 * p + 1, r] += 1.0
        W[1 * n + 1, r] += 1.0
        
        # M7
        r = rank_base + 6
        U[0 * m + 1, r] += 1.0
        U[1 * m + 1, r] += -1.0
        V[1 * p + 0, r] += 1.0
        V[1 * p + 1, r] += 1.0
        W[0 * n + 0, r] += 1.0
        
        # M8 (Rank 1 for j=0,1 and k=2)
        r = rank_base + 7
        U[0 * m + 0, r] += 1.0
        U[1 * m + 0, r] += 1.0
        U[0 * m + 1, r] += 1.0
        U[1 * m + 1, r] += 1.0
        V[0 * p + 2, r] += 1.0
        V[1 * p + 2, r] += 1.0
        W[2 * n + 0, r] += 1.0
        W[2 * n + 1, r] += 1.0
        
        # M9 (Rank 1 for j=2 and k=0,1,2)
        r = rank_base + 8
        U[0 * m + 2, r] += 1.0
        U[1 * m + 2, r] += 1.0
        V[2 * p + 0, r] += 1.0
        V[2 * p + 1, r] += 1.0
        V[2 * p + 2, r] += 1.0
        W[0 * n + 0, r] += 1.0
        W[0 * n + 1, r] += 1.0
        W[1 * n + 0, r] += 1.0
        W[1 * n + 1, r] += 1.0

    # Block 1: k in 0,1,2
    add_strassen_block(0, 0, 0)
    
    # Block 2: k in 3,4,5
    # k_base = 3
    # We need to adjust the k indices in the helper or pass offset.
    # The helper above uses hardcoded k=0,1,2 for the 2x2x2 part and k=2 for the 2x2x1 part.
    # For the second block, k should be 3,4 for 2x2x2 and 5 for 2x2x1.
    
    # Let's write a more flexible helper or just duplicate with offsets.
    
    def add_strassen_block_offset(j_base, k_base, rank_base):
        # Local i: 0,1
        # Local j: 0,1 -> Global j: j_base + lj
        # Local k: 0,1 -> Global k: k_base + lk
        # Extra k for 2x2x1: k_base + 2
        
        # M1
        r = rank_base + 0
        U[0 * m + j_base + 0, r] += 1.0
        U[1 * m + j_base + 1, r] += 1.0
        V[(j_base + 0) * p + (k_base + 0), r] += 1.0
        V[(j_base + 1) * p + (k_base + 1), r] += 1.0
        W[(k_base + 0) * n + 0, r] += 1.0
        W[(k_base + 1) * n + 1, r] += 1.0
        
        # M2
        r = rank_base + 1
        U[1 * m + j_base + 0, r] += 1.0
        U[1 * m + j_base + 1, r] += 1.0
        V[(j_base + 0) * p + (k_base + 0), r] += 1.0
        W[(k_base + 0) * n + 1, r] += 1.0
        W[(k_base + 1) * n + 1, r] += -1.0
        
        # M3
        r = rank_base + 2
        U[0 * m + j_base + 0, r] += 1.0
        V[(j_base + 0) * p + (k_base + 1), r] += 1.0
        V[(j_base + 1) * p + (k_base + 1), r] += -1.0
        W[(k_base + 1) * n + 0, r] += 1.0
        W[(k_base + 1) * n + 1, r] += 1.0
        
        # M4
        r = rank_base + 3
        U[1 * m + j_base + 1, r] += 1.0
        V[(j_base + 1) * p + (k_base + 0), r] += -1.0
        V[(j_base + 0) * p + (k_base + 0), r] += 1.0
        W[(k_base + 0) * n + 0, r] += 1.0
        W[(k_base + 0) * n + 1, r] += 1.0
        
        # M5
        r = rank_base + 4
        U[0 * m + j_base + 0, r] += 1.0
        U[0 * m + j_base + 1, r] += 1.0
        V[(j_base + 1) * p + (k_base + 1), r] += 1.0
        W[(k_base + 0) * n + 0, r] += -1.0
        W[(k_base + 1) * n + 0, r] += 1.0
        
        # M6
        r = rank_base + 5
        U[1 * m + j_base + 0, r] += 1.0
        U[0 * m + j_base + 0, r] += -1.0
        V[(j_base + 0) * p + (k_base + 0), r] += 1.0
        V[(j_base + 0) * p + (k_base + 1), r] += 1.0
        W[(k_base + 1) * n + 1, r] += 1.0
        
        # M7
        r = rank_base + 6
        U[0 * m + j_base + 1, r] += 1.0
        U[1 * m + j_base + 1, r] += -1.0
        V[(j_base + 1) * p + (k_base + 0), r] += 1.0
        V[(j_base + 1) * p + (k_base + 1), r] += 1.0
        W[(k_base + 0) * n + 0, r] += 1.0
        
        # M8 (Rank 1 for j=j_base, j_base+1 and k=k_base+2)
        r = rank_base + 7
        U[0 * m + j_base + 0, r] += 1.0
        U[1 * m + j_base + 0, r] += 1.0
        U[0 * m + j_base + 1, r] += 1.0
        U[1 * m + j_base + 1, r] += 1.0
        V[(j_base + 0) * p + (k_base + 2), r] += 1.0
        V[(j_base + 1) * p + (k_base + 2), r] += 1.0
        W[(k_base + 2) * n + 0, r] += 1.0
        W[(k_base + 2) * n + 1, r] += 1.0
        
        # M9 (Rank 1 for j=j_base+2 and k=k_base, k_base+1, k_base+2)
        r = rank_base + 8
        U[0 * m + j_base + 2, r] += 1.0
        U[1 * m + j_base + 2, r] += 1.0
        V[(j_base + 2) * p + (k_base + 0), r] += 1.0
        V[(j_base + 2) * p + (k_base + 1), r] += 1.0
        V[(j_base + 2) * p + (k_base + 2), r] += 1.0
        W[(k_base + 0) * n + 0, r] += 1.0
        W[(k_base + 0) * n + 1, r] += 1.0
        W[(k_base + 1) * n + 0, r] += 1.0
        W[(k_base + 1) * n + 1, r] += 1.0
        W[(k_base + 2) * n + 0, r] += 1.0
        W[(k_base + 2) * n + 1, r] += 1.0

    # Call for Block 1
    add_strassen_block_offset(0, 0, 0)
    # Call for Block 2
    add_strassen_block_offset(0, 3, 9)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
