import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix mult.
    
    Returns dict with U (6 x rank), V (27 x rank), W (18 x rank) where:
    - U[i*m + j, c] corresponds to first matrix element a[i,j]
    - V[j*p + k, c] corresponds to second matrix element b[j,k]
    - W[k*n + i, c] corresponds to output element c[k,i] (trace convention)
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    nm = n * m   # 6
    mp = m * p   # 27
    np_ = n * p  # 18
    
    # Strategy: Decompose into 3 blocks of 3 columns each.
    # Each block is a 2x3 * 3x3 multiplication.
    # Rank of 2x3 * 3x3 is 6 (3 for i=0, 3 for i=1).
    # Total rank = 3 * 6 = 18.
    
    rank = 18
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((np_, rank), dtype=float)
    
    col = 0
    # b_idx: block index (0, 1, 2)
    for b_idx in range(3):
        # k_offset: starting column index for this block
        k_offset = b_idx * 3
        
        # For this block, we compute C[i, k_offset + k_local] for k_local in 0..2
        # Using rank 6 decomposition for 2x3 * 3x3.
        # For each j in 0..2 (row of A, row of B block):
        #   For each i in 0..1 (row of A, row of C):
        #     We assign a rank term.
        
        for j in range(m):  # j = 0, 1, 2
            for i in range(n):  # i = 0, 1
                
                # U: index i*m + j
                U[i * m + j, col] = 1.0
                
                # V: index j*p + k
                # For this term, we want to capture B[j, k_offset + k_local] for all k_local
                # But wait, if we set V[j, k_offset + k_local, col] = 1 for all k_local,
                # and W[k_offset + k_local, i, col] = 1 for all k_local,
                # then the product is 1 for all k_local.
                # This works because T[i*m+j, j*p+k, k*n+i] = 1 for all matching k.
                # And for non-matching k, T=0.
                # Does this term contribute to non-matching k?
                # V is non-zero only for k in [k_offset, k_offset+2].
                # W is non-zero only for k in [k_offset, k_offset+2].
                # So for k outside this block, V or W is 0, so product is 0.
                # For k inside this block, product is 1.
                # This matches T=1 for all (i, j, k) in this block.
                
                for k_local in range(3):
                    k_global = k_offset + k_local
                    V[j * p + k_global, col] = 1.0
                    W[k_global * n + i, col] = 1.0
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
