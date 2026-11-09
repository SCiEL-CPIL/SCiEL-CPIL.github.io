import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Decompose 4x5x5 into blocks:
    # Split rows of A (4) into 2+2, cols of A/rows of B (5) into 2+3, cols of B (5) into 2+3
    # This gives 4 sub-problems: (2x2x2), (2x2x3), (2x3x2), (2x3x3)
    
    # For each sub-block, use schoolbook or known optimal if available
    # 2x2x2: Strassen rank 7 (but we need exact, so use schoolbook rank 8 for safety, 
    #         or implement Strassen carefully)
    # Actually, let's use a hybrid: use Strassen for 2x2x2, schoolbook for others
    
    # Build up the full decomposition using block structure
    
    # Define block sizes
    # A is 4x5: split into [2,2] x [2,3]
    # B is 5x5: split into [2,3] x [2,3]
    # C is 4x5: split into [2,2] x [2,3]
    
    a_row_splits = [2, 2]  # 4 = 2+2
    a_col_splits = [2, 3]  # 5 = 2+3
    b_row_splits = [2, 3]  # 5 = 2+3
    b_col_splits = [2, 3]  # 5 = 2+3
    
    # Compute cumulative offsets
    a_row_offsets = [0]
    for s in a_row_splits:
        a_row_offsets.append(a_row_offsets[-1] + s)
    a_col_offsets = [0]
    for s in a_col_splits:
        a_col_offsets.append(a_col_offsets[-1] + s)
    b_row_offsets = [0]
    for s in b_row_splits:
        b_row_offsets.append(b_row_offsets[-1] + s)
    b_col_offsets = [0]
    for s in b_col_splits:
        b_col_offsets.append(b_col_offsets[-1] + s)
    
    # For each pair of blocks (i,j) where i indexes A's row block, j indexes A's col block / B's row block,
    # and k indexes B's col block, we have a sub-problem of size (a_row_splits[i] x a_col_splits[j] x b_col_splits[k])
    
    # Collect all rank-1 terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    for i in range(len(a_row_splits)):
        for j in range(len(a_col_splits)):
            for k in range(len(b_col_splits)):
                ni = a_row_splits[i]
                mj = a_col_splits[j]
                pk = b_col_splits[k]
                
                # Use schoolbook for this sub-block (rank = ni*mj*pk)
                # We could use Strassen for 2x2x2 but let's keep it simple first
                for ii in range(ni):
                    for jj in range(mj):
                        for kk in range(pk):
                            # Global indices
                            gi = a_row_offsets[i] + ii  # global row in A
                            gj = a_col_offsets[j] + jj  # global col in A / row in B
                            gk = b_col_offsets[k] + kk  # global col in B / col in C
                            
                            # U index: gi * m + gj
                            # V index: gj * p + gk
                            # W index: gk * n + gi
                            U_terms.append((gi * m + gj, 1))
                            V_terms.append((gj * p + gk, 1))
                            W_terms.append((gk * n + gi, 1))
    
    # Total rank
    rank = len(U_terms)
    
    # Build sparse U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx in range(rank):
        U[U_terms[col_idx][0]][col_idx] = U_terms[col_idx][1]
        V[V_terms[col_idx][0]][col_idx] = V_terms[col_idx][1]
        W[W_terms[col_idx][0]][col_idx] = W_terms[col_idx][1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
