import time
import numpy as np

def construct():
    n, m, p = 3, 4, 6
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (18, rank)
    
    # We will build the decomposition by summing contributions from blocks.
    # Block structure:
    # A cols (j) split into j_blocks = [(0, 2), (2, 4)]
    # B rows (j) same as A cols.
    # B cols (k) split into k_blocks = [(0, 2), (2, 4), (4, 6)]
    # A rows (i) are 0, 1, 2.
    
    # For each j_block, k_block, and i_row:
    # We add a rank-1 term for the 1x2x2 sub-tensor.
    
    rank = 0
    U = np.zeros((n * m, 36), dtype=float)
    V = np.zeros((m * p, 36), dtype=float)
    W = np.zeros((p * n, 36), dtype=float)
    
    col = 0
    for j_s, j_e in [(0, 2), (2, 4)]:
        for k_s, k_e in [(0, 2), (2, 4), (4, 6)]:
            for i in range(n):
                # This corresponds to the tensor T_{ijk} for i fixed, j in [j_s, j_e), k in [k_s, k_e)
                # The sub-tensor is 1x2x2 with all 1s.
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                # The rank-1 decomposition for 1x2x2 all-ones:
                # U_{ij} = 1 for j in block
                # V_{jk} = 1 for j,k in block
                # W_{ki} = 1 for k in block
                
                # We add 1 rank term.
                for j in range(j_s, j_e):
                    U[i * m + j, col] = 1.0
                for j in range(j_s, j_e):
                    for k in range(k_s, k_e):
                        V[j * p + k, col] = 1.0
                for k in range(k_s, k_e):
                    W[k * n + i, col] = 1.0
                
                col += 1
                
    assert col == 36
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
