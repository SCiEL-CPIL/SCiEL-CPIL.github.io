"""Rewritten program using algorithmic generation for tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Approach: Algorithmic generation.
    Decomposition:
    Split k into {0,1} and {2,3}.
    Each part is a 2x3x2 multiplication.
    For 2x3x2, split j into {0,1} and {2}.
    j in {0,1} is a 2x2x2 block (Rank 7 via Strassen).
    j in {2} is a 2x1x2 block (Rank 4 via Schoolbook).
    Total rank per 2x3x2 block: 7 + 4 = 11.
    Total rank: 11 + 11 = 22.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # Define the rank-7 decomposition for 2x2x2 (Strassen)
    # u, v, w are vectors of length 4
    # Indices: u[i*2+j], v[j*2+k], w[k*2+i]
    strassen_terms = [
        ((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)),
        ((0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1)),
        ((0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0)),
        ((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)),
        ((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0)),
        ((0, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)),
        ((0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 0, 1)),
    ]
    
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_block(k_offset):
        nonlocal col
        # Add 2x3x2 block for k in {k_offset, k_offset+1}
        
        # Part 1: j in {0, 1} (2x2x2 Strassen)
        for u_vec, v_vec, w_vec in strassen_terms:
            # u_vec corresponds to A[i, j] for i in {0,1}, j in {0,1}
            # v_vec corresponds to B[j, k] for j in {0,1}, k in {0,1}
            # w_vec corresponds to C[i, k] for i in {0,1}, k in {0,1}
            
            for i in range(2):
                for j in range(2):
                    val = u_vec[i * 2 + j]
                    if val != 0:
                        U[i * m + j][col] += val
                        # V index: j * p + (k_offset + k_local)
                        # But v_vec index is j_local * 2 + k_local
                        # We need to map v_vec index to global V index
                        pass
            
            # It's easier to just iterate over the vector indices
            for idx_u in range(4):
                if u_vec[idx_u] != 0:
                    i_local = idx_u // 2
                    j_local = idx_u % 2
                    U[i_local * m + j_local][col] += u_vec[idx_u]
            
            for idx_v in range(4):
                if v_vec[idx_v] != 0:
                    j_local = idx_v // 2
                    k_local = idx_v % 2
                    V[j_local * p + (k_offset + k_local)][col] += v_vec[idx_v]
            
            for idx_w in range(4):
                if w_vec[idx_w] != 0:
                    k_local = idx_w // 2
                    i_local = idx_w % 2
                    W[(k_offset + k_local) * n + i_local][col] += w_vec[idx_w]
            
            col += 1
        
        # Part 2: j in {2} (2x1x2 Schoolbook)
        # A[i, 2] * B[2, k] -> C[i, k]
        for i in range(2):
            for k_local in range(2):
                U[i * m + 2][col] = 1.0
                V[2 * p + (k_offset + k_local)][col] = 1.0
                W[(k_offset + k_local) * n + i][col] = 1.0
                col += 1

    add_block(0)
    add_block(2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
