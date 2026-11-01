import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Strategy: Decompose into p independent (n, m, 1) tensor instances.
    # Each instance corresponds to one column of the output matrix.
    # The rank of (n, m, 1) tensor is n * m.
    
    rank_base = n * m
    total_rank = p * rank_base
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Iterate over each of the p independent (n, m, 1) instances
    for k in range(p):
        rank_offset = k * rank_base
        
        # For each instance, we decompose the (n, m, 1) tensor
        # which is sum_{i,j} e_i (A) * e_j (B) * e_i (W)
        # Note: The W index for the k-th column is k*n + i
        for i in range(n):
            for j in range(m):
                local_rank_idx = i * m + j
                global_rank_idx = rank_offset + local_rank_idx
                
                # U index: row i*m + j
                u_idx = i * m + j
                U[u_idx][global_rank_idx] = 1
                
                # V index: row j*p + k
                v_idx = j * p + k
                V[v_idx][global_rank_idx] = 1
                
                # W index: row k*n + i (trace convention)
                w_idx = k * n + i
                W[w_idx][global_rank_idx] = 1

    return {"U": U, "V": V, "W": W}
