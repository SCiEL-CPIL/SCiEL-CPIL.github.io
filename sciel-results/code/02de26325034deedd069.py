"""
Matrix multiplication 5x5x5 tensor decomposition.

Approach: Block decomposition with 5 = 2 + 2 + 1 partition.
- Use Strassen's algorithm (rank 7) for 2x2x2 blocks
- Use schoolbook for all other blocks

Total rank: 117
"""

DIMENSIONS = [5, 5, 5]

def _schoolbook_factors(n, m, p):
    """Return U, V, W for schoolbook n x m x p matrix multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_factors():
    """
    Return U, V, W for Strassen's 2x2x2 algorithm.
    """
    rank = 7
    U = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    V = [
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, -1, 1],
    ]
    W = [
        [1, 0, 0, -1, 1, 0, 0],
        [0, 1, 0, 0, -1, 0, 1],
        [0, 0, 1, 1, 0, -1, 0],
        [1, 0, -1, 0, 0, 1, 0],
    ]
    return U, V, W

def _get_factors(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen_factors()
    else:
        return _schoolbook_factors(n, m, p)

def construct():
    n, m, p = 5, 5, 5
    # Partition 5 into 2, 2, 1
    block_sizes = [2, 2, 1]
    block_starts = [0, 2, 4]
    
    # Calculate total rank
    total_rank = 0
    for k in range(3):
        for l in range(3):
            for mid in range(3):
                nk = block_sizes[k]
                nm = block_sizes[mid]
                np = block_sizes[l]
                # Product is nk x nm * nm x np
                _, _, _ = _get_factors(nk, nm, np)
                # Rank is the number of columns in U
                # We need to calculate rank without constructing to be efficient? 
                # Or just construct and sum.
                # For simplicity, construct and sum.
                U_sub, V_sub, W_sub = _get_factors(nk, nm, np)
                total_rank += len(U_sub[0])
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    for k in range(3):
        for l in range(3):
            for mid in range(3):
                nk = block_sizes[k]
                nm = block_sizes[mid]
                np = block_sizes[l]
                
                U_sub, V_sub, W_sub = _get_factors(nk, nm, np)
                sub_rank = len(U_sub[0])
                
                # Scatter U
                # Global row i_global = block_starts[k] + i_loc
                # Global col j_global = block_starts[mid] + j_loc
                # U_sub index: i_loc * nm + j_loc
                for i_loc in range(nk):
                    for j_loc in range(nm):
                        i_global = block_starts[k] + i_loc
                        j_global = block_starts[mid] + j_loc
                        u_global_idx = i_global * m + j_global
                        u_sub_idx = i_loc * nm + j_loc
                        for r in range(sub_rank):
                            U[u_global_idx][current_rank + r] += U_sub[u_sub_idx][r]
                
                # Scatter V
                # Global row j_global = block_starts[mid] + j_loc
                # Global col k_global = block_starts[l] + k_loc
                # V_sub index: j_loc * np + k_loc
                for j_loc in range(nm):
                    for k_loc in range(np):
                        j_global = block_starts[mid] + j_loc
                        k_global = block_starts[l] + k_loc
                        v_global_idx = j_global * p + k_global
                        v_sub_idx = j_loc * np + k_loc
                        for r in range(sub_rank):
                            V[v_global_idx][current_rank + r] += V_sub[v_sub_idx][r]
                
                # Scatter W
                # Global row k_global = block_starts[l] + k_loc
                # Global col i_global = block_starts[k] + i_loc
                # W_sub index: k_loc * nk + i_loc
                for k_loc in range(np):
                    for i_loc in range(nk):
                        k_global = block_starts[l] + k_loc
                        i_global = block_starts[k] + i_loc
                        w_global_idx = k_global * n + i_global
                        w_sub_idx = k_loc * nk + i_loc
                        for r in range(sub_rank):
                            W[w_global_idx][current_rank + r] += W_sub[w_sub_idx][r]
                
                current_rank += sub_rank
                
    return {"U": U, "V": V, "W": W}
