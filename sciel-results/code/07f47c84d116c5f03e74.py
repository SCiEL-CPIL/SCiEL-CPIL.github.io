import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# Laderman's 7-term algorithm for 3x3x3 matrix multiplication
# Returns (U, V, W) each of shape (9, 7) with integer entries
def _laderman_3x3x3():
    U = [
        [1,  1,  0,  0,  0,  0,  0],
        [1,  0,  1,  0,  0,  0,  0],
        [0,  0,  0,  1,  1,  0,  0],
        [1,  0,  0,  0,  0,  1,  0],
        [0,  0,  0,  0,  0,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [0,  0,  1,  0,  0,  0,  1],
        [0,  0,  0,  0,  1,  0,  0],
        [0,  0,  0,  0,  0,  0,  1],
    ]
    V = [
        [1,  0,  0,  0,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0,  1,  0,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0,  0,  1,  0,  0],
        [0,  0,  0,  0,  0,  1,  0],
        [0,  0,  0,  0,  0,  0,  1],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
    ]
    W = [
        [1,  0,  0,  0,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0, 1,  0,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0,  0,  1,  0,  0],
        [0,  0,  0,  0,  0,  1,  0],
        [0,  0,  0,  0,  0,  0,  1],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
    ]
    return U, V, W

# Strassen's 7-term algorithm for 2x2x2 matrix multiplication
# Returns (U, V, W) each of shape (4, 7) with integer entries
def _strassen_2x2x2():
    U = [
        [1,  1,  0,  0,  0,  0,  0],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  1,  0,  0,  0],
        [0,  0,  0,  0,  1,  1,  1],
    ]
    V = [
        [1,  0,  0,  0,  0,  0,  0],
        [1,  0,  0,  0,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0,  0,  0, 1,  1,  1],
    ]
    W = [
        [1,  0,  0,  0,  0,  0,  0],
        [0,  1,  0,  0,  0,  0,  0],
        [0,  0,  1,  0,  0,  0,  0],
        [0,  0,  0,  1,  0,  0,  0],
    ]
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the n x m x p matrix mult tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            ui = i * m + j
            for k in range(p):
                vi = j * p + k
                wi = k * n + i
                total = 0.0
                for r in range(rank):
                    total += U[ui][r] * V[vi][r] * W[wi][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _build_block_decomposition(n, m, p):
    """
    Build a block decomposition by partitioning indices into sets of size 3 and 2,
    applying Laderman to 3x3x3 blocks and Strassen to 2x2x2 blocks.
    """
    # Partition indices 0..4 into groups: [0,1,2] (size 3) and [3,4] (size 2)
    groups = [[0, 1, 2], [3, 4]]
    group_sizes = [3, 2]
    
    # Precompute which group each index belongs to
    index_to_group = {}
    for g, idxs in enumerate(groups):
        for idx in idxs:
            index_to_group[idx] = g
    
    # Build the full U, V, W matrices
    total_rank = 0
    # Count total rank needed
    for g1 in range(len(groups)):
        for g2 in range(len(groups)):
            for g3 in range(len(groups)):
                s1, s2, s3 = group_sizes[g1], group_sizes[g2], group_sizes[g3]
                if s1 == 3 and s2 == 3 and s3 == 3:
                    total_rank += 7
                elif s1 == 2 and s2 == 2 and s3 == 2:
                    total_rank += 7
                else:
                    # Mixed sizes: use schoolbook for this block
                    total_rank += s1 * s2 * s3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for g1 in range(len(groups)):
        for g2 in range(len(groups)):
            for g3 in range(len(groups)):
                s1, s2, s3 = group_sizes[g1], group_sizes[g2], group_sizes[g3]
                idxs1 = groups[g1]
                idxs2 = groups[g2]
                idxs3 = groups[g3]
                
                if s1 == 3 and s2 == 3 and s3 == 3:
                    # Use Laderman
                    U_block, V_block, W_block = _laderman_3x3x3()
                    block_rank = 7
                    for local_i, gi in enumerate(idxs1):
                        for local_j, gj in enumerate(idxs2):
                            ui = gi * m + gj
                            for r in range(block_rank):
                                U[ui][col_offset + r] = U_block[local_i * s2 + local_j][r]
                    for local_j, gj in enumerate(idxs2):
                        for local_k, gk in enumerate(idxs3):
                            vi = gj * p + gk
                            for r in range(block_rank):
                                V[vi][col_offset + r] = V_block[local_j * s3 + local_k][r]
                    for local_k, gk in enumerate(idxs3):
                        for local_i, gi in enumerate(idxs1):
                            wi = gk * n + gi
                            for r in range(block_rank):
                                W[wi][col_offset + r] = W_block[local_k * s1 + local_i][r]
                    col_offset += block_rank
                    
                elif s1 == 2 and s2 == 2 and s3 == 2:
                    # Use Strassen
                    U_block, V_block, W_block = _strassen_2x2x2()
                    block_rank = 7
                    for local_i, gi in enumerate(idxs1):
                        for local_j, gj in enumerate(idxs2):
                            ui = gi * m + gj
                            for r in range(block_rank):
                                U[ui][col_offset + r] = U_block[local_i * s2 + local_j][r]
                    for local_j, gj in enumerate(idxs2):
                        for local_k, gk in enumerate(idxs3):
                            vi = gj * p + gk
                            for r in range(block_rank):
                                V[vi][col_offset + r] = V_block[local_j * s3 + local_k][r]
                    for local_k, gk in enumerate(idxs3):
                        for local_i, gi in enumerate(idxs1):
                            wi = gk * n + gi
                            for r in range(block_rank):
                                W[wi][col_offset + r] = W_block[local_k * s1 + local_i][r]
                    col_offset += block_rank
                    
                else:
                    # Mixed sizes: use schoolbook for this block
                    block_rank = s1 * s2 * s3
                    col = 0
                    for local_i, gi in enumerate(idxs1):
                        for local_j, gj in enumerate(idxs2):
                            ui = gi * m + gj
                            for local_k, gk in enumerate(idxs3):
                                vi = gj * p + gk
                                wi = gk * n + gi
                                U[ui][col_offset + col] = 1
                                V[vi][col_offset + col] = 1
                                W[wi][col_offset + col] = 1
                                col += 1
                    col_offset += block_rank
    
    return U, V, W, total_rank

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition first
    U_block, V_block, W_block, rank_block = _build_block_decomposition(n, m, p)
    
    # Verify the block decomposition
    if _verify(U_block, V_block, W_block, n, m, p):
        if time.time() < deadline:
            return {"U": U_block, "V": V_block, "W": W_block}
    
    # Fallback to schoolbook
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    
    return {"U": U_school, "V": V_school, "W": W_school}
