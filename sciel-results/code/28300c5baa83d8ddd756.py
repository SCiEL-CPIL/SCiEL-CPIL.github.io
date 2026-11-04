import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# Precomputed ranks for small blocks
BLOCK_RANKS = {
    (1, 1, 1): 1,
    (1, 1, 2): 2,
    (1, 1, 3): 3,
    (1, 2, 1): 2,
    (1, 2, 2): 4,
    (1, 2, 3): 6,
    (1, 3, 1): 3,
    (1, 3, 2): 6,
    (1, 3, 3): 9,
    (2, 1, 1): 2,
    (2, 1, 2): 4,
    (2, 1, 3): 6,
    (2, 2, 1): 4,
    (2, 2, 2): 7,
    (2, 2, 3): 12,
    (2, 3, 1): 6,
    (2, 3, 2): 12,
    (2, 3, 3): 18,
    (3, 1, 1): 3,
    (3, 1, 2): 6,
    (3, 1, 3): 9,
    (3, 2, 1): 6,
    (3, 2, 2): 11,
    (3, 2, 3): 17,
    (3, 3, 1): 9,
    (3, 3, 2): 18,
    (3, 3, 3): 23
}

def get_block_rank(ni, mj, pk):
    key = (ni, mj, pk)
    if key in BLOCK_RANKS:
        return BLOCK_RANKS[key]
    return ni * mj * pk

def strassen_factors():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def schoolbook_factors(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def build_decomposition(n_sizes, m_sizes, p_sizes):
    n, m, p = 3, 5, 6
    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                total_rank += get_block_rank(ni, mj, pk)
                
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    n_offsets = np.cumsum([0] + n_sizes)[:-1]
    m_offsets = np.cumsum([0] + m_sizes)[:-1]
    p_offsets = np.cumsum([0] + p_sizes)[:-1]
    
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_blk, V_blk, W_blk = strassen_factors()
                else:
                    U_blk, V_blk, W_blk = schoolbook_factors(ni, mj, pk)
                
                r = U_blk.shape[1]
                
                for li in range(ni):
                    for lj in range(mj):
                        full_row_u = (i_start + li) * m + (j_start + lj)
                        U[full_row_u, col_offset:col_offset + r] = U_blk[li * mj + lj, :]
                        
                for lj in range(mj):
                    for lk in range(pk):
                        full_row_v = (j_start + lj) * p + (k_start + lk)
                        V[full_row_v, col_offset:col_offset + r] = V_blk[lj * pk + lk, :]
                        
                for lk in range(pk):
                    for li in range(ni):
                        full_row_w = (k_start + lk) * n + (i_start + li)
                        W[full_row_w, col_offset:col_offset + r] = W_blk[lk * ni + li, :]
                        
                col_offset += r
                
    return U, V, W, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)
    return bool(np.allclose(vals, 1.0, atol=atol))

def find_best_partition():
    best_rank = 999
    best_partition = None
    # Try partitions of 3, 5, 6
    # 3: [3], [1,2], [2,1], [1,1,1]
    # 5: [5], [1,4], [4,1], [2,3], [3,2], [1,1,3], [1,3,1], [3,1,1], [2,1,2], [1,2,2], [2,2,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1]
    # 6: similar
    
    n_parts = [[3], [1,2], [2,1], [1,1,1]]
    m_parts = [[5], [1,4], [4,1], [2,3], [3,2], [1,1,3], [1,3,1], [3,1,1], [2,1,2], [1,2,2], [2,2,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1], [2,2,1], [1,2,2], [2,1,2], [3,2], [2,3], [1,4], [4,1], [5]]
    p_parts = [[6], [1,5], [5,1], [2,4], [4,2], [3,3], [1,1,4], [1,4,1], [4,1,1], [2,2,2], [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1], [1,1,1,3], [1,1,3,1], [1,3,1,1], [3,1,1,1], [1,1,2,2], [1,2,1,2], [1,2,2,1], [2,1,1,2], [2,1,2,1], [2,2,1,1], [1,1,1,1,2], [1,1,1,2,1], [1,1,2,1,1], [1,2,1,1,1], [2,1,1,1,1], [1,1,1,1,1,1]]
    
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                if sum(np_) != 3 or sum(mp_) != 5 or sum(pp_) != 6:
                    continue
                rank = 0
                for ni in np_:
                    for mj in mp_:
                        for pk in pp_:
                            rank += get_block_rank(ni, mj, pk)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (np_, mp_, pp_)
                    
    return best_partition, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Find best partition
    best_partition, best_rank = find_best_partition()
    
    if best_partition:
        U, V, W, rank = build_decomposition(best_partition[0], best_partition[1], best_partition[2])
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
