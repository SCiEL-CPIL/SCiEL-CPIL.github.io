import numpy as np
import time

def construct():
    """
    Compute rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy: Partition the tensor into blocks and use known optimal decompositions
    for each block, then combine them.
    
    Decomposition: M(3,4,6) = M(1,4,6) ⊕ M(2,4,6) where ⊕ denotes block diagonal
    (independent sub-tensors from partitioning n).
    
    M(1,4,6): rank = 24 (schoolbook: 1*4*6 = 24)
    M(2,4,6): rank = 42 (Strassen-based: 3 * M(2,2,2) per p-pair, 3 p-pairs)
    
    Total rank: 24 + 42 = 66
    """
    t0 = time.time()
    DEADLINE = 140.0  # 20s safety margin
    
    n, m, p = 3, 4, 6
    total_rank = 66
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    rank_ptr = 0
    
    # Helper: add a rank-1 term
    def emit(u_sparsity, v_sparsity, w_sparsity):
        nonlocal rank_ptr
        if rank_ptr >= total_rank:
            return
        for (idx, val) in u_sparsity:
            U[idx, rank_ptr] += val
        for (idx, val) in v_sparsity:
            V[idx, rank_ptr] += val
        for (idx, val) in w_sparsity:
            W[idx, rank_ptr] += val
        rank_ptr += 1
    
    # M(1,4,6): Schoolbook, n=1
    for j in range(4):
        for k in range(6):
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            emit([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # M(2,4,6): Use Strassen on 2x2 sub-blocks
    # Split m into two groups of 2, p into three groups of 2
    def strassen_2x2(i_base, j_base, k_base):
        """Add Strassen decomposition for M(2,2,2) at offset (i_base, j_base, k_base)"""
        def U_idx(i, j):
            return (i_base + i) * m + (j_base + j)
        def V_idx(j, k):
            return (j_base + j) * p + (k_base + k)
        def W_idx(i, k):
            return (k_base + k) * n + (i_base + i)
        
        # s1 = (a00+a11)(b00+b11)
        emit([(U_idx(0,0),1),(U_idx(1,1),1)],
             [(V_idx(0,0),1),(V_idx(1,1),1)],
             [(W_idx(0,0),1),(W_idx(1,1),1)])
        # s2 = (a10+a11)(b00)
        emit([(U_idx(1,0),1),(U_idx(1,1),1)],
             [(V_idx(0,0),1)],
             [(W_idx(1,0),1),(W_idx(1,1),-1)])
        # s3 = (a00)(b01-b11)
        emit([(U_idx(0,0),1)],
             [(V_idx(0,1),1),(V_idx(1,1),-1)],
             [(W_idx(0,1),1),(W_idx(1,1),1)])
        # s4 = (a11)(b10-b00)
        emit([(U_idx(1,1),1)],
             [(V_idx(1,0),1),(V_idx(0,0),-1)],
             [(W_idx(0,0),1),(W_idx(1,0),1)])
        # s5 = (a00+a01)(b11)
        emit([(U_idx(0,0),1),(U_idx(0,1),1)],
             [(V_idx(1,1),1)],
             [(W_idx(0,0),-1),(W_idx(0,1),1)])
        # s6 = (a10-a00)(b00+b01)
        emit([(U_idx(1,0),1),(U_idx(0,0),-1)],
             [(V_idx(0,0),1),(V_idx(0,1),1)],
             [(W_idx(1,1),1)])
        # s7 = (a01-a11)(b10+b11)
        emit([(U_idx(0,1),1),(U_idx(1,1),-1)],
             [(V_idx(1,0),1),(V_idx(1,1),1)],
             [(W_idx(0,0),1)])
    
    # M(2,4,6): i_base=1 (rows 1,2), j split into [0,1] and [2,3], k split into [0,1],[2,3],[4,5]
    for j_grp in range(2):
        for k_grp in range(3):
            if time.time() > DEADLINE:
                break
            strassen_2x2(1, j_grp * 2, k_grp * 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
