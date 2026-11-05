import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_2x2(off_i, off_j, off_k):
        # M1 = (A00+A11)(B00+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M2 = (A10+A11)B00
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = -1
        add_col(u, v, w)

        # M3 = A00(B01-B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M4 = A11(B10-B00)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = -1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M5 = (A00+A01)B11
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+0)] = -1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+1)*5+(off_j+1)] = -1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

    # Strategy: Try different block partitions to find better rank
    # Option 1: Current 2+2+1 partition with Strassen (rank 117)
    # Option 2: Try 3+2 partition with optimized cross terms
    
    # Let's try a systematic search over block partitions
    # For each partition, compute the rank using known sub-problem ranks
    
    def compute_rank_for_partition(partition):
        """
        partition: list of tuples (i_block, j_block, k_block) where each is a tuple of indices
        Returns estimated rank
        """
        total_rank = 0
        for i_blk in partition:
            for j_blk in partition:
                for k_blk in partition:
                    ni = len(i_blk)
                    nj = len(j_blk)
                    nk = len(k_blk)
                    # Use Strassen for 2x2x2, schoolbook otherwise
                    if ni == 2 and nj == 2 and nk == 2:
                        total_rank += 7
                    else:
                        total_rank += ni * nj * nk
        return total_rank
    
    # Try various partitions
    partitions = [
        # 2+2+1 (current)
        ([0,1], [2,3], [4]),
        # 3+2
        ([0,1,2], [3,4]),
        # 1+2+2
        ([0], [1,2], [3,4]),
        # 2+1+2
        ([0,1], [2], [3,4]),
        # 1+1+3
        ([0], [1], [2,3,4]),
        # 3+1+1
        ([0,1,2], [3], [4]),
    ]
    
    best_rank = float('inf')
    best_partition = None
    
    for part in partitions:
        # Convert to tuple of tuples for compute_rank_for_partition
        part_tuples = tuple(tuple(x) for x in part)
        rank = compute_rank_for_partition(part_tuples)
        if rank < best_rank:
            best_rank = rank
            best_partition = part_tuples
    
    # Use the best partition found
    if best_partition is None:
        best_partition = ((0,1), (2,3), (4))
    
    # Build the decomposition using the best partition
    covered = set()
    
    for i_blk in best_partition:
        for j_blk in best_partition:
            for k_blk in best_partition:
                i0 = i_blk[0]
                j0 = j_blk[0]
                k0 = k_blk[0]
                ni = len(i_blk)
                nj = len(j_blk)
                nk = len(k_blk)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen for this 2x2x2 block
                    add_strassen_2x2(i0, j0, k0)
                    for di in range(ni):
                        for dj in range(nj):
                            for dk in range(nk):
                                covered.add((i0+di, j0+dj, k0+dk))
                else:
                    # Use schoolbook for this block
                    for di in range(ni):
                        for dj in range(nj):
                            for dk in range(nk):
                                i = i0 + di
                                j = j0 + dj
                                k = k0 + dk
                                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                                u_vec[i*5 + j] = 1
                                v_vec[j*5 + k] = 1
                                w_vec[k*5 + i] = 1
                                add_col(u_vec, v_vec, w_vec)
                                covered.add((i, j, k))
    
    # Add any uncovered terms with schoolbook
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if (i, j, k) not in covered:
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[i*5 + j] = 1
                    v_vec[j*5 + k] = 1
                    w_vec[k*5 + i] = 1
                    add_col(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    if time.time() > deadline - 5:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
