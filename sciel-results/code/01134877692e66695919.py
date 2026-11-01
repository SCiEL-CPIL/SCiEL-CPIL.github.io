"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank from 50 to 46.
Attempts to find a lower rank decomposition via search within the Strassen family.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)
        
        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)
        
        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)
        
        # M1 = (a+d)(e+h)
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        # M2 = (c+d)e
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        # M3 = a(f-h)
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        # M4 = d(g-e)
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        # M5 = (a+b)h
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        # M6 = (c-a)(e+f)
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        # M7 = (b-d)(g+h)
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])
    
    # Partition 5 = 2+2+1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)
    
    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)
    
    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)
    
    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val
    
    return U, V, W, total_rank

def _try_alternative_partitions():
    """
    Try alternative partitionings of the inner dimension 5 to see if any
    yield a lower rank than the 2+2+1 partition.
    """
    n, m, p = 2, 5, 5
    
    # Try all compositions of 5 into parts of size 1 or 2
    # (since we can only do Strassen on 2x2 blocks)
    def compositions(total, max_part=2):
        if total == 0:
            yield []
            return
        for part in range(1, min(max_part, total) + 1):
            for rest in compositions(total - part, max_part):
                yield [part] + rest
    
    best_rank = float('inf')
    best_partition = None
    
    for comp in compositions(5):
        # For each composition, compute the rank
        # Each part of size 2: use Strassen for the 2x2 x 2x2 sub-blocks
        # Each part of size 1: schoolbook
        # 
        # For a part of size s (where s is 1 or 2):
        #   A block is 2xs, B block is sx5
        #   If s=2: partition B's 5 columns into 2+2+1
        #     Strassen 2x2x2: rank 7 (for cols 0-1)
        #     Strassen 2x2x2: rank 7 (for cols 2-3)
        #     Schoolbook 2x2x2x1: rank 4 (for col 4)
        #     Total: 18
        #   If s=1:
        #     Schoolbook 2x1x1x5: rank 10
        #
        rank = 0
        for s in comp:
            if s == 2:
                rank += 18
            elif s == 1:
                rank += 10
        if rank < best_rank:
            best_rank = rank
            best_partition = comp
    
    return best_rank, best_partition

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # First, try to find a better partition
    best_rank, best_partition = _try_alternative_partitions()
    
    # The 2+2+1 partition gives rank 46
    # Check if any other partition is better
    if best_rank < 46 and best_partition:
        # We'd need to implement the alternative partition
        # For now, fall back to the known-good 2+2+1 partition
        pass
    
    # Build the standard rank-46 decomposition
    U, V, W, total_rank = _build_strassen_decomposition()
    
    # Time guard
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
