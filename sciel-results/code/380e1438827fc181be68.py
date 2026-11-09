"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on all 2x2x2 blocks (rank 117)."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 2+2+1 block partition (3 blocks of sizes 2, 2, 1).
    All 2x2x2 sub-blocks use Strassen (rank 7 each).
    1x1x1 sub-blocks use schoolbook (rank 1 each).
    
    Number of 2x2x2 blocks: C(3,2)^3 = 3^3 = 27... wait, let me count properly.
    Blocks: [0,1], [2,3], [4]
    For each triple (bi, bj, bk) where bi,bj,bk in {0,1,2}:
      - If all three are size-2 blocks: Strassen 2x2x2 (rank 7)
      - If any is size-1: schoolbook
    
    Size-2 blocks: indices 0,1 (block 0) and 2,3 (block 1)
    Size-1 block: index 4 (block 2)
    
    Triples where all three dimensions use size-2 blocks:
      bi in {0,1}, bj in {0,1}, bk in {0,1} -> 8 blocks of rank 7 = 56
    
    Remaining blocks (at least one dimension is size-1):
      Total 27 - 8 = 19 blocks, each schoolbook.
      For each such block, rank = |i_list| * |j_list| * |k_list|
      
      Let me enumerate:
      bi=2 (size 1), bj in {0,1}, bk in {0,1}: 4 blocks, each 1*2*2=4 -> 16
      bi in {0,1}, bj=2, bk in {0,1}: 4 blocks, each 2*1*2=4 -> 16
      bi in {0,1}, bj in {0,1}, bk=2: 4 blocks, each 2*2*1=4 -> 16
      bi=2, bj=2, bk in {0,1}: 2 blocks, each 1*1*2=2 -> 4
      bi=2, bj in {0,1}, bk=2: 2 blocks, each 1*2*1=2 -> 4
      bi in {0,1}, bj=2, bk=2: 2 blocks, each 2*1*1=2 -> 4
      bi=2, bj=2, bk=2: 1 block, 1*1*1=1 -> 1
      
      Total schoolbook: 16+16+16+4+4+4+1 = 61
      
    Total rank: 56 + 61 = 117
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Block index groups: 2+2+1 partition
    blocks = [[0, 1], [2, 3], [4]]
    
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
                    
    # Standard Strassen 2x2x2 terms for local indices 0,1
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(offset_i, offset_j, offset_k):
        """
        Adds 7 columns for a Strassen 2x2x2 decomposition.
        offset_i, offset_j, offset_k: starting global indices for the 2x2x2 block.
        """
        I_map = {0: offset_i, 1: offset_i + 1}
        J_map = {0: offset_j, 1: offset_j + 1}
        K_map = {0: offset_k, 1: offset_k + 1}
        
        for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
            u_entries = []
            v_entries = []
            w_entries = []
            
            for (li, lj), val in u_coeffs.items():
                gi = I_map[li]
                gj = J_map[lj]
                u_entries.append((gi * m + gj, val))
                
            for (lj, lk), val in v_coeffs.items():
                gj = J_map[lj]
                gk = K_map[lk]
                v_entries.append((gj * p + gk, val))
                
            for (li, lk), val in w_coeffs.items():
                gi = I_map[li]
                gk = K_map[lk]
                w_entries.append((gk * n + gi, val))
                
            add_column(u_entries, v_entries, w_entries)

    # Build all 27 blocks
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                if _deadline_exceeded():
                    break
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    add_strassen_2x2x2(i_list[0], j_list[0], k_list[0])
                else:
                    schoolbook_block(i_list, j_list, k_list)
            if _deadline_exceeded():
                break
        if _deadline_exceeded():
            break
    
    rank = len(columns)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
