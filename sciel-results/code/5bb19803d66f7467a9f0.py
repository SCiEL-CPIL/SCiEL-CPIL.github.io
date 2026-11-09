"""5x5x5 matrix multiplication: rank-110 decomposition via 4+1 block Strassen."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 10.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-110 exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 4+1 block partition. For the (4,4,4) block product, applies
    recursive Strassen (rank 49 instead of schoolbook 64), saving 15 ranks.
    The remaining 7 block products use schoolbook.
    Total: 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []  # list of (u_dict, v_dict, w_dict)
    
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
    
    def schoolbook_block(i_list, j_list, k_list):
        """Standard rank-1 decomposition for each (i,j,k) in the block."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # Strassen 2x2x2 terms (shared)
    _STRASSEN_2x2 = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,1): 1, (0,0): -1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    def strassen_4x4x4(off_i, off_j, off_k):
        """
        Add 49 rank-1 terms implementing recursive Strassen for 4x4x4
        matrix multiplication at global offsets (off_i, off_j, off_k).
        
        Splits 4x4x4 into 2x2x2 sub-blocks and applies Strassen recursively.
        """
        for top_a_c, top_b_c, top_c_c in _STRASSEN_2x2:
            for bot_a_c, bot_b_c, bot_c_c in _STRASSEN_2x2:
                u_entries = []
                v_entries = []
                w_entries = []
                for (qa, qb), va in top_a_c.items():
                    for (li, lj), vb in bot_a_c.items():
                        gi = off_i + 2*qa + li
                        gj = off_j + 2*qb + lj
                        u_entries.append((gi * m + gj, va * vb))
                for (qc, qd), va in top_b_c.items():
                    for (lj, lk), vb in bot_b_c.items():
                        gj = off_j + 2*qc + lj
                        gk = off_k + 2*qd + lk
                        v_entries.append((gj * p + gk, va * vb))
                for (qe, qf), va in top_c_c.items():
                    for (li, lk), vb in bot_c_c.items():
                        gi = off_i + 2*qe + li
                        gk = off_k + 2*qf + lk
                        w_entries.append((gk * n + gi, va * vb))
                add_column(u_entries, v_entries, w_entries)
    
    # 4+1 block partition
    blocks = [[0, 1, 2, 3], [4]]
    
    # Iterate over all 2x2x2 = 8 block products
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 4 and len(j_list) == 4 and len(k_list) == 4:
                    strassen_4x4x4(i_list[0], j_list[0], k_list[0])
                else:
                    schoolbook_block(i_list, j_list, k_list)
    
    rank = len(columns)
    
    # Build output matrices
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
