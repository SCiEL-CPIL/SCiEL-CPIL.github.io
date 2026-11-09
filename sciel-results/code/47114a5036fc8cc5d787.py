"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20
    
    # Rank 65: Recursive Strassen 4x4x4 (rank 49) + Schoolbook 4x4x1 (rank 16)
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Part 2: Schoolbook for column 4 (k=4) ---
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4, i=row of C
    
    # --- Part 1: Recursive Strassen 4x4x4 ---
    # Level 1: 7 Strassen products, each is a 2x2x2 multiplication
    # Level 2: Each 2x2x2 uses Strassen (7 scalar products)
    
    def a_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j
    
    def b_idx(bi, u, v):
        j = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return j * p + k
    
    def c_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return k * n + i
    
    # Outer Strassen definitions
    A_side = [
        [(0, 1.0), (3, 1.0)],
        [(2, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (0, -1.0)],
        [(1, 1.0), (3, -1.0)],
    ]
    
    B_side = [
        [(0, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(1, 1.0), (3, -1.0)],
        [(2, 1.0), (0, -1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (3, 1.0)],
    ]
    
    out_coeff = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0},
    ]
    
    inner_A_side = [
        [(0, 1.0), (3, 1.0)],
        [(2, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (0, -1.0)],
        [(1, 1.0), (3, -1.0)],
    ]
    
    inner_B_side = [
        [(0, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(1, 1.0), (3, -1.0)],
        [(2, 1.0), (0, -1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (3, 1.0)],
    ]
    
    inner_out_coeff = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0},
    ]
    
    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk
            
            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    U[a_idx(bi, u_local, v_local), col] += c1 * c2
            
            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    V[b_idx(bi, u_local, v_local), col] += c1 * c2
            
            for (bi_out, c_inner) in inner_out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    u_local = bi_out // 2
                    v_local = bi_out % 2
                    W[c_idx(bi_outer, u_local, v_local), col] += c_inner * c_outer
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
