"""Matrix multiplication 3x4 x 4x6 with reduced rank via block partitioning."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Block partitioning strategy:
    # Split middle dim (4) into 2+2
    # For each (3x2)*(2x6) block, split rows into 1+2, cols into 2+2+2
    # (1x2)*(2x2): rank 2, (2x2)*(2x2): rank 7 (Strassen)
    # Total: 2 * (3*2 + 3*7) = 2 * 27 = 54
    
    # Precompute block decompositions
    # (1x2)*(2x2) -> rank 2
    U_122 = [[1, 0],
             [0, 1]]
    V_122 = [[1, 0],
             [0, 1],
             [1, 0],
             [0, 1]]
    W_122 = [[1, 0],
             [0, 1]]
    
    # (2x2)*(2x2) Strassen -> rank 7
    U_222 = [[1, 0, 1, 0, 1, -1, 0],
             [0, 0, 0, 0, 1, 0, 1],
             [0, 1, 0, 0, 0, 1, 0],
             [1, 1, 0, 1, 0, 0, -1]]
    V_222 = [[1, 1, 0, -1, 0, 1, 0],
             [1, 0, 1, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 1],
             [0, 0, -1, 0, 1, 0, 1]]
    W_222 = [[1, 0, 0, 1, -1, 1, 0],
             [0, 0, 1, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [1, -1, 1, 0, 0, 0, 1]]
    
    # Total rank
    total_rank = 2 * (3 * 2 + 3 * 7)  # = 54
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 54
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 54
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 54
    
    col_offset = 0
    for main_block in range(2):
        j_off = main_block * 2  # 0 or 2
        for row_part in range(2):
            if row_part == 0:
                i_start = 0
                num_i = 1
            else:
                i_start = 1
                num_i = 2
            for col_part in range(3):
                k_start = col_part * 2  # 0, 2, 4
                k_end = k_start + 2
                
                if num_i == 1:
                    Ub, Vb, Wb, r = U_122, V_122, W_122, 2
                    u_map = [(i_start, j_off), (i_start, j_off + 1)]
                    v_map = [(j_off, k_start), (j_off, k_start + 1),
                             (j_off + 1, k_start), (j_off + 1, k_start + 1)]
                    w_map = [(i_start, k_start), (i_start, k_start + 1)]
                else:
                    Ub, Vb, Wb, r = U_222, V_222, W_222, 7
                    u_map = [(i_start, j_off), (i_start, j_off + 1),
                             (i_start + 1, j_off), (i_start + 1, j_off + 1)]
                    v_map = [(j_off, k_start), (j_off, k_start + 1),
                             (j_off + 1, k_start), (j_off + 1, k_start + 1)]
                    w_map = [(i_start, k_start), (i_start, k_start + 1),
                             (i_start + 1, k_start), (i_start + 1, k_start + 1)]
                
                for c in range(r):
                    for l in range(len(u_map)):
                        val = Ub[l][c]
                        if val != 0:
                            gi, gj = u_map[l]
                            U[gi * m + gj][col_offset + c] += val
                    for l in range(len(v_map)):
                        val = Vb[l][c]
                        if val != 0:
                            gj, gk = v_map[l]
                            V[gj * p + gk][col_offset + c] += val
                    for l in range(len(w_map)):
                        val = Wb[l][c]
                        if val != 0:
                            gi, gk = w_map[l]
                            W[gk * n + gi][col_offset + c] += val
                
                col_offset += r
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
