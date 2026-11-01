"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 2x5x6 multiplication into:
    # 1. A 2x4x6 part (rows 0-3 of the middle matrix)
    # 2. A 2x1x6 part (row 4 of the middle matrix)
    
    # Part 2: 2x1x6
    # Rank is 12 (identity tensor on 2x6 space)
    rank_part2 = 12
    
    # Part 1: 2x4x6
    # Split into two 2x2x6 blocks.
    # Each 2x2x6 is split into three 2x2x2 blocks.
    # Each 2x2x2 uses Strassen (rank 7).
    # So 2x2x6 rank is 3 * 7 = 21.
    # Two blocks: 21 + 21 = 42.
    rank_part1 = 42
    
    total_rank = rank_part1 + rank_part2
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2x2 block
    # i_offset, j_offset, k_offset
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Strassen constants for 2x2x2
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10+B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # We need to map these to the global indices.
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        # For a block starting at i_off, j_off, k_off
        # The local i is 0,1. Global i = i_off + local_i.
        # Local j is 0,1. Global j = j_off + local_j.
        # Local k is 0,1. Global k = k_off + local_k.
        
        # Strassen terms (u, v, w)
        # u is 2x2 matrix (flattened to 4 elements? No, U is n*m x rank)
        # U has shape (n*m, rank). For a 2x2 block, n=2, m=2 locally?
        # No, the global n,m,p are fixed.
        # We are filling U, V, W for the global dimensions.
        
        # The Strassen decomposition provides 7 terms.
        # Each term is a rank-1 tensor.
        # For term c:
        # U_c is a vector of length n*m.
        # V_c is a vector of length m*p.
        # W_c is a vector of length p*n.
        
        # We define the 7 Strassen terms as (U_vec, V_vec, W_vec)
        # where U_vec has non-zeros only at indices corresponding to the 2x2 A block.
        # V_vec has non-zeros only at indices corresponding to the 2x2 B block.
        # W_vec has non-zeros only at indices corresponding to the 2x2 C block.
        
        # Local indices:
        # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # Global indices:
        # U: (i_off+li, j_off+lj) -> (i_off+li)*m + (j_off+lj)
        # V: (j_off+lj, k_off+lk) -> (j_off+lj)*p + (k_off+lk)
        # W: (k_off+lk, i_off+li) -> (k_off+lk)*n + (i_off+li)
        
        # Strassen terms:
        # 1: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1 (with signs later)
        
        # It's easier to define the 7 terms explicitly.
        
        terms = [
            # (U_nonzero_indices, V_nonzero_indices, W_nonzero_indices)
            # Each is a list of (local_index, value)
            # Local U index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Local V index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Local W index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
            # M3
            ([(0, 1)], [(2, 1), (3, 1)], [(1, 1)]),
            # M4
            ([(3, 1)], [(1, 1), (2, -1)], [(0, 1), (2, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)]),
        ]
        
        for u_vals, v_vals, w_vals in terms:
            for (li_lu, val_u) in u_vals:
                gi = i_off + (li_lu // 2)
                gj = j_off + (li_lu % 2)
                U[gi * m + gj][col] += val_u
            for (li_lv, val_v) in v_vals:
                gj = j_off + (li_lv // 2)
                gk = k_off + (li_lv % 2)
                V[gj * p + gk][col] += val_v
            for (li_lw, val_w) in w_vals:
                gk = k_off + (li_lw // 2)
                gi = i_off + (li_lw % 2)
                W[gk * n + gi][col] += val_w
            col += 1

    # Part 1: 2x4x6
    # Block 1: i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 0)
    # Block 2: i=0,1; j=0,1; k=2,3
    add_strassen(0, 0, 2)
    # Block 3: i=0,1; j=0,1; k=4,5
    add_strassen(0, 0, 4)
    # Block 4: i=0,1; j=2,3; k=0,1
    add_strassen(0, 2, 0)
    # Block 5: i=0,1; j=2,3; k=2,3
    add_strassen(0, 2, 2)
    # Block 6: i=0,1; j=2,3; k=4,5
    add_strassen(0, 2, 4)
    
    # Part 2: 2x1x6
    # j=4. i=0,1. k=0..5.
    # Rank 12.
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
