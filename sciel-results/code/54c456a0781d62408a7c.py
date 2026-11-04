import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 solution using Strassen decomposition for 2x2x2 sub-blocks
    rank = 56
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition coefficients
    # U corresponds to A (i,j), V to B (j,k), W to C (k,i)
    # The tensor entry T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        """
        Add a Strassen decomposition for a 2x2x2 block.
        i_base: row offset in A (and C)
        j_base: col offset in A (and row offset in B)
        k_base: col offset in B (and row offset in C)
        """
        for r in range(7):
            cr = base_rank + r
            
            # Update U (A part)
            # Local indices (li, lj) map to global (i_base+li, j_base+lj)
            # Global index in U: (i_base+li)*m + (j_base+lj)
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][cr] += val
            
            # Update V (B part)
            # Local indices (lj, lk) map to global (j_base+lj, k_base+lk)
            # Global index in V: (j_base+lj)*p + (k_base+lk)
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][cr] += val
            
            # Update W (C part)
            # Local indices (lk, li) map to global (k_base+lk, i_base+li)
            # Global index in W: (k_base+lk)*n + (i_base+li)
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][cr] += val

    # Partition A (2x4) into two 2x2 blocks horizontally:
    # A0: rows 0-1, cols 0-1 (i_base=0, j_base=0)
    # A1: rows 0-1, cols 2-3 (i_base=0, j_base=2)
    
    # Partition B (4x8) into two 2x8 blocks vertically:
    # B_top: rows 0-1, cols 0-7
    # B_bottom: rows 2-3, cols 0-7
    
    # B_top is partitioned into four 2x2 blocks horizontally:
    # Bt0: cols 0-1 (k_base=0)
    # Bt1: cols 2-3 (k_base=2)
    # Bt2: cols 4-5 (k_base=4)
    # Bt3: cols 6-7 (k_base=6)
    
    # B_bottom is partitioned into four 2x2 blocks horizontally:
    # Bb0: cols 0-1 (k_base=0)
    # Bb1: cols 2-3 (k_base=2)
    # Bb2: cols 4-5 (k_base=4)
    # Bb3: cols 6-7 (k_base=6)
    
    # C = A0 * B_top + A1 * B_bottom
    # A0 * B_top = A0*Bt0 + A0*Bt1 + A0*Bt2 + A0*Bt3
    # A1 * B_bottom = A1*Bb0 + A1*Bb1 + A1*Bb2 + A1*Bb3
    
    # Block 1: A0 * Bt0
    add_block(0, 0, 0, 0)
    
    # Block 2: A0 * Bt1
    add_block(7, 0, 0, 2)
    
    # Block 3: A0 * Bt2
    add_block(14, 0, 0, 4)
    
    # Block 4: A0 * Bt3
    add_block(21, 0, 0, 6)
    
    # Block 5: A1 * Bb0
    add_block(28, 0, 2, 0)
    
    # Block 6: A1 * Bb1
    add_block(35, 0, 2, 2)
    
    # Block 7: A1 * Bb2
    add_block(42, 0, 2, 4)
    
    # Block 8: A1 * Bb3
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
