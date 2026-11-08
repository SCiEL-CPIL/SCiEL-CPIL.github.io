import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Strategy:
    1. Identify all possible 2x2x2 Strassen blocks.
    2. Find the maximum set of disjoint blocks (Maximum Independent Set on conflict graph).
    3. Use Strassen decomposition for these blocks.
    4. Use schoolbook decomposition for the remaining indices.
    
    Rank = 7 * num_blocks + (total_indices - 8 * num_blocks) = total_indices - num_blocks.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    total_indices = n * m * p  # 105

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # 1. Generate all possible 2x2x2 blocks
    # A block is defined by (i1, i2, j1, j2, k1, k2) with i1<i2, j1<j2, k1<k2
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # Indices covered
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))
    
    num_blocks = len(all_blocks)
    
    # 2. Build conflict graph
    # conflicts[i] is a bitmask or set of indices j such that block i and block j overlap
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # 3. Find Maximum Independent Set (MIS)
    # Since num_blocks is small (24), we can use a simple branch and bound or even brute force with pruning.
    # We will use a recursive backtracking with bitmasks for speed.
    
    # Convert conflicts to bitmasks for faster intersection checks
    # mask[i] has bit j set if i and j conflict
    conflict_masks = [0] * num_blocks
    for i in range(num_blocks):
        for j in conflicts[i]:
            conflict_masks[i] |= (1 << j)
    
    best_size = 0
    best_set = []
    
    # Order blocks by degree (descending) for better pruning? 
    # Or just use the natural order. With 24 nodes, it's fast enough.
    
    def backtrack(idx, current_size, current_set, available_mask):
        nonlocal best_size, best_set
        
        # Pruning: if current_size + remaining_possible <= best_size, return
        # remaining_possible is popcount(available_mask)
        if current_size + bin(available_mask).count('1') <= best_size:
            return
            
        if idx >= num_blocks:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Option 1: Include block idx
        if (available_mask >> idx) & 1:
            # Check if idx is in available_mask
            # If we include idx, we remove idx and all its conflicts from available_mask
            new_available = available_mask & ~(1 << idx) & ~conflict_masks[idx]
            current_set.append(idx)
            backtrack(idx + 1, current_size + 1, current_set, new_available)
            current_set.pop()
            
        # Option 2: Exclude block idx
        backtrack(idx + 1, current_size, current_set, available_mask)
        
    initial_mask = (1 << num_blocks) - 1
    backtrack(0, 0, [], initial_mask)
    
    # 4. Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # Standard Strassen 2x2x2 decomposition
        # M1 = (a11+a22)*(b11+b22) -> c11 += M1, c22 += M1
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11 -> c21 += M2, c22 -= M2
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22) -> c12 += M3, c22 += M3
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11) -> c11 += M4, c21 += M4
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22 -> c11 -= M5, c12 += M5
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12) -> c22 += M6
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22) -> c11 += M7
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for idx in best_set:
        block_params, _ = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for truly uncovered indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format (list of lists)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
