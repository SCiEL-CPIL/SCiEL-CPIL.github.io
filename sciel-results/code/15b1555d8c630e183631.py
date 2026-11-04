import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy set-packing algorithm to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 105 - num_blocks.
    
    Improvements:
    1. Greedy Set Packing with conflict minimization heuristic.
    2. Multiple random restarts with tie-breaking.
    3. Efficient bitmask operations for disjointness.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Total number of (i,j,k) triples
    total_triples = n * m * p  # 105

    # Precompute all possible 2x2x2 blocks
    all_blocks = []  # list of (params, bitmask)
    
    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k

    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask))

    num_blocks_total = len(all_blocks)
    
    # Precompute conflict counts? No, dynamic.
    # We will use a greedy approach that dynamically removes conflicting blocks.

    best_num_blocks = 0
    best_packing = []

    def greedy_set_packing():
        """
        Greedy set packing:
        1. Start with all blocks.
        2. Repeatedly pick the block with the fewest conflicts (conflict = sharing a triple with another available block).
        3. If tie, pick randomly.
        4. Remove the picked block and all blocks that conflict with it.
        5. Repeat until empty.
        """
        available_indices = set(range(num_blocks_total))
        packing = []
        
        while available_indices:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate conflicts for each available block
            # Conflict count = number of OTHER available blocks that share a triple
            min_conflict = float('inf')
            candidates = []
            
            # Optimization: Instead of O(N^2) every step, we can just check a few?
            # With 630 blocks, O(N^2) is ~400k ops per step. 
            # Max steps ~15. Total ~6M ops. Fast enough.
            
            # To speed up, we can precompute the list of conflicting indices for each block
            # But the "available" set changes.
            
            for idx in available_indices:
                bm = all_blocks[idx][1]
                conflicts = 0
                for other_idx in available_indices:
                    if other_idx == idx:
                        continue
                    if (bm & all_blocks[other_idx][1]) != 0:
                        conflicts += 1
                
                if conflicts < min_conflict:
                    min_conflict = conflicts
                    candidates = [idx]
                elif conflicts == min_conflict:
                    candidates.append(idx)
            
            # Pick one randomly from candidates
            if not candidates:
                break
            chosen_idx = random.choice(candidates)
            packing.append(all_blocks[chosen_idx][0])
            
            # Remove chosen and all conflicting blocks
            bm_chosen = all_blocks[chosen_idx][1]
            to_remove = set()
            to_remove.add(chosen_idx)
            for other_idx in available_indices:
                if other_idx != chosen_idx and (bm_chosen & all_blocks[other_idx][1]) != 0:
                    to_remove.add(other_idx)
            
            available_indices -= to_remove

        return packing

    # Run multiple trials
    num_trials = 50
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        packing = greedy_set_packing()
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            
            # Early exit if we hit a high number
            if best_num_blocks >= 12:
                break

    # Construct U, V, W based on the best packing found
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
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Convert to matrix format
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
# EVOLVE-BLOCK-END
