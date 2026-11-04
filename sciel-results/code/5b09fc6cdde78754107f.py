import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    Strategy: Maximize the number of disjoint 2x2x2 Strassen blocks via
    randomized greedy + local search. Each block reduces rank by 1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible blocks with bitmasks
    all_blocks = []
    i_pairs = [(i1, i2) for i1 in range(n) for i2 in range(i1+1, n)]
    j_pairs = [(j1, j2) for j1 in range(m) for j2 in range(j1+1, m)]
    k_pairs = [(k1, k2) for k1 in range(p) for k2 in range(k1+1, p)]
    
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                mask = 0
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            cell_idx = i * m * p + j * p + k
                            mask |= (1 << cell_idx)
                all_blocks.append(((i1, i2, j1, j2, k1, k2), mask))
    
    num_blocks = len(all_blocks)
    
    # Randomized greedy search
    best_count = 0
    best_packing = []
    
    num_trials = 3000
    
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
            
        random.shuffle(all_blocks)
        current_mask = 0
        current_packing = []
        
        for block_params, mask in all_blocks:
            if (current_mask & mask) == 0:
                current_mask |= mask
                current_packing.append(block_params)
                
        if len(current_packing) > best_count:
            best_count = len(current_packing)
            best_packing = current_packing[:]
            
            if best_count >= 13:
                break

    # Local search refinement
    if best_count < 13 and time.time() < deadline - 5.0:
        # Build mask lookup
        block_masks = {i: all_blocks[i][1] for i in range(num_blocks)}
        block_params_list = [all_blocks[i][0] for i in range(num_blocks)]
        
        # Find which block indices are in best_packing
        packing_set = set()
        for bp in best_packing:
            for idx in range(num_blocks):
                if all_blocks[idx][0] == bp:
                    packing_set.add(idx)
                    break
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_packing_indices = list(packing_set)
            current_count = len(current_packing_indices)
            
            # Try to add each non-packing block
            for idx in range(num_blocks):
                if idx in packing_set:
                    continue
                mask = block_masks[idx]
                
                # Find conflicts
                conflicts = [ci for ci in current_packing_indices if (block_masks[ci] & mask) != 0]
                
                # If we remove conflicts and add this block, do we improve?
                if len(conflicts) == 0:
                    # Can add directly
                    packing_set.add(idx)
                    current_count += 1
                    improved = True
                elif len(conflicts) == 1:
                    # Remove 1, add 1: no improvement
                    pass
                elif len(conflicts) >= 2:
                    # Remove k, add 1: only improves if k > 1, but we're adding 1
                    # So we need to remove at most 0 to improve, or this is neutral
                    pass
            
            # Try pairwise swaps: remove 1 block, add 2
            for ci in list(packing_set):
                if time.time() > deadline - 5.0:
                    break
                removed_mask = block_masks[ci]
                # Temporarily remove
                packing_set.remove(ci)
                base_mask = 0
                for ci2 in packing_set:
                    base_mask |= block_masks[ci2]
                
                # Try adding two blocks that don't conflict with each other or existing
                added = 0
                for idx1 in range(num_blocks):
                    if idx1 in packing_set or idx1 == ci:
                        continue
                    m1 = block_masks[idx1]
                    if (base_mask & m1) != 0:
                        continue
                    # Try adding idx1
                    new_mask = base_mask | m1
                    for idx2 in range(idx1+1, num_blocks):
                        if idx2 in packing_set or idx2 == ci:
                            continue
                        m2 = block_masks[idx2]
                        if (new_mask & m2) != 0:
                            continue
                        # Can add both idx1 and idx2
                        packing_set.add(idx1)
                        packing_set.add(idx2)
                        added = 2
                        break
                    if added == 2:
                        break
                
                if added < 2:
                    # Revert
                    packing_set.add(ci)
                else:
                    improved = True
                    break
            
            # Update best
            if len(packing_set) > best_count:
                best_count = len(packing_set)
                best_packing = [all_blocks[i][0] for i in packing_set]
                if best_count >= 13:
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
    covered_mask = 0
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    cell_idx = i * m * p + j * p + k
                    covered_mask |= (1 << cell_idx)

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                cell_idx = i * m * p + j * p + k
                if not (covered_mask & (1 << cell_idx)):
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
