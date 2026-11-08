import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a refined search strategy: greedy set-packing with local search to maximize
    the number of disjoint 2x2x2 Strassen blocks, then schoolbook for the rest.
    
    Strategy:
    1. Generate all possible 2x2x2 Strassen blocks.
    2. Use multiple random restarts with greedy packing.
    3. Apply local search: try swapping 1 block for 2 to increase packing size.
    4. Use the best packing found to construct U, V, W.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    def get_block_indices(params):
        i1, i2, j1, j2, k1, k2 = params
        indices = set()
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    indices.add((i, j, k))
        return indices

    # Phase 1: Multiple random restarts with greedy packing
    best_num_blocks = 0
    best_packing = []
    
    num_trials = 500
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
            
        random.shuffle(all_blocks)
        used_indices = set()
        current_packing = []
        
        for block_params, block_indices in all_blocks:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
                
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
            
            # Early exit if we hit a good count
            if best_num_blocks >= 8:
                break

    # Phase 2: Local search to improve packing
    # Try to swap 1 block for 2 to increase the count
    if time.time() < deadline - 20.0:
        current_packing = best_packing[:]
        improved = True
        max_local_iterations = 50
        
        while improved and time.time() < deadline - 20.0:
            improved = False
            for _ in range(max_local_iterations):
                if time.time() > deadline - 20.0:
                    break
                    
                # Recompute used indices
                used_indices = set()
                for b in current_packing:
                    used_indices |= get_block_indices(b)
                
                # Try removing one block and adding two
                for idx_remove in range(len(current_packing)):
                    if time.time() > deadline - 20.0:
                        break
                        
                    removed_block = current_packing[idx_remove]
                    removed_indices = get_block_indices(removed_block)
                    
                    # Temporarily remove
                    temp_used = used_indices - removed_indices
                    temp_packing = current_packing[:idx_remove] + current_packing[idx_remove+1:]
                    
                    # Try to add two disjoint blocks
                    available = [b for b in all_blocks if b[0] not in temp_packing]
                    random.shuffle(available)
                    
                    added_first = None
                    for b1_params, b1_indices in available[:30]:
                        if b1_indices & temp_used:
                            continue
                        # Can we add a second block?
                        temp_used_after_first = temp_used | b1_indices
                        for b2_params, b2_indices in available[:30]:
                            if b2_params == b1_params:
                                continue
                            if b2_indices & temp_used_after_first:
                                continue
                            if b2_params in temp_packing:
                                continue
                            # Found a valid swap: remove 1, add 2
                            new_packing = temp_packing + [b1_params, b2_params]
                            if len(new_packing) > len(current_packing):
                                current_packing = new_packing
                                improved = True
                                if len(current_packing) > best_num_blocks:
                                    best_num_blocks = len(current_packing)
                                    best_packing = current_packing[:]
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Also try a different local search: hill-climbing on a different metric
    # Try to maximize the number of blocks by trying all single swaps
    if time.time() < deadline - 10.0 and best_num_blocks < 9:
        current_packing = best_packing[:]
        for iteration in range(20):
            if time.time() > deadline - 10.0:
                break
            
            used_indices = set()
            for b in current_packing:
                used_indices |= get_block_indices(b)
            
            best_swap = None
            best_swap_len = len(current_packing)
            
            for idx_remove in range(len(current_packing)):
                if time.time() > deadline - 10.0:
                    break
                    
                removed_block = current_packing[idx_remove]
                removed_indices = get_block_indices(removed_block)
                temp_used = used_indices - removed_indices
                temp_packing = current_packing[:idx_remove] + current_packing[idx_remove+1:]
                
                # Try adding one block
                for b_add_params, b_add_indices in all_blocks:
                    if b_add_params in temp_packing:
                        continue
                    if b_add_indices & temp_used:
                        continue
                    # This is a 1-for-1 swap, doesn't help
                    # Try adding two blocks
                    for b_add2_params, b_add2_indices in all_blocks:
                        if b_add2_params == b_add_params:
                            continue
                        if b_add2_params in temp_packing:
                            continue
                        if b_add2_indices & temp_used:
                            continue
                        temp_used_new = temp_used | b_add_indices
                        if b_add2_indices & temp_used_new:
                            continue
                        # Valid: remove 1, add 2
                        new_len = len(temp_packing) + 2
                        if new_len > best_swap_len:
                            best_swap_len = new_len
                            best_swap = (idx_remove, b_add_params, b_add2_params)
            
            if best_swap:
                idx_remove, b1_params, b2_params = best_swap
                removed_block = current_packing[idx_remove]
                removed_indices = get_block_indices(removed_block)
                temp_used = used_indices - removed_indices
                current_packing = current_packing[:idx_remove] + current_packing[idx_remove+1:]
                current_packing.append(b1_params)
                current_packing.append(b2_params)
                if len(current_packing) > best_num_blocks:
                    best_num_blocks = len(current_packing)
                    best_packing = current_packing[:]

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
