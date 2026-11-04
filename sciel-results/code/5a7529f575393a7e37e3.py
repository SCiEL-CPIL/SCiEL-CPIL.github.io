"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition search."""
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an optimized search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks = len(all_blocks)
    
    # Precompute conflict graph: block_conflicts[i] is a set of indices j such that block i and block j share an index
    block_conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                block_conflicts[i].add(j)
                block_conflicts[j].add(i)

    best_num_blocks = 0
    best_packing = []
    
    random.seed(42)
    
    # Helper to calculate score of a packing (number of blocks)
    def get_packing_size(packing_indices):
        return len(packing_indices)

    # Phase 1: Randomized Greedy with multiple orderings
    # We can afford many trials because N=630 is small
    num_trials = 5000
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
            
        # Random shuffle
        indices = list(range(num_blocks))
        random.shuffle(indices)
        
        used_indices = set()
        current_packing = []
        
        for idx in indices:
            block_params, block_set = all_blocks[idx]
            if not (block_set & used_indices):
                current_packing.append(idx)
                used_indices |= block_set
                
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing.copy()
            if best_num_blocks >= 10: # Theoretical max is floor(105/8)=13, but likely lower due to geometry
                break

    # Phase 2: Local Search - try swapping blocks to improve packing
    # We operate on indices of blocks
    current_packing = best_packing.copy()
    
    def is_conflict_free(packing_indices):
        # Check if any two blocks in packing share an index
        # This is O(K^2 * 8) which is fast for K~10
        for i in range(len(packing_indices)):
            for j in range(i+1, len(packing_indices)):
                if all_blocks[packing_indices[i]][1] & all_blocks[packing_indices[j]][1]:
                    return False
        return True

    # Re-verify best_packing just in case
    if not is_conflict_free(best_packing):
        # Fallback to greedy if something went wrong
        current_packing = []
        used_indices = set()
        indices = list(range(num_blocks))
        random.shuffle(indices)
        for idx in indices:
             block_params, block_set = all_blocks[idx]
             if not (block_set & used_indices):
                current_packing.append(idx)
                used_indices |= block_set
        best_num_blocks = len(current_packing)
        best_packing = current_packing

    improved = True
    max_local_iters = 100
    local_iter = 0
    
    while improved and local_iter < max_local_iters and time.time() < deadline - 10.0:
        improved = False
        local_iter += 1
        
        current_size = len(current_packing)
        current_set = set(current_packing)
        
        # 1. Try adding a block
        for idx in range(num_blocks):
            if idx in current_set:
                continue
            # Check if idx conflicts with any in current_packing
            conflict = False
            for p_idx in current_packing:
                if idx in block_conflicts[p_idx]:
                    conflict = True
                    break
            if not conflict:
                current_packing.append(idx)
                current_set.add(idx)
                improved = True
                break
        
        if improved:
            continue

        # 2. Try removing one block and adding two
        # Iterate over blocks to remove
        for remove_idx in current_packing:
            if time.time() > deadline - 10.0:
                break
            
            trial_packing = [x for x in current_packing if x != remove_idx]
            trial_set = set(trial_packing)
            
            # Find candidates to add
            # A candidate c is valid if it doesn't conflict with any block in trial_packing
            # And it is not remove_idx (which is already removed)
            # We want to add 2 such candidates
            
            candidates = []
            for add_idx in range(num_blocks):
                if add_idx in trial_set:
                    continue
                if add_idx == remove_idx:
                    continue
                
                # Check conflict with trial_packing
                ok = True
                for p_idx in trial_packing:
                    if add_idx in block_conflicts[p_idx]:
                        ok = False
                        break
                if ok:
                    candidates.append(add_idx)
            
            # Now we need to pick 2 from candidates that don't conflict with each other
            # Sort candidates by "popularity" (fewer conflicts) to find good pairs faster?
            # Or just check pairs.
            # Candidates list can be large, but we just need one valid pair.
            found_pair = False
            for i_c in range(len(candidates)):
                for j_c in range(i_c + 1, len(candidates)):
                    c1 = candidates[i_c]
                    c2 = candidates[j_c]
                    # Check if c1 and c2 conflict
                    if c1 not in block_conflicts[c2]:
                        # Valid pair
                        new_packing = trial_packing + [c1, c2]
                        # Verify full validity just in case (should be redundant if logic holds)
                        # new_set = set(new_packing)
                        # if is_conflict_free(new_packing):
                        current_packing = new_packing
                        improved = True
                        found_pair = True
                        break
                if found_pair:
                    break
            
            if found_pair:
                break
        
        # 3. Try removing two blocks and adding three
        # This is more expensive, so only do it if size hasn't improved in previous steps
        if not improved and len(current_packing) < 12:
             # Generate pairs to remove
             for i_r in range(len(current_packing)):
                 if time.time() > deadline - 10.0:
                     break
                 for j_r in range(i_r + 1, len(current_packing)):
                     r1 = current_packing[i_r]
                     r2 = current_packing[j_r]
                     
                     trial_packing = [x for k, x in enumerate(current_packing) if k != i_r and k != j_r]
                     trial_set = set(trial_packing)
                     
                     candidates = []
                     for add_idx in range(num_blocks):
                         if add_idx in trial_set:
                             continue
                         if add_idx == r1 or add_idx == r2:
                             continue
                         
                         ok = True
                         for p_idx in trial_packing:
                             if add_idx in block_conflicts[p_idx]:
                                 ok = False
                                 break
                         if ok:
                             candidates.append(add_idx)
                     
                     # Find 3 mutually non-conflicting candidates from candidates
                     # This is O(N^3) potentially, but we can break early
                     found_triple = False
                     for i_c in range(len(candidates)):
                         if found_triple: break
                         c1 = candidates[i_c]
                         for j_c in range(i_c + 1, len(candidates)):
                             if found_triple: break
                             c2 = candidates[j_c]
                             if c1 in block_conflicts[c2]:
                                 continue
                             for k_c in range(j_c + 1, len(candidates)):
                                 c3 = candidates[k_c]
                                 if c1 in block_conflicts[c3] or c2 in block_conflicts[c3]:
                                     continue
                                 # Valid triple
                                 new_packing = trial_packing + [c1, c2, c3]
                                 current_packing = new_packing
                                 improved = True
                                 found_triple = True
                                 break
                     
                     if found_triple:
                         break
                 if improved:
                     break

        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing.copy()
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
    for idx in best_packing:
        block_params, block_indices = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered.update(block_indices)

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
