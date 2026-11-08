import time
import random
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Strategy:
    1. Identify all possible 2x2x2 Strassen blocks.
    2. Formulate a Maximum Set Packing problem: select the maximum number of disjoint Strassen blocks.
       Each selected block reduces the rank by 1 (from 8 terms to 7).
    3. Solve using a Branch-and-Bound heuristic with constraint propagation (greedy + local search).
    4. Construct U, V, W matrices based on the selected blocks and remaining schoolbook terms.
    
    Theoretical bounds:
    - Schoolbook rank: 3*5*7 = 105.
    - Max possible blocks: floor(105/8) = 13.
    - Min possible rank: 105 - 13 = 92.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 Strassen blocks
    # A block is defined by choosing 2 rows (i), 2 cols (j) from A, and 2 cols (k) from B.
    # Indices: i1 < i2, j1 < j2, k1 < k2
    all_blocks = []
    block_sets = []
    
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
                            block_sets.append(frozenset(indices))
    
    num_blocks = len(all_blocks)
    
    # Pre-compute conflict graph: two blocks conflict if they share any (i,j,k) index
    # conflict_mask[b] is a bitset of blocks that conflict with b
    # Since num_blocks is small (~ C(3,2)*C(5,2)*C(7,2) = 3*10*21 = 630), we can use lists of sets or bitsets.
    # 630 bits fits in Python integers easily.
    conflict_masks = [0] * num_blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if not (block_sets[i] & block_sets[j]):
                continue
            # They conflict
            conflict_masks[i] |= (1 << j)
            conflict_masks[j] |= (1 << i)

    best_num_blocks = 0
    best_packing_indices = []
    
    # Branch and Bound Heuristic
    # We use a randomized greedy approach with local search (2-swap) to find a large independent set.
    # Since exact Max Independent Set is NP-hard, we rely on multiple random restarts and local improvements.
    
    def get_conflicting_set(selected_indices):
        """Returns a set of indices that conflict with any selected index."""
        conflicts = set()
        for idx in selected_indices:
            mask = conflict_masks[idx]
            # Extract bits from mask
            # Optimization: pre-compute list of conflicting indices for each block? 
            # Or just iterate. 630 blocks is small.
            for j in range(num_blocks):
                if mask & (1 << j):
                    conflicts.add(j)
        return conflicts

    def greedy_select(available_indices):
        """
        Greedily select blocks from available_indices.
        available_indices is a set of block indices.
        Returns a list of selected indices.
        """
        selected = []
        # Sort by degree (number of conflicts) or just random? 
        # Let's sort by a static heuristic: number of conflicts (higher degree first is often bad for greedy IS, 
        # but for packing, maybe we want "tight" blocks first? Actually, standard greedy for Max IS is min-degree.
        # Let's try sorting by number of conflicts in the *remaining* set dynamically? That's expensive.
        # Let's just pick randomly from available, or sorted by original degree.
        
        # Pre-compute original degrees
        degrees = [bin(conflict_masks[i]).count('1') for i in range(num_blocks)]
        
        # Sort available by degree (ascending) - Min-Vertex-Degree heuristic
        sorted_avail = sorted(available_indices, key=lambda x: degrees[x])
        
        for idx in sorted_avail:
            # Check if it still conflicts with nothing selected
            # Since we remove conflicts, if idx is in available, it doesn't conflict with selected.
            # But wait, 'available' is passed in. We need to update it.
            pass
            
        # Correct Greedy:
        selected = []
        remaining = set(available_indices)
        while remaining:
            # Pick block with minimum degree in the *induced subgraph* of remaining?
            # That's O(N^2) per step. N=630. 630*630*630 ~ 250M ops. Too slow for many trials.
            # Simpler: Pick random or static degree.
            # Let's pick the one with min static degree among remaining.
            # Or just random for diversity.
            
            # Let's use a simple randomized greedy:
            # Pick a random block from remaining?
            # Better: Pick block with fewest conflicts in remaining.
            # Approximation: Use static degree.
            
            # To speed up, we can just iterate through a shuffled list of available_indices
            # and pick the first one that doesn't conflict with selected.
            # But we want *maximum* packing.
            
            # Let's try: Sort remaining by static degree, pick first valid.
            # "Valid" means no conflict with selected.
            # Since we remove conflicts from 'remaining' when we pick, all in 'remaining' are valid.
            # So we just pick the one with min degree.
            
            min_deg_idx = -1
            min_deg = float('inf')
            for idx in remaining:
                if degrees[idx] < min_deg:
                    min_deg = degrees[idx]
                    min_deg_idx = idx
            
            if min_deg_idx == -1:
                break
                
            selected.append(min_deg_idx)
            
            # Remove min_deg_idx and all its conflicts from remaining
            to_remove = {min_deg_idx}
            mask = conflict_masks[min_deg_idx]
            for j in range(num_blocks):
                if mask & (1 << j):
                    to_remove.add(j)
            
            remaining -= to_remove
            
        return selected

    # Local Search: 2-Swap
    # Given a packing S, can we remove one block b1 and add two blocks b2, b3 
    # such that b2, b3 don't conflict with S \ {b1} and don't conflict with each other?
    # This increases size by 1.
    
    def local_search(packing):
        packing_set = set(packing)
        improved = True
        while improved:
            improved = False
            # Try removing each block in packing
            for b1 in list(packing_set):
                new_packing_set = packing_set - {b1}
                
                # Find candidates that don't conflict with new_packing_set
                # Candidate c must not conflict with any in new_packing_set.
                # And c != b1.
                
                # To be efficient, we can pre-compute for each block the set of non-conflicting blocks.
                # non_conflicts[b] = set of blocks that do NOT conflict with b.
                
                # Let's find pairs (b2, b3) in non_conflicts[b1] 
                # such that b2, b3 are not in new_packing_set, 
                # b2, b3 don't conflict with each other,
                # and b2, b3 don't conflict with new_packing_set.
                
                # This is expensive if done naively.
                # Heuristic: Only try swaps where b2 and b3 were previously "blocked" by b1.
                
                # Let's just try a few random candidates for speed.
                # Or iterate through all pairs? 630^2 is 400k. 630 blocks in packing * 400k = 250M. Too slow.
                
                # Alternative: 
                # 1. Compute candidates = {c | c not in new_packing_set and c does not conflict with any in new_packing_set}
                # 2. From candidates, find two that don't conflict with each other.
                
                # Step 1:
                candidates = []
                for c in range(num_blocks):
                    if c in new_packing_set:
                        continue
                    # Check conflict with new_packing_set
                    conflict_found = False
                    for s in new_packing_set:
                        if conflict_masks[s] & (1 << c):
                            conflict_found = True
                            break
                    if not conflict_found:
                        candidates.append(c)
                
                # Step 2: Find pair in candidates with no conflict
                # We want to add 2 blocks.
                # Iterate candidates
                found_pair = False
                for i in range(len(candidates)):
                    b2 = candidates[i]
                    # Check if b2 conflicts with any in new_packing_set? Already checked.
                    # Check if b2 conflicts with other candidates?
                    # We just need ONE pair (b2, b3).
                    # Iterate j > i
                    for j in range(i + 1, len(candidates)):
                        b3 = candidates[j]
                        if not (conflict_masks[b2] & (1 << b3)):
                            # Found a valid pair
                            new_packing_set.add(b2)
                            new_packing_set.add(b3)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    packing_set = new_packing_set
                    # Restart while loop? Or continue? 
                    # Usually restart to be safe.
                    break
        
        return list(packing_set)

    # Run multiple trials
    num_trials = 1000
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        # Randomized Greedy
        # Shuffle blocks to break ties in degree or just random order
        random_indices = list(range(num_blocks))
        random.shuffle(random_indices)
        
        # Greedy selection using random order as tie-breaker or primary?
        # Let's use the min-degree heuristic but with random initialization if ties.
        # Actually, let's just use the greedy_select function defined above which uses min-degree.
        # To add diversity, we can randomly remove a few "top" candidates or just rely on local search.
        
        initial_packing = greedy_select(set(range(num_blocks)))
        
        # Local Search
        improved_packing = local_search(initial_packing)
        
        if len(improved_packing) > best_num_blocks:
            best_num_blocks = len(improved_packing)
            best_packing_indices = improved_packing[:]
            
            if best_num_blocks == 13:
                break

    # Construct U, V, W
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

    covered = set()
    for b_idx in best_packing_indices:
        i1, i2, j1, j2, k1, k2 = all_blocks[b_idx][0]
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
