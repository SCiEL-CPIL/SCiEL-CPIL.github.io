<<<<<<< SEARCH
    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Pre-compute a set for fast membership check
        selected_set = set(selected_list)
        
        # Pre-compute reverse neighbors for faster "who covers this point" checks if needed,
        # but here we use coverage_count directly.
        
        # To speed up swaps, we can pre-compute a mapping from index to position in list?
        # No, list changes. But we can use a dictionary index->pos.
        index_to_pos = {idx: i for i, idx in enumerate(selected_list)}
        
        while time.time() < deadline:
            # Randomly select a codeword to try moving
            if not selected_list:
                break
                
            c = selected_list[rng.integers(len(selected_list))]
            
            # Try all neighbors of c as potential replacements
            # This is more thorough than random sampling
            # Optimization: Only try neighbors that are NOT in selected_set
            for pos in range(1, ball_size):
                c_prime = neighbors[c][pos]
                
                if c_prime in selected_set:
                    continue
                
                # Tentative swap c -> c_prime
                # Check validity: 
                # 1. Points in neighbors[c] with count 1 must be covered by c_prime or remain covered?
                #    Actually, if we remove c, points in neighbors[c] with count 1 become count 0 (uncovered) 
                #    UNLESS they are also in neighbors[c_prime] (in which case count becomes 1+1-1=1? No.)
                #    Let's trace:
                #    Before: count[p]
                #    Remove c: count[p] -= 1 for p in neighbors[c]
                #    Add c_prime: count[p] += 1 for p in neighbors[c_prime]
                #    
                #    If p in neighbors[c] \ neighbors[c_prime]:
                #       new_count = count[p] - 1. Must be >= 0. So count[p] must be >= 1.
                #       If count[p] == 1, it becomes 0. INVALID.
                #    If p in neighbors[c_prime] \ neighbors[c]:
                #       new_count = count[p] + 1. Always >= 1. OK.
                #    If p in intersection:
                #       new_count = count[p]. OK.
                #
                #    So validity condition: 
                #    No point p in (neighbors[c] \ neighbors[c_prime]) has count[p] == 1.
                #    Equivalently: min(coverage_count[neighbors[c] \ neighbors[c_prime]]) >= 1 (if set non-empty)
                
                # Efficient check:
                # We need to check points in neighbors[c] that are NOT in neighbors[c_prime].
                # Since ball_size is small (17), we can just loop.
                
                valid = True
                for p in neighbors[c]:
                    # If p is in neighbors[c_prime], it's safe (count increases or stays same)
                    # We need to check if p is in neighbors[c_prime].
                    # Using a set for neighbors[c_prime] would be fast, but we don't have one precomputed for every c_prime on the fly easily.
                    # However, we can check: is p == c_prime or p in neighbors[c_prime][1:]?
                    # Or just check coverage_count after tentative update?
                    # Tentative update is O(17). Check is O(17).
                    
                    # Let's do the tentative update and check min.
                    pass
                
                # Better approach:
                # 1. Decrement neighbors[c]
                # 2. Increment neighbors[c_prime]
                # 3. Check if any count < 1 in the union of neighbors[c] and neighbors[c_prime]?
                #    Actually, only points in neighbors[c] could drop to 0.
                #    Points in neighbors[c_prime] only increase.
                #    So we only need to check points in neighbors[c].
                
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1
                
                # Check if any point in neighbors[c] is now 0
                if np.any(coverage_count[neighbors[c]] == 0):
                    # Invalid, revert
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                else:
                    # Valid swap
                    # Update list and set
                    pos_in_list = index_to_pos[c]
                    selected_list[pos_in_list] = c_prime
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    index_to_pos.pop(c)
                    index_to_pos[c_prime] = pos_in_list
                    
                    # Try to prune
                    pruned = prune(selected_list)
                    if len(pruned) < len(selected_list):
                        selected_list = pruned
                        selected_set = set(selected_list)
                        index_to_pos = {idx: i for i, idx in enumerate(selected_list)}
                        coverage_count = np.zeros(total_points, dtype=np.int32)
                        for idx in selected_list:
                            coverage_count[neighbors[idx]] += 1
                        
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                    else:
                        # If no pruning, do we keep the swap or revert?
                        # The swap didn't reduce size, but it might allow future prunings.
                        # However, if it doesn't reduce size immediately, it's a "sideways" move.
                        # To avoid getting stuck in cycles or drifting, we might revert unless it enables a prune.
                        # But here we already checked prune. If prune didn't help, the size is same.
                        # Let's keep the swap to explore the space (Random Walk).
                        # But we must be careful not to cycle.
                        # For now, keep it.
                        pass
                    break # Restart outer loop after a swap
        
        return best_code
=======
    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Pre-compute a set for fast membership check
        selected_set = set(selected_list)
        
        # Pre-compute distance-2 neighbors for each index (for expanded swap moves)
        # dist2_neighbors[idx] = list of indices at Hamming distance 2 from idx
        dist2_neighbors = np.empty((total_points, 48), dtype=np.int32)
        for idx in range(total_points):
            temp_idx = idx
            word = [0] * n
            for i in range(n):
                word[i] = temp_idx % q
                temp_idx //= q
            
            count = 0
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for d1 in range(1, q):
                        for d2 in range(1, q):
                            new_word = word[:]
                            new_word[pos1] = (word[pos1] + d1) % q
                            new_word[pos2] = (word[pos2] + d2) % q
                            new_idx = 0
                            for i in range(n):
                                new_idx = new_idx * q + new_word[n - 1 - i]
                            # Actually let me compute properly
                            new_idx = 0
                            for i in range(n):
                                new_idx += new_word[i] * powers[i]
                            dist2_neighbors[idx, count] = new_idx
                            count += 1
            # Pad with -1 if needed (shouldn't be, 48 = C(8,2)*4)
        
        # Pre-compute neighbor sets for fast intersection checks
        # neighbor_set[idx] = frozenset of the 17 points in the ball
        neighbor_set = [frozenset(neighbors[i].tolist()) for i in range(total_points)]
        
        index_to_pos = {idx: i for i, idx in enumerate(selected_list)}
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            c = selected_list[rng.integers(len(selected_list))]
            
            # Build list of candidate replacements: distance-1 and distance-2 neighbors
            candidates = []
            # Distance 1
            for pos in range(1, ball_size):
                c_prime = neighbors[c][pos]
                if c_prime not in selected_set:
                    candidates.append(c_prime)
            # Distance 2 (expanded neighborhood)
            for pos in range(48):
                c_prime = dist2_neighbors[c][pos]
                if c_prime not in selected_set:
                    candidates.append(c_prime)
            
            if not candidates:
                continue
            
            # Shuffle candidates for diversity
            rng.shuffle(candidates)
            
            swap_done = False
            for c_prime in candidates:
                # Tentative swap c -> c_prime
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1
                
                # Check if any point in neighbors[c] is now 0
                if np.any(coverage_count[neighbors[c]] == 0):
                    # Invalid, revert
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                else:
                    # Valid swap
                    pos_in_list = index_to_pos[c]
                    selected_list[pos_in_list] = c_prime
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    index_to_pos.pop(c)
                    index_to_pos[c_prime] = pos_in_list
                    
                    # Try to prune
                    pruned = prune(selected_list)
                    if len(pruned) < len(selected_list):
                        selected_list = pruned
                        selected_set = set(selected_list)
                        index_to_pos = {idx: i for i, idx in enumerate(selected_list)}
                        coverage_count = np.zeros(total_points, dtype=np.int32)
                        for idx in selected_list:
                            coverage_count[neighbors[idx]] += 1
                        
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                    else:
                        # Keep the swap as a sideways move
                        pass
                    swap_done = True
                    break # Restart outer loop after a swap
            
            if not swap_done:
                # No valid swap found for this c, try another
                continue
        
        return best_code
>>>>>>> REPLACE

Now let me also improve the greedy phase with better tie-breaking:

