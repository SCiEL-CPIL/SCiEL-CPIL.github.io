<<<<<<< SEARCH
    def local_search_optimize(selected_indices, deadline):
        """Improved local search with multiple strategies"""
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
        
        # Strategy 1: Single move (replace one codeword with a neighbor)
        # Strategy 2: Try to remove a codeword and see if coverage is still complete
        # Strategy 3: 2-opt: remove two codewords, add one that covers the gap
        
        no_improve_count = 0
        max_no_improve = 4000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improvement_made = False
            
            # Strategy 2: Try to remove codewords (most direct improvement)
            # Pick a random codeword and check if removing it breaks coverage
            for _ in range(100):  # Try up to 100 removals per iteration
                if not selected_list:
                    break
                if time.time() > deadline:
                    break
                    
                c = selected_list[rng.integers(len(selected_list))]
                
                # Check if removing c leaves any point uncovered
                risky_points = neighbors[c][coverage_count[neighbors[c]] == 1]
                
                if len(risky_points) == 0:
                    # Safe to remove
                    coverage_count[neighbors[c]] -= 1
                    idx_in_list = selected_list.index(c)
                    selected_list.pop(idx_in_list)
                    selected_set.remove(c)
                    improvement_made = True
                    no_improve_count = 0
                    break
                else:
                    # Check if we can find a replacement that covers all risky points
                    # and is not already selected
                    # Try neighbors of c
                    for pos in range(1, ball_size):
                        c_prime = neighbors[c][pos]
                        if c_prime in selected_set:
                            continue
                        
                        # Check if c_prime covers all risky points
                        if np.all(np.isin(risky_points, neighbors[c_prime])):
                            # Perform swap
                            coverage_count[neighbors[c]] -= 1
                            coverage_count[neighbors[c_prime]] += 1
                            
                            idx_in_list = selected_list.index(c)
                            selected_list[idx_in_list] = c_prime
                            selected_set.remove(c)
                            selected_set.add(c_prime)
                            
                            improvement_made = True
                            no_improve_count = 0
                            break
                    if improvement_made:
                        break
            
            if not improvement_made:
                no_improve_count += 1
            
            # Track best
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
            
            # Occasionally do a full prune
            if no_improve_count % 100 == 0 and no_improve_count > 0:
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    selected_set = set(selected_list)
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
        
        return best_code
=======
    def local_search_optimize(selected_indices, deadline):
        """Improved local search with multiple strategies"""
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Use a dict for O(1) lookup of position in list
        selected_set = set(selected_list)
        pos_map = {c: i for i, c in enumerate(selected_list)}
        
        no_improve_count = 0
        max_no_improve = 6000
        
        def try_remove_or_swap(c):
            """Try to remove c or swap it with a neighbor. Returns True if successful."""
            nonlocal coverage_count, selected_list, selected_set, pos_map, no_improve_count
            
            risky_points = neighbors[c][coverage_count[neighbors[c]] == 1]
            
            if len(risky_points) == 0:
                # Safe to remove
                coverage_count[neighbors[c]] -= 1
                idx_in_list = pos_map[c]
                last_idx = selected_list.pop(idx_in_list)
                if idx_in_list != len(selected_list):
                    selected_list.append(last_idx)
                    pos_map[last_idx] = idx_in_list
                del pos_map[c]
                selected_set.remove(c)
                no_improve_count = 0
                return True
            else:
                # Try to find a replacement among neighbors of c
                # Sort neighbors by how many risky points they cover
                risky_set = set(risky_points.tolist())
                candidates = []
                for pos in range(1, ball_size):
                    c_prime = neighbors[c][pos]
                    if c_prime in selected_set:
                        continue
                    # Count how many risky points c_prime covers
                    covered = sum(1 for p in neighbors[c_prime] if p in risky_set)
                    if covered == len(risky_set):
                        candidates.append(c_prime)
                        break  # Take the first valid one for speed
                
                if candidates:
                    c_prime = candidates[0]
                    coverage_count[neighbors[c]] -= 1
                    coverage_count[neighbors[c_prime]] += 1
                    idx_in_list = pos_map[c]
                    selected_list[idx_in_list] = c_prime
                    del pos_map[c]
                    pos_map[c_prime] = idx_in_list
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    no_improve_count = 0
                    return True
                return False
        
        def try_two_opt():
            """Try to remove two codewords and add one that covers the gap. Returns True if successful."""
            nonlocal coverage_count, selected_list, selected_set, pos_map, no_improve_count
            
            if len(selected_list) < 2:
                return False
            
            # Pick two random codewords
            i = rng.integers(len(selected_list))
            j = rng.integers(len(selected_list) - 1)
            if j >= i:
                j += 1
            
            c1 = selected_list[i]
            c2 = selected_list[j]
            
            # Find points that would become uncovered
            temp_count = coverage_count.copy()
            temp_count[neighbors[c1]] -= 1
            temp_count[neighbors[c2]] -= 1
            
            uncovered = temp_count[temp_count <= 0]
            if len(uncovered) > 0:
                return False
            
            # Find a single ball that covers all points that would be at risk
            # Risky points: those with count == 1 after removing c1 and c2
            risky1 = neighbors[c1][coverage_count[neighbors[c1]] == 1]
            risky2 = neighbors[c2][coverage_count[neighbors[c2]] == 1]
            all_risky = set(risky1.tolist()) | set(risky2.tolist())
            
            if len(all_risky) == 0:
                # Both safe to remove
                coverage_count[neighbors[c1]] -= 1
                coverage_count[neighbors[c2]] -= 1
                # Remove both from list (remove higher index first)
                hi = max(i, j)
                lo = min(i, j)
                c_hi = selected_list.pop(hi)
                del pos_map[c_hi]
                selected_set.remove(c_hi)
                c_lo = selected_list.pop(lo)
                del pos_map[c_lo]
                selected_set.remove(c_lo)
                no_improve_count = 0
                return True
            
            # Try to find a ball that covers all risky points
            # Sample some candidate balls
            sample_size = min(50, total_points)
            candidates = rng.choice(total_points, sample_size, replace=False)
            for c_new in candidates:
                if c_new in selected_set:
                    continue
                if np.all(np.isin(list(all_risky), neighbors[c_new])):
                    # Perform the move: remove c1, c2, add c_new
                    coverage_count[neighbors[c1]] -= 1
                    coverage_count[neighbors[c2]] -= 1
                    coverage_count[neighbors[c_new]] += 1
                    
                    hi = max(i, j)
                    lo = min(i, j)
                    c_hi = selected_list.pop(hi)
                    del pos_map[c_hi]
                    selected_set.remove(c_hi)
                    c_lo = selected_list.pop(lo)
                    del pos_map[c_lo]
                    selected_set.remove(c_lo)
                    
                    selected_list.append(c_new)
                    pos_map[c_new] = len(selected_list) - 1
                    selected_set.add(c_new)
                    
                    no_improve_count = 0
                    return True
            
            return False
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improvement_made = False
            
            # Strategy 2: Try to remove or swap codewords
            for _ in range(50):
                if not selected_list:
                    break
                if time.time() > deadline:
                    break
                    
                c = selected_list[rng.integers(len(selected_list))]
                if try_remove_or_swap(c):
                    improvement_made = True
                    break
            
            if not improvement_made and len(selected_list) >= 2:
                # Try 2-opt
                if try_two_opt():
                    improvement_made = True
            
            if not improvement_made:
                no_improve_count += 1
            
            # Track best
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]
            
            # Occasionally do a full prune
            if no_improve_count % 50 == 0 and no_improve_count > 0:
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    selected_set = set(selected_list)
                    pos_map = {c: i for i, c in enumerate(selected_list)}
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
        
        return best_code
>>>>>>> REPLACE
