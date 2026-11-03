    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        selected_arr = np.array(list(selected_set), dtype=np.int64)
        
        # Precompute conflict counts with selected set
        conflict_with_selected = np.zeros(N, dtype=np.int64)
        for k in selected_arr:
            conflict_with_selected += Conflict[k]
            
        improved = True
        max_iterations = 20 # Increased
        iteration = 0
        
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 1. 1-for-2 Swap
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iteration)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                neighbors_of_i = neighbor_lists[i]
                valid_js = [j for j in neighbors_of_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                compatible_with_rest = []
                for j in valid_js:
                    if conflict_with_selected[j] == 1:
                        compatible_with_rest.append(j)
                
                if len(compatible_with_rest) < 2:
                    continue
                
                found_pair = False
                for a in range(len(compatible_with_rest)):
                    if found_pair:
                        break
                    j1 = compatible_with_rest[a]
                    for b in range(a + 1, len(compatible_with_rest)):
                        j2 = compatible_with_rest[b]
                        if not Conflict[j1, j2]:
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            
                            conflict_with_selected -= Conflict[i]
                            conflict_with_selected += Conflict[j1]
                            conflict_with_selected += Conflict[j2]
                            break
                            
            # 2. Kick Move: Remove 2 random, re-greedy
            if not improved and len(selected_set) > 10:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Remove 2 random elements
                to_remove = rng_ls.choice(list(selected_set), size=2, replace=False)
                for k in to_remove:
                    selected_set.remove(k)
                    conflict_with_selected -= Conflict[k]
                    
                # Re-greedy fill
                active = np.ones(N, dtype=bool)
                for k in selected_set:
                    active[Conflict[k]] = False
                    active[k] = False
                    
                # Greedy add
                while active.any():
                    if time.time() > deadline:
                        break
                    active_indices = np.where(active)[0]
                    if len(active_indices) == 0:
                        break
                    
                    # Pick random from active
                    idx = rng_ls.choice(active_indices)
                    selected_set.add(idx)
                    conflict_with_selected += Conflict[idx]
                    active[Conflict[idx]] = False
                    active[idx] = False
                    
                # Check if improved
                # We don't track "improved" for kick directly in the loop variable, 
                # but we can check size increase.
                # Actually, the while loop condition is `improved`. 
                # We should set improved=True if size increased.
                # But we don't know the previous size here easily without storing it.
                # Let's store size before kick.
                
        return list(selected_set)
    