<<<<<<< SEARCH
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute overlap graph and clusters (one-time cost)
    overlap_A = None
    cluster_indices = None
    cluster_block_indices_list = None
    
    if num_blocks <= 50000:  # Only build if not too large
        try:
            overlap_A, sampled_indices = build_overlap_graph_sampled(1500)
            if overlap_A is not None:
                labels = spectral_partition(overlap_A, num_clusters=4)
                cluster_indices = sampled_indices
                cluster_block_indices_list = []
                for c in range(4):
                    mask = labels == c
                    cluster_block_indices_list.append(sampled_indices[mask].tolist())
        except Exception:
            pass
    
    trial = 0
    while time.time() < deadline - 3.0:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 10.0:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_to_subsets_arr[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        # Strategy 3: Spectral-guided greedy with cluster-based priority
        if time.time() < deadline - 5.0 and cluster_block_indices_list is not None:
            # Create priority vector: blocks in earlier clusters get higher priority
            priorities = np.zeros(num_blocks, dtype=np.float64)
            rng_p = np.random.default_rng(trial * 13 + 7)
            
            # Randomly assign priority boosts to clusters
            for c_idx, blocks_in_cluster in enumerate(cluster_block_indices_list):
                boost = rng_p.uniform(1.0, 5.0)
                for b_idx in blocks_in_cluster:
                    if b_idx < num_blocks:
                        priorities[b_idx] = boost
            
            spectral_res = weighted_greedy_with_priority(priorities, random_seed=trial + 8000)
            min_res3 = minimize(spectral_res, trial + 9000)
            
            if len(min_res3) < best_len:
                best_len = len(min_res3)
                best_blocks = min_res3
        
        # Strategy 4: Different noise level for diversity
        if time.time() < deadline - 4.0:
            rng3 = np.random.default_rng(trial * 7 + 42)
            uncovered = np.ones(num_subsets, dtype=bool)
            gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
            remaining_uncovered = num_subsets
            selected_blocks = []
            
            while remaining_uncovered > 0:
                noise = rng3.uniform(0, 1.0, size=num_blocks)
                scores = gains + noise
                best_idx = int(np.argmax(scores))
                
                if gains[best_idx] == 0:
                    break
                
                selected_blocks.append(best_idx)
                
                b_subs = block_to_subsets_arr[best_idx]
                newly_covered_mask = uncovered[b_subs]
                if not np.any(newly_covered_mask):
                    continue
                newly_covered = b_subs[newly_covered_mask]
                n_new = len(newly_covered)
                
                uncovered[newly_covered] = False
                remaining_uncovered -= n_new
                
                if n_new > 0:
                    all_block_idxs = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered])
                    np.add.at(gains, all_block_idxs, -1)
            
            min_res4 = minimize(selected_blocks, trial + 5000)
            if len(min_res4) < best_len:
                best_len = len(min_res4)
                best_blocks = min_res4

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
=======
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute overlap graph and clusters (one-time cost) — limit to 5s
    overlap_A = None
    cluster_indices = None
    cluster_block_indices_list = None
    
    if num_blocks <= 50000 and time.time() < start_time + 5.0:  # Only build if not too large and time permits
        try:
            overlap_A, sampled_indices = build_overlap_graph_sampled(1000)
            if overlap_A is not None:
                labels = spectral_partition(overlap_A, num_clusters=4)
                cluster_indices = sampled_indices
                cluster_block_indices_list = []
                for c in range(4):
                    mask = labels == c
                    cluster_block_indices_list.append(sampled_indices[mask].tolist())
        except Exception:
            pass
    
    trial = 0
    while time.time() < deadline - 3.0:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 10.0:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_to_subsets_arr[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        # Strategy 3: Weighted greedy with random priorities (replaces expensive spectral-guided)
        if time.time() < deadline - 5.0:
            priorities = np.zeros(num_blocks, dtype=np.float64)
            rng_p = np.random.default_rng(trial * 13 + 7)
            
            # Randomly assign priority boosts to a random subset of blocks
            n_boost = min(500, num_blocks)
            boost_idx = rng_p.choice(num_blocks, size=n_boost, replace=False)
            for b_idx in boost_idx:
                priorities[b_idx] = rng_p.uniform(1.0, 5.0)
            
            spectral_res = weighted_greedy_with_priority(priorities, random_seed=trial + 8000)
            min_res3 = minimize(spectral_res, trial + 9000)
            
            if len(min_res3) < best_len:
                best_len = len(min_res3)
                best_blocks = min_res3
        
        # Strategy 4: Different noise level for diversity
        if time.time() < deadline - 4.0:
            rng3 = np.random.default_rng(trial * 7 + 42)
            uncovered = np.ones(num_subsets, dtype=bool)
            gains = block_subset_lengths.copy()
            remaining_uncovered = num_subsets
            selected_blocks = []
            
            while remaining_uncovered > 0:
                noise = rng3.uniform(0, 1.0, size=num_blocks)
                scores = gains + noise
                best_idx = int(np.argmax(scores))
                
                if gains[best_idx] == 0:
                    break
                
                selected_blocks.append(best_idx)
                
                b_subs = block_to_subsets_arr[best_idx]
                newly_covered_mask = uncovered[b_subs]
                if not np.any(newly_covered_mask):
                    continue
                newly_covered = b_subs[newly_covered_mask]
                n_new = len(newly_covered)
                
                uncovered[newly_covered] = False
                remaining_uncovered -= n_new
                
                if n_new > 0:
                    total_len = sum(len(subset_to_blocks_arr[s]) for s in newly_covered)
                    all_block_idxs = np.empty(total_len, dtype=np.int32)
                    pos = 0
                    for s in newly_covered:
                        arr = subset_to_blocks_arr[s]
                        all_block_idxs[pos:pos+len(arr)] = arr
                        pos += len(arr)
                    np.add.at(gains, all_block_idxs, -1)
            
            min_res4 = minimize(selected_blocks, trial + 5000)
            if len(min_res4) < best_len:
                best_len = len(min_res4)
                best_blocks = min_res4

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
>>>>>>> REPLACE
