<<<<<<< SEARCH
    # Phase 2: Local search with removals and swaps
    stuck_counter = 0
    
    while time.time() < deadline - 8.0:
        if current_size == 0:
            break
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            
        if stuck_counter > 8 and time.time() < deadline - 12.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
            
            if not risky_points:
                current_cov_count[neighbors[c1]] -= 1
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                if len(risky_points) == 1:
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                else:
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                    for x in risky_points[1:]:
                        candidate_mask &= neighbor_mask[x]
                        if not candidate_mask.any():
                            break
                
                candidate_mask[c1] = False
                for c_code in current_code:
                    candidate_mask[c_code] = False
                
                valid_candidates = np.where(candidate_mask)[0]
                
                if len(valid_candidates) > 0:
                    c2 = int(random.choice(valid_candidates))
                    current_cov_count[neighbors[c1]] -= 1
                    current_code[c1_idx] = c2
                    current_cov_count[neighbors[c2]] += 1
                    stuck_counter = 0
            stuck_counter = 0
=======
    # Phase 2: Deterministic removal pass first
    for _pass in range(5):
        changed = False
        for i in range(len(current_code) - 1, -1, -1):
            c = current_code[i]
            if np.all(current_cov_count[neighbors[c]] > 1):
                current_cov_count[neighbors[c]] -= 1
                current_code.pop(i)
                current_size -= 1
                changed = True
        if not changed:
            break
        if current_size < best_size:
            best_size = current_size
            best_code = current_code[:]

    # Phase 3: Local search with removals and swaps
    stuck_counter = 0
    removal_attempts = 0
    
    while time.time() < deadline - 8.0:
        if current_size == 0:
            break
        
        # Try removal first with bias toward "loose" codewords
        # A codeword is "loose" if it covers few unique points
        if removal_attempts < 3:
            # Find a removable codeword deterministically
            removable = -1
            for i in range(current_size):
                c = current_code[i]
                if np.all(current_cov_count[neighbors[c]] > 1):
                    removable = i
                    break
            
            if removable >= 0:
                c = current_code[removable]
                current_cov_count[neighbors[c]] -= 1
                current_code.pop(removable)
                current_size -= 1
                stuck_counter = 0
                removal_attempts = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                continue
        
        # Random selection for swap
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            removal_attempts = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            removal_attempts += 1
            
        if stuck_counter > 5 and time.time() < deadline - 12.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
            
            if not risky_points:
                current_cov_count[neighbors[c1]] -= 1
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                removal_attempts = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Build candidate mask from intersection of neighbors of risky points
                if len(risky_points) == 1:
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                else:
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                    for x in risky_points[1:]:
                        candidate_mask &= neighbor_mask[x]
                        if not candidate_mask.any():
                            break
                
                candidate_mask[c1] = False
                for c_code in current_code:
                    candidate_mask[c_code] = False
                
                valid_candidates = np.where(candidate_mask)[0]
                
                if len(valid_candidates) > 0:
                    # Score candidates by how many currently-uncovered-adjacent points they'd cover
                    # Prefer candidates that cover more points with cov_count==1
                    gains = np.array([np.sum(current_cov_count[neighbors[cand]] == 1) for cand in valid_candidates])
                    # Add small random perturbation for diversity
                    gains += np.random.rand(len(valid_candidates)) * 0.1
                    best_cand = valid_candidates[np.argmax(gains)]
                    c2 = int(best_cand)
                    current_cov_count[neighbors[c1]] -= 1
                    current_code[c1_idx] = c2
                    current_cov_count[neighbors[c2]] += 1
                    stuck_counter = 0
                    removal_attempts = 0
            stuck_counter = 0
            removal_attempts = 0
>>>>>>> REPLACE

