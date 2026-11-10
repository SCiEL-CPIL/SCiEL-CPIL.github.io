<<<<<<< SEARCH
    # Try multiple initial structures and keep the best
    best_masks = None
    best_score = -1
    
    num_starts = 5
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        if start_idx == 0:
            masks = build_initial_structure()
        else:
            # Perturbed start: take the best so far and randomly swap some
            if best_masks is None:
                masks = build_initial_structure()
            else:
                masks = best_masks[:]
                # Randomly swap 5-10 sets
                num_swaps = random.randint(5, 10)
                mask_set = set(masks)
                for _ in range(num_swaps):
                    idx = random.randint(0, m - 1)
                    old = masks[idx]
                    new = random.choice(all_subsets)
                    if new not in mask_set:
                        mask_set.remove(old)
                        mask_set.add(new)
                        masks[idx] = new
        
        # Compute initial score
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        
        if score > best_score:
            best_score = score
            best_masks = masks[:]
        
        # Local search on this structure
        current_masks = masks[:]
        current_set = set(current_masks)
        current_score = score
        
        # Build comparability matrix
        comp = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(current_masks[i], current_masks[j]):
                    comp[i][j] = 1
                    comp[j][i] = 1
        
        # Simulated annealing local search
        temperature = 2.0
        cooling = 0.9995
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5:
            if no_improve >= max_no_improve:
                if temperature > min_temp:
                    temperature = max(min_temp, temperature * 0.5)
                    no_improve = 0
                    max_no_improve = int(max_no_improve * 1.5)
                else:
                    break
            
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            new_mask = random.choice(all_subsets)
            
            if new_mask == old_mask or new_mask in current_set:
                continue
            
            # Compute delta
            old_contrib = 0
            for j in range(m):
                if j != idx:
                    old_contrib += comp[idx][j]
            
            new_contrib = 0
            for j in range(m):
                if j != idx:
                    if is_comparable(new_mask, current_masks[j]):
                        new_contrib += 1
            
            delta = new_contrib - old_contrib
            
            if delta > 0:
                # Accept
                for j in range(m):
                    if j != idx:
                        comp[idx][j] = 0
                        comp[j][idx] = 0
                        if is_comparable(new_mask, current_masks[j]):
                            comp[idx][j] = 1
                            comp[j][idx] = 1
                
                current_masks[idx] = new_mask
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_score += delta
                no_improve = 0
                temperature = max(min_temp, temperature * cooling)
            else:
                # Simulated annealing: accept with probability exp(delta/T)
                if temperature > min_temp:
                    import math
                    prob = math.exp(delta / temperature)
                    if random.random() < prob:
                        for j in range(m):
                            if j != idx:
                                comp[idx][j] = 0
                                comp[j][idx] = 0
                                if is_comparable(new_mask, current_masks[j]):
                                    comp[idx][j] = 1
                                    comp[j][idx] = 1
                        
                        current_masks[idx] = new_mask
                        current_set.remove(old_mask)
                        current_set.add(new_mask)
                        current_score += delta
                        no_improve = 0
                        temperature = max(min_temp, temperature * cooling)
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if current_score > best_score:
                best_score = current_score
                best_masks = current_masks[:]
=======
    # Try multiple initial structures and keep the best
    best_masks = None
    best_score = -1
    
    num_starts = 4
    for start_idx in range(num_starts):
        if time.time() > deadline - 12:
            break
        
        if start_idx == 0:
            masks = build_initial_structure()
        else:
            # Perturbed start: take the best so far and gently mutate
            if best_masks is None:
                masks = build_initial_structure()
            else:
                masks = best_masks[:]
                mask_set = set(masks)
                # Gentle mutation: swap 2-4 sets with nearby candidates
                num_swaps = random.randint(2, 4)
                for _ in range(num_swaps):
                    idx = random.randint(0, m - 1)
                    old = masks[idx]
                    # Generate a nearby candidate
                    candidates = generate_candidates(old, mask_set, k=5)
                    if candidates:
                        new = random.choice(candidates)
                        if new not in mask_set:
                            mask_set.remove(old)
                            mask_set.add(new)
                            masks[idx] = new
        
        # Compute initial score
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        
        if score > best_score:
            best_score = score
            best_masks = masks[:]
        
        # Local search on this structure
        current_masks = masks[:]
        current_set = set(current_masks)
        current_score = score
        
        # Build comparability matrix
        comp = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(current_masks[i], current_masks[j]):
                    comp[i][j] = 1
                    comp[j][i] = 1
        
        # Simulated annealing local search
        temperature = 3.0
        cooling = 0.9997
        min_temp = 0.05
        no_improve = 0
        max_no_improve = 1500
        
        while time.time() < deadline - 3:
            if no_improve >= max_no_improve:
                if temperature > min_temp:
                    temperature = max(min_temp, temperature * 0.5)
                    no_improve = 0
                    max_no_improve = int(max_no_improve * 1.3)
                else:
                    break
            
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Use structured candidate generation instead of uniform random
            candidates = generate_candidates(old_mask, current_set, k=10)
            if not candidates:
                candidates = [random.choice(all_subsets)]
            new_mask = random.choice(candidates)
            
            if new_mask == old_mask or new_mask in current_set:
                no_improve += 1
                continue
            
            # Compute delta
            old_contrib = 0
            for j in range(m):
                if j != idx:
                    old_contrib += comp[idx][j]
            
            new_contrib = 0
            for j in range(m):
                if j != idx:
                    if is_comparable(new_mask, current_masks[j]):
                        new_contrib += 1
            
            delta = new_contrib - old_contrib
            
            if delta > 0:
                # Accept
                for j in range(m):
                    if j != idx:
                        comp[idx][j] = 0
                        comp[j][idx] = 0
                        if is_comparable(new_mask, current_masks[j]):
                            comp[idx][j] = 1
                            comp[j][idx] = 1
                
                current_masks[idx] = new_mask
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_score += delta
                no_improve = 0
                temperature = max(min_temp, temperature * cooling)
            else:
                # Simulated annealing: accept with probability exp(delta/T)
                if temperature > min_temp:
                    prob = math.exp(delta / temperature)
                    if random.random() < prob:
                        for j in range(m):
                            if j != idx:
                                comp[idx][j] = 0
                                comp[j][idx] = 0
                                if is_comparable(new_mask, current_masks[j]):
                                    comp[idx][j] = 1
                                    comp[j][idx] = 1
                        
                        current_masks[idx] = new_mask
                        current_set.remove(old_mask)
                        current_set.add(new_mask)
                        current_score += delta
                        no_improve = 0
                        temperature = max(min_temp, temperature * cooling)
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if current_score > best_score:
                best_score = current_score
                best_masks = current_masks[:]
>>>>>>> REPLACE

Now let me also add a final polish pass that does a more thorough local search with tighter tolerance:

