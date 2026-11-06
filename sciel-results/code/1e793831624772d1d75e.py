<<<<<<< SEARCH
        # SA Loop
        steps = 0
        max_steps = 200000 # High number, limited by time
        
        while time.time() < deadline - 2:
            steps += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            
            # Update sums for APs containing i
            # Delta S_j = -2 * old_sign
            delta_s = -2.0 * old_sign
            
            # We need to update S in place for the objective calculation, 
            # but we also need to revert if rejected.
            # So we store the old values or compute the new objective from scratch?
            # Computing from scratch is O(num_aps). 
            # Updating delta is O(len(aps_containing[i])).
            # len(aps_containing[i]) is ~128.
            # num_aps is ~32000.
            # So incremental update is much faster for the state S.
            # But we need to calculate the NEW soft objective.
            # Soft obj depends on ALL S.
            # If we update S in place, we can calculate new_obj.
            # If rejected, we revert S.
            
            # Store old S values for affected APs?
            # Or just copy the whole S array? 32k floats is ~256KB. Copying is fast.
            # Let's try in-place update with revert.
            
            affected_aps = aps_containing[i]
            # Apply change
            S[affected_aps] += delta_s
            
            new_obj = calc_soft_obj(S, beta)
            delta_obj = new_obj - current_obj
            
            accept = False
            if delta_obj < 0:
                accept = True
            else:
                if delta_obj / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_obj / T)
                    
            if accept:
                signs[i] = -old_sign
                current_obj = new_obj
                # Update hard max check periodically
                if steps % 100 == 0:
                    current_hard_max = get_hard_max(S)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
            else:
                # Revert S
                S[affected_aps] -= delta_s
                
            # Annealing
            if steps % 1000 == 0:
                T *= 0.95
                beta *= 0.98
                if beta < 0.1:
                    beta = 0.1
                if T < 0.1:
                    T = 0.1
                    
            # Early stop if optimal
            if best_hard_max <= 1:
                break
=======
        # SA Loop with 1-flip and 2-flip moves
        steps = 0
        max_steps = 200000 # High number, limited by time
        
        while time.time() < deadline - 2:
            steps += 1
            
            # Decide move type: 80% 1-flip, 20% 2-flip
            use_two_flip = (steps % 5 == 0) and rng.random() < 0.3
            
            if not use_two_flip:
                # 1-flip move
                # Targeted selection: 30% chance to pick from worst APs
                if rng.random() < 0.3 and steps > 100:
                    # Get worst APs and pick a random element from one
                    worst_aps = get_worst_aps(S, k=10)
                    if len(worst_aps) > 0:
                        worst_ap_idx = rng.choice(worst_aps)
                        terms = ap_indices[worst_ap_idx]
                        i = rng.choice(terms)
                    else:
                        i = rng.randint(0, N - 1)
                else:
                    i = rng.randint(0, N - 1)
                
                old_sign = signs[i]
                delta_s = -2.0 * old_sign
                affected_aps = aps_containing[i]
                
                # Apply change
                S[affected_aps] += delta_s
                
                new_obj = calc_soft_obj(S, beta)
                delta_obj = new_obj - current_obj
                
                accept = False
                if delta_obj < 0:
                    accept = True
                else:
                    if delta_obj / T > 100:
                        accept = False
                    else:
                        accept = rng.random() < math.exp(-delta_obj / T)
                        
                if accept:
                    signs[i] = -old_sign
                    current_obj = new_obj
                    if steps % 100 == 0:
                        current_hard_max = get_hard_max(S)
                        if current_hard_max < best_hard_max:
                            best_hard_max = current_hard_max
                            best_signs = signs.copy()
                else:
                    S[affected_aps] -= delta_s
                    
            else:
                # 2-flip move: pick two positions to flip simultaneously
                # Targeted: bias toward positions in worst APs
                if steps > 100:
                    worst_aps = get_worst_aps(S, k=5)
                    if len(worst_aps) > 0:
                        # Pick two positions from different worst APs
                        i = rng.randint(0, N - 1)
                        # For j, try to pick from a worst AP
                        if worst_aps:
                            worst_ap_idx = rng.choice(worst_aps)
                            terms = ap_indices[worst_ap_idx]
                            j = rng.choice(terms)
                            if j == i:
                                j = rng.randint(0, N - 1)
                        else:
                            j = rng.randint(0, N - 1)
                    else:
                        i = rng.randint(0, N - 1)
                        j = rng.randint(0, N - 1)
                        if j == i:
                            j = (j + 1) % N
                else:
                    i = rng.randint(0, N - 1)
                    j = rng.randint(0, N - 1)
                    if j == i:
                        j = (j + 1) % N
                
                if i == j:
                    continue  # Skip if same position
                
                # Compute combined delta for both positions
                old_sign_i = signs[i]
                old_sign_j = signs[j]
                
                # Affected APs for both positions
                affected_i = aps_containing[i]
                affected_j = aps_containing[j]
                
                # Build combined affected set with net deltas
                # For each AP, the delta is sum of contributions from i and j
                # If AP contains i: delta_i = -2 * old_sign_i
                # If AP contains j: delta_j = -2 * old_sign_j
                # Net delta = delta_i + delta_j if both, else just one
                
                # Use a dict to accumulate deltas
                delta_map = {}
                for ap_idx in affected_i:
                    delta_map[ap_idx] = delta_map.get(ap_idx, 0.0) - 2.0 * old_sign_i
                for ap_idx in affected_j:
                    delta_map[ap_idx] = delta_map.get(ap_idx, 0.0) - 2.0 * old_sign_j
                
                # Apply all changes
                for ap_idx, delta in delta_map.items():
                    S[ap_idx] += delta
                
                new_obj = calc_soft_obj(S, beta)
                delta_obj = new_obj - current_obj
                
                accept = False
                if delta_obj < 0:
                    accept = True
                else:
                    if delta_obj / T > 100:
                        accept = False
                    else:
                        accept = rng.random() < math.exp(-delta_obj / T)
                        
                if accept:
                    signs[i] = -old_sign_i
                    signs[j] = -old_sign_j
                    current_obj = new_obj
                    if steps % 100 == 0:
                        current_hard_max = get_hard_max(S)
                        if current_hard_max < best_hard_max:
                            best_hard_max = current_hard_max
                            best_signs = signs.copy()
                else:
                    # Revert all changes
                    for ap_idx, delta in delta_map.items():
                        S[ap_idx] -= delta
                
            # Annealing
            if steps % 1000 == 0:
                T *= 0.95
                beta *= 0.98
                if beta < 0.1:
                    beta = 0.1
                if T < 0.1:
                    T = 0.1
                    
            # Early stop if optimal
            if best_hard_max <= 1:
                break
>>>>>>> REPLACE
