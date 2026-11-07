    def local_search(selected, time_limit):
        selected_set = set(selected)
        current_sum = sum(selected)
        selected_mask = 0
        for x in selected:
            selected_mask |= cand_bitmasks[x]
            
        improved = True
        no_improve_count = 0
        max_no_improve = 20 # Restart after this many failed iterations
        
        while time.time() < time_limit - 5.0: # Keep 5s buffer
            improved = False
            
            # Try pure addition
            for j in sorted_candidates:
                if j in selected_set:
                    continue
                if (cand_bitmasks[j] & selected_mask) == 0:
                    selected.append(j)
                    selected_set.add(j)
                    selected_mask |= cand_bitmasks[j]
                    current_sum += j
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
                
            # Try 1-Swap
            found_swap = False
            sel_list = list(selected)
            random.shuffle(sel_list)
            
            for i in sel_list:
                if time.time() > time_limit - 2.0:
                    break
                others_mask = selected_mask & ~cand_bitmasks[i]
                best_j = -1
                best_gain = 0
                for j in sorted_candidates:
                    if j in selected_set:
                        continue
                    if (cand_bitmasks[j] & others_mask) == 0:
                        gain = j - i
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                            # Since sorted desc, first compatible j is the largest compatible j.
                            break
                
                if best_j != -1 and best_gain > 0:
                    selected.remove(i)
                    selected_set.remove(i)
                    selected.append(best_j)
                    selected_set.add(best_j)
                    selected_mask = others_mask | cand_bitmasks[best_j]
                    current_sum += best_gain
                    improved = True
                    found_swap = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
                
            # Try Limited 2-Swap
            # Instead of exhaustive, try a few random pairs
            if time.time() < time_limit - 2.0 and len(selected) >= 2:
                found_2swap = False
                # Try up to 10 random pairs of (i, k) to remove
                for _ in range(10):
                    if time.time() > time_limit - 1.0:
                        break
                    i, k = random.sample(selected, 2)
                    others_mask = selected_mask & ~cand_bitmasks[i] & ~cand_bitmasks[k]
                    
                    # Find best pair (j, l) to add
                    # To speed up, iterate candidates once and find best compatible j, then best compatible l for that j?
                    # Or just find any improving pair.
                    best_pair = None
                    best_gain = 0
                    
                    # Efficient way: 
                    # 1. Find all candidates compatible with others_mask.
                    # 2. Among those, find best pair (j,l) that are coprime to each other.
                    # This is still O(C^2) in worst case.
                    # Let's limit candidates to top 50 largest values for speed?
                    # Or just break early if we find a good gain.
                    
                    # Simple heuristic: Check top 20 largest candidates
                    top_cands = sorted_candidates[:20]
                    for j_idx, j in enumerate(top_cands):
                        if j in selected_set:
                            continue
                        if (cand_bitmasks[j] & others_mask) != 0:
                            continue
                        
                        # Check l in top_cands
                        for l in top_cands[j_idx+1:]:
                            if l in selected_set:
                                continue
                            if (cand_bitmasks[l] & others_mask) != 0:
                                continue
                            if (cand_bitmasks[j] & cand_bitmasks[l]) != 0:
                                continue
                            
                            gain = j + l - i - k
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (j, l)
                    
                    if best_pair and best_gain > 0:
                        j, l = best_pair
                        selected.remove(i)
                        selected.remove(k)
                        selected_set.remove(i)
                        selected_set.remove(k)
                        selected.append(j)
                        selected.append(l)
                        selected_set.add(j)
                        selected_set.add(l)
                        selected_mask = 0
                        for x in selected:
                            selected_mask |= cand_bitmasks[x]
                        current_sum += best_gain
                        improved = True
                        found_2swap = True
                        break
                        
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb
                    if len(selected) >= 2:
                        to_remove = random.sample(selected, 2)
                        for x in to_remove:
                            selected.remove(x)
                            selected_set.remove(x)
                        selected_mask = 0
                        for x in selected:
                            selected_mask |= cand_bitmasks[x]
                        current_sum = sum(selected)
                        
                        added = 0
                        shuffled_cands = list(candidates)
                        random.shuffle(shuffled_cands)
                        for x in shuffled_cands:
                            if added >= 2:
                                break
                            if x in selected_set:
                                continue
                            if (cand_bitmasks[x] & selected_mask) == 0:
                                selected.append(x)
                                selected_set.add(x)
                                selected_mask |= cand_bitmasks[x]
                                current_sum += x
                                added += 1
                    no_improve_count = 0
        
        return selected
