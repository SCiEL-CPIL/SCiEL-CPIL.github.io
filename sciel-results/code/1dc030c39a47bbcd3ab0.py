<<<<<<< SEARCH
def _local_search(bits, deadline, max_passes=20):
    """
    Local search: try to add elements, and if blocked, try swapping.
    Uses bitmask-based conflict checking for speed.
    """
    current = bits
    current_size = bin(current).count('1')
    
    for _pass in range(max_passes):
        if time.time() > deadline - 2:
            break
        improved = False
        
        # First try to add any element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True
        
        if not improved:
            # Try swaps: for each x not in set, find minimal removal set
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return current
                if (current >> x) & 1:
                    continue
                
                # Find all APs that would be completed (using precomputed mask-to-elems)
                blocking_aps = []
                for mask in _conflict_mask_list[x]:
                    if (current & mask) == mask:
                        blocking_aps.append(_mask_to_elems[mask])
                
                if not blocking_aps:
                    if _can_add(x, current):
                        current |= (1 << x)
                        current_size += 1
                        improved = True
                    continue
                
                # Try removing 1 element from blocking APs
                candidates = set()
                for ap in blocking_aps:
                    for elem in ap:
                        candidates.add(elem)
                
                # Try each single removal
                found_swap = False
                for rem in candidates:
                    if time.time() > deadline - 2:
                        return current
                    trial = current & ~(1 << rem)
                    if _can_add(x, trial):
                        trial |= (1 << x)
                        trial = _greedy_forward(trial)
                        trial_size = bin(trial).count('1')
                        if trial_size > current_size:
                            current = trial
                            current_size = trial_size
                            improved = True
                            found_swap = True
                            break
                
                # If no single swap worked, try 2-element removals
                if not found_swap and len(blocking_aps) <= 4 and time.time() < deadline - 3:
                    all_elems = set()
                    for ap in blocking_aps:
                        all_elems.update(ap)
                    elem_list = list(all_elems)
                    if len(elem_list) <= 12:
                        for i in range(len(elem_list)):
                            if time.time() > deadline - 2:
                                return current
                            for j in range(i + 1, len(elem_list)):
                                trial = current & ~(1 << elem_list[i]) & ~(1 << elem_list[j])
                                if _can_add(x, trial):
                                    trial |= (1 << x)
                                    trial = _greedy_forward(trial)
                                    trial_size = bin(trial).count('1')
                                    if trial_size > current_size:
                                        current = trial
                                        current_size = trial_size
                                        improved = True
                                        found_swap = True
                                        break
                            if found_swap:
                                break
            
            if not improved:
                break
    
    return current
=======
def _local_search(bits, deadline, max_passes=30):
    """
    Local search with incremental AP count tracking for speed.
    """
    current = bits
    current_size = bin(current).count('1')
    ap_counts = _build_ap_counts(current)
    
    for _pass in range(max_passes):
        if time.time() > deadline - 2:
            break
        improved = False
        
        # First try to add any element using fast AP count check
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add_fast(x, current, ap_counts):
                current |= (1 << x)
                current_size += 1
                improved = True
                # Update AP counts
                for ap_idx, _pos in _elem_ap_positions[x]:
                    ap_counts[ap_idx] += 1
        
        if not improved:
            # Try swaps: for each x not in set, find minimal removal set
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return current
                if (current >> x) & 1:
                    continue
                
                # Find all APs that would be completed
                blocking_aps = []
                for ap_idx, _pos in _elem_ap_positions[x]:
                    if ap_counts[ap_idx] >= 3:
                        blocking_aps.append(ap_idx)
                
                if not blocking_aps:
                    if _can_add_fast(x, current, ap_counts):
                        current |= (1 << x)
                        current_size += 1
                        improved = True
                        for ap_idx, _pos in _elem_ap_positions[x]:
                            ap_counts[ap_idx] += 1
                    continue
                
                # Try removing 1 element from blocking APs
                candidates = set()
                for ap_idx in blocking_aps:
                    a, b, c, e = _all_4aps[ap_idx]
                    for elem in (a, b, c, e):
                        if elem != x:
                            candidates.add(elem)
                
                found_swap = False
                for rem in candidates:
                    if time.time() > deadline - 2:
                        return current
                    # Check if removing rem unblocks x
                    trial_ok = True
                    for ap_idx, _pos in _elem_ap_positions[x]:
                        if ap_counts[ap_idx] >= 3:
                            # Check if rem is in this AP
                            ap_elem_set = set(_all_4aps[ap_idx])
                            if rem not in ap_elem_set:
                                trial_ok = False
                                break
                    if not trial_ok:
                        continue
                    
                    # Apply the swap
                    trial = current & ~(1 << rem) | (1 << x)
                    # Update counts: decrement for rem, increment for x
                    for ap_idx, _pos in _elem_ap_positions[rem]:
                        ap_counts[ap_idx] -= 1
                    for ap_idx, _pos in _elem_ap_positions[x]:
                        ap_counts[ap_idx] += 1
                    
                    # Verify and do greedy forward
                    trial = _greedy_forward(trial)
                    trial_size = bin(trial).count('1')
                    if trial_size > current_size:
                        current = trial
                        current_size = trial_size
                        ap_counts = _build_ap_counts(current)
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Revert
                        for ap_idx, _pos in _elem_ap_positions[rem]:
                            ap_counts[ap_idx] += 1
                        for ap_idx, _pos in _elem_ap_positions[x]:
                            ap_counts[ap_idx] -= 1
                        current = current & ~(1 << x) | (1 << rem)
                
                # If no single swap worked, try 2-element removals
                if not found_swap and len(blocking_aps) <= 5 and time.time() < deadline - 3:
                    all_elems = set()
                    for ap_idx in blocking_aps:
                        a, b, c, e = _all_4aps[ap_idx]
                        for elem in (a, b, c, e):
                            if elem != x:
                                all_elems.add(elem)
                    elem_list = list(all_elems)
                    if len(elem_list) <= 14:
                        for i in range(len(elem_list)):
                            if time.time() > deadline - 2:
                                return current
                            for j in range(i + 1, len(elem_list)):
                                rem1, rem2 = elem_list[i], elem_list[j]
                                # Check if removing both unblocks x
                                trial_ok = True
                                for ap_idx, _pos in _elem_ap_positions[x]:
                                    if ap_counts[ap_idx] >= 3:
                                        ap_elem_set = set(_all_4aps[ap_idx])
                                        if rem1 not in ap_elem_set and rem2 not in ap_elem_set:
                                            trial_ok = False
                                            break
                                if not trial_ok:
                                    continue
                                
                                trial = current & ~(1 << rem1) & ~(1 << rem2) | (1 << x)
                                trial = _greedy_forward(trial)
                                trial_size = bin(trial).count('1')
                                if trial_size > current_size:
                                    current = trial
                                    current_size = trial_size
                                    ap_counts = _build_ap_counts(current)
                                    improved = True
                                    found_swap = True
                                    break
                            if found_swap:
                                break
            
            if not improved:
                break
    
    return current
>>>>>>> REPLACE

