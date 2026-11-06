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
                
                # Find all APs that would be completed
                blocking_aps = []
                for mask in _conflict_mask_list[x]:
                    if (current & mask) == mask:
                        # Decode the mask back to elements
                        elems = []
                        for e in range(1, N + 1):
                            if (mask >> e) & 1:
                                elems.append(e)
                        blocking_aps.append(tuple(elems))
                
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
def _local_search(bits, deadline, max_passes=20):
    """
    Local search: try to add elements, and if blocked, try swapping.
    Uses precomputed conflict data for speed.
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
                
                # Find all APs that would be completed (using precomputed elems)
                blocking_aps = _get_blocking_aps(x, current)
                
                if not blocking_aps:
                    if _can_add(x, current):
                        current |= (1 << x)
                        current_size += 1
                        improved = True
                    continue
                
                # Try removing 1 element from blocking APs
                candidates = set()
                for ap in blocking_aps:
                    candidates.update(ap)
                
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
            
            if not improved:
                break
    
    return current
>>>>>>> REPLACE

