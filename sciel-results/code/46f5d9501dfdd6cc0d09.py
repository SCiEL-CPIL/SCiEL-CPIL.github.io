<<<<<<< SEARCH
def _greedy_remove(nets, n_wires, time_limit):
    """
    Greedy removal: repeatedly try to remove each comparator.
    Multiple passes until no more removals possible or time runs out.
    """
    nets = list(nets)
    improved = True
    while improved:
        improved = False
        if time.time() > time_limit:
            break
        # Try removing each comparator in order
        for idx in range(len(nets)):
            if time.time() > time_limit:
                break
            comp = nets[idx]
            # Build network without this comparator
            test_net = nets[:idx] + nets[idx+1:]
            is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
            if is_valid is True:
                # Remove it
                nets = test_net
                improved = True
                break  # Restart from beginning since indices shifted
        # If we made progress, loop again to try more removals
    return nets

def _swap_search(nets, n_wires, time_limit, max_swaps_per_round=50):
    """
    Try swapping adjacent comparators to see if it enables further removals.
    Also try moving a comparator to a different position.
    """
    nets = list(nets)
    n_comps = len(nets)
    if n_comps < 2:
        return nets
    
    rng = random.Random(12345)
    best_net = nets
    best_count = n_comps
    
    for _ in range(max_swaps_per_round):
        if time.time() > time_limit:
            break
        if len(best_net) < 2:
            break
        # Try a random swap of two comparators
        i, j = rng.sample(range(len(best_net)), 2)
        test_net = list(best_net)
        test_net[i], test_net[j] = test_net[j], test_net[i]
        
        # First check if the swapped network is still valid
        is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
        if is_valid is not True:
            continue
        
        # Try greedy removal on the swapped network
        improved_net = _greedy_remove(test_net, n_wires, time_limit - 3.0)
        if len(improved_net) < best_count:
            best_net = improved_net
            best_count = len(best_net)
    
    return best_net

def _insert_remove_search(nets, n_wires, time_limit):
    """
    Try inserting a comparator at a strategic position, then doing greedy removal.
    This can sometimes reduce the total count by enabling a cascade of removals.
    """
    nets = list(nets)
    best_net = nets
    best_count = len(nets)
    rng = random.Random(99999)
    
    # Candidate comparator pairs: focus on pairs that are "close" in index
    # or involve wires that are frequently compared
    candidate_pairs = []
    for i in range(n_wires):
        for j in range(i + 1, n_wires):
            if j - i <= 3:  # Only nearby wires as insertion candidates
                candidate_pairs.append((i, j))
    
    rng.shuffle(candidate_pairs)
    
    for pair in candidate_pairs[:30]:  # Try up to 30 insertions
        if time.time() > time_limit:
            break
        if len(best_net) >= MAX_COMPARATORS:
            break
        
        # Try inserting at a random position
        pos = rng.randint(0, len(best_net))
        test_net = list(best_net)
        test_net.insert(pos, pair)
        
        # Check validity
        is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
        if is_valid is not True:
            continue
        
        # Try greedy removal
        improved_net = _greedy_remove(test_net, n_wires, time_limit - 3.0)
        if len(improved_net) < best_count:
            best_net = improved_net
            best_count = len(best_net)
    
    return best_net
=======
def _greedy_remove(nets, n_wires, time_limit):
    """
    Greedy removal: repeatedly try to remove each comparator.
    Multiple passes until no more removals possible or time runs out.
    Uses a priority-based approach to try most "redundant" comparators first.
    """
    nets = list(nets)
    improved = True
    while improved:
        improved = False
        if time.time() > time_limit:
            break
        
        # Try removing each comparator, starting from the middle (often more redundant)
        # and working outward
        mid = len(nets) // 2
        order = [mid]
        for offset in range(1, mid + 1):
            if mid - offset >= 0:
                order.append(mid - offset)
            if mid + offset < len(nets):
                order.append(mid + offset)
        
        for idx in order:
            if time.time() > time_limit:
                break
            comp = nets[idx]
            # Build network without this comparator
            test_net = nets[:idx] + nets[idx+1:]
            is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
            if is_valid is True:
                # Remove it
                nets = test_net
                improved = True
                break  # Restart from beginning since indices shifted
        # If we made progress, loop again to try more removals
    return nets

def _simulated_annealing(nets, n_wires, time_limit, max_iters=2000):
    """
    Simulated annealing to escape local minima.
    Temporarily accepts worse networks to find better configurations.
    """
    nets = list(nets)
    best_net = nets
    best_count = len(nets)
    current_temp = 5.0
    cooling_rate = 0.98
    rng = random.Random(42)
    
    start_time = time.time()
    iters = 0
    
    while current_temp > 0.1 and time.time() < time_limit and iters < max_iters:
        iters += 1
        
        # Generate a neighbor by one of several moves
        move_type = rng.choice(['remove', 'add', 'swap', 'move'])
        test_net = list(nets)
        
        if move_type == 'remove' and len(test_net) > 1:
            idx = rng.randint(0, len(test_net) - 1)
            test_net.pop(idx)
        elif move_type == 'add' and len(test_net) < MAX_COMPARATORS:
            # Add a random comparator
            i = rng.randint(0, n_wires - 1)
            j = rng.randint(i + 1, n_wires - 1)
            pos = rng.randint(0, len(test_net))
            test_net.insert(pos, (i, j))
        elif move_type == 'swap' and len(test_net) > 1:
            i, j = rng.sample(range(len(test_net)), 2)
            test_net[i], test_net[j] = test_net[j], test_net[i]
        elif move_type == 'move' and len(test_net) > 1:
            i, j = rng.sample(range(len(test_net)), 2)
            comp = test_net.pop(i)
            test_net.insert(j, comp)
        else:
            current_temp *= cooling_rate
            continue
        
        # Check validity
        is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
        
        if is_valid is True:
            # Accept if better, or with some probability if worse
            new_count = len(test_net)
            if new_count < len(nets):
                nets = test_net
                if new_count < best_count:
                    best_net = test_net
                    best_count = new_count
            else:
                # Accept worse with probability exp(-delta/temp)
                delta = new_count - len(nets)
                if delta > 0:
                    prob = math.exp(-delta / current_temp)
                    if rng.random() < prob:
                        nets = test_net
        # If invalid, reject
        
        current_temp *= cooling_rate
    
    return best_net

def _swap_search(nets, n_wires, time_limit, max_swaps_per_round=50):
    """
    Try swapping adjacent comparators to see if it enables further removals.
    Also try moving a comparator to a different position.
    """
    nets = list(nets)
    n_comps = len(nets)
    if n_comps < 2:
        return nets
    
    rng = random.Random(12345)
    best_net = nets
    best_count = n_comps
    
    for _ in range(max_swaps_per_round):
        if time.time() > time_limit:
            break
        if len(best_net) < 2:
            break
        # Try a random swap of two comparators
        i, j = rng.sample(range(len(best_net)), 2)
        test_net = list(best_net)
        test_net[i], test_net[j] = test_net[j], test_net[i]
        
        # First check if the swapped network is still valid
        is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
        if is_valid is not True:
            continue
        
        # Try greedy removal on the swapped network
        improved_net = _greedy_remove(test_net, n_wires, time_limit - 3.0)
        if len(improved_net) < best_count:
            best_net = improved_net
            best_count = len(best_net)
    
    return best_net

def _insert_remove_search(nets, n_wires, time_limit):
    """
    Try inserting a comparator at a strategic position, then doing greedy removal.
    This can sometimes reduce the total count by enabling a cascade of removals.
    """
    nets = list(nets)
    best_net = nets
    best_count = len(nets)
    rng = random.Random(99999)
    
    # Candidate comparator pairs: focus on pairs that are "close" in index
    # or involve wires that are frequently compared
    candidate_pairs = []
    for i in range(n_wires):
        for j in range(i + 1, n_wires):
            if j - i <= 3:  # Only nearby wires as insertion candidates
                candidate_pairs.append((i, j))
    
    rng.shuffle(candidate_pairs)
    
    for pair in candidate_pairs[:30]:  # Try up to 30 insertions
        if time.time() > time_limit:
            break
        if len(best_net) >= MAX_COMPARATORS:
            break
        
        # Try inserting at a random position
        pos = rng.randint(0, len(best_net))
        test_net = list(best_net)
        test_net.insert(pos, pair)
        
        # Check validity
        is_valid = _verify_sorts_fast(test_net, n_wires, timeout=2.0)
        if is_valid is not True:
            continue
        
        # Try greedy removal
        improved_net = _greedy_remove(test_net, n_wires, time_limit - 3.0)
        if len(improved_net) < best_count:
            best_net = improved_net
            best_count = len(best_net)
    
    return best_net
>>>>>>> REPLACE

