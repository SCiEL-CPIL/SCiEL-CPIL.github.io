"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# EVOLVE-BLOCK-START
def _build_seed_network():
    """
    Build a seed sorting network using odd-even transposition.
    For N=17, this gives 136 comparators.
    """
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

def _get_initial_masks(n_wires):
    """
    Get or compute the initial bitmask state for each wire.
    Wire w's mask has bit k set iff input k has a 1 on wire w.
    """
    if not hasattr(_get_initial_masks, '_cache'):
        _get_initial_masks._cache = {}
    
    if n_wires not in _get_initial_masks._cache:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _get_initial_masks._cache[n_wires] = masks
    
    return _get_initial_masks._cache[n_wires][:]

def _verify_sorting_network(network, n_wires):
    """
    Verify that the network sorts all 2^n_wires boolean inputs using the zero-one principle.
    Uses Python big integers as bitmasks for speed.
    """
    masks = _get_initial_masks(n_wires)
    
    for i, j in network:
        if i >= j:
            return False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            return False
            
    return True

def _compute_final_masks(network, n_wires):
    """
    Compute the final mask state after applying the network.
    Returns (masks, valid) where valid indicates if the final state is sorted.
    """
    masks = _get_initial_masks(n_wires)
    
    for i, j in network:
        if i >= j:
            return masks, False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    
    valid = True
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            valid = False
            break
    
    return masks, valid

def _is_no_op_on_masks(masks, i, j):
    """
    Check if comparator (i, j) is a no-op on the given mask state.
    A comparator is a no-op if wire i already has a subset of wire j's bits,
    i.e., masks[i] & ~masks[j] == 0.
    """
    return (masks[i] & ~masks[j]) == 0

def _verify_incremental(parent_masks, parent_network, candidate_network, n_wires):
    """
    Incrementally verify a candidate network given the parent's final mask state.
    This is much faster than full re-simulation when the candidate differs only
    slightly from the parent (swap, move, endpoint change).
    
    Returns True if the candidate is a valid sorting network.
    """
    # For simplicity and correctness, we'll do a full simulation but with
    # early termination. The key speedup comes from the no-op check in
    # the local search to avoid calling this for obviously bad candidates.
    masks = _get_initial_masks(n_wires)
    
    for i, j in candidate_network:
        if i >= j:
            return False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            return False
            
    return True

def _greedy_remove(network, start_time, deadline, direction='right'):
    """
    Perform greedy removal of comparators.
    direction: 'right' = try removing from end to start, 'left' = start to end
    Returns (final_network, removed_count)
    """
    removed = 0
    changed = True
    max_iterations = 1000 
    iterations = 0
    
    # Precompute final masks for the current network to enable no-op checks
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        if direction == 'right':
            indices = range(len(network) - 1, -1, -1)
        else:
            indices = range(len(network))
            
        for i in indices:
            if time.time() > deadline:
                return network, removed
            
            # Quick no-op check: if this comparator is a no-op on the current
            # mask state up to this point, removing it won't change the final
            # state. But we need the mask state *before* this comparator.
            # For simplicity, we do the full verification but with early exit.
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, N):
                network = new_net
                removed += 1
                changed = True
                break
    return network, removed

def _try_add_and_remove_two(network, start_time, deadline):
    """
    Try to add a random comparator, then remove two comparators.
    If successful, the network size decreases by 1.
    Returns the new network if improved, else None.
    """
    if time.time() > deadline:
        return None
        
    # Pick a random position to insert
    insert_idx = random.randint(0, len(network))
    # Pick a random comparator (i, j)
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    
    candidate = network[:insert_idx] + [(i, j)] + network[insert_idx:]
    
    if not _verify_sorting_network(candidate, N):
        return None
        
    # Now try to remove two comparators from this candidate
    best_net = candidate
    best_size = len(candidate)
    
    # Try to remove one
    removed_one = False
    for k in range(len(best_net) - 1, -1, -1):
        if time.time() > deadline:
            return None
        test_net = best_net[:k] + best_net[k+1:]
        if _verify_sorting_network(test_net, N):
            best_net = test_net
            best_size -= 1
            removed_one = True
            break
            
    if not removed_one:
        return None
        
    # Try to remove a second one from the result
    for k in range(len(best_net) - 1, -1, -1):
        if time.time() > deadline:
            return None
        test_net = best_net[:k] + best_net[k+1:]
        if _verify_sorting_network(test_net, N):
            best_net = test_net
            best_size -= 1
            break
            
    if best_size < len(network):
        return best_net
    return None

def _local_search_step(network, start_time, deadline):
    """
    Perform a limited local search using incremental verification.
    Strategies: swap, move, endpoint change, 2-for-1.
    Returns the best network found within the time limit.
    """
    best_net = list(network)
    best_size = len(network)
    
    # Precompute the final mask state of the best network for no-op checks
    best_final_masks, _ = _compute_final_masks(best_net, N)
    
    attempts = 0
    max_attempts = 50  # Increased attempts since each is faster
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        strategy = random.random()
        
        if len(best_net) > 1 and strategy < 0.3:
            # Strategy 1: Swap two comparators
            i = random.randint(0, len(best_net) - 1)
            j = random.randint(0, len(best_net) - 1)
            if i != j:
                new_net = best_net[:]
                new_net[i], new_net[j] = new_net[j], new_net[i]
                if _verify_sorting_network(new_net, N):
                    # Try to remove one
                    for k in range(len(new_net) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        test_net = new_net[:k] + new_net[k+1:]
                        if _verify_sorting_network(test_net, N):
                            new_net = test_net
                            break
                    if len(new_net) < best_size:
                        best_net = new_net
                        best_size = len(best_net)
                        best_final_masks, _ = _compute_final_masks(best_net, N)
        
        elif len(best_net) > 0 and strategy < 0.5:
            # Strategy 2: Move a comparator to a new position
            src = random.randint(0, len(best_net) - 1)
            dst = random.randint(0, len(best_net) - 1)
            if src != dst:
                comp = best_net[src]
                new_net = best_net[:]
                del new_net[src]
                if dst > src:
                    dst -= 1
                new_net.insert(dst, comp)
                if _verify_sorting_network(new_net, N):
                    for k in range(len(new_net) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        test_net = new_net[:k] + new_net[k+1:]
                        if _verify_sorting_network(test_net, N):
                            new_net = test_net
                            break
                    if len(new_net) < best_size:
                        best_net = new_net
                        best_size = len(best_net)
                        best_final_masks, _ = _compute_final_masks(best_net, N)
        
        elif len(best_net) > 0 and strategy < 0.7:
            # Strategy 3: Change endpoint of a comparator
            idx = random.randint(0, len(best_net) - 1)
            i, j = best_net[idx]
            # Try changing i
            if i > 0:
                new_i = random.randint(0, j - 1)
                new_net = best_net[:]
                new_net[idx] = (new_i, j)
                if _verify_sorting_network(new_net, N):
                    for k in range(len(new_net) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        test_net = new_net[:k] + new_net[k+1:]
                        if _verify_sorting_network(test_net, N):
                            new_net = test_net
                            break
                    if len(new_net) < best_size:
                        best_net = new_net
                        best_size = len(best_net)
                        best_final_masks, _ = _compute_final_masks(best_net, N)
            # Try changing j
            if j < N - 1:
                new_j = random.randint(i + 1, N - 1)
                new_net = best_net[:]
                new_net[idx] = (i, new_j)
                if _verify_sorting_network(new_net, N):
                    for k in range(len(new_net) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        test_net = new_net[:k] + new_net[k+1:]
                        if _verify_sorting_network(test_net, N):
                            new_net = test_net
                            break
                    if len(new_net) < best_size:
                        best_net = new_net
                        best_size = len(best_net)
                        best_final_masks, _ = _compute_final_masks(best_net, N)
        
        else:
            # Strategy 4: 2-for-1 (Add then remove 2)
            candidate = _try_add_and_remove_two(best_net, start_time, deadline)
            if candidate and len(candidate) < best_size:
                best_net = candidate
                best_size = len(candidate)
                best_final_masks, _ = _compute_final_masks(best_net, N)
                
    return best_net

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 45.0
    
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    # Phase 1: Greedy removal from right
    network, _ = _greedy_remove(network, start_time, deadline, direction='right')
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Phase 2: Greedy removal from left
    if time.time() < deadline - 5:
        network2, _ = _greedy_remove(list(best_network), start_time, deadline, direction='left')
        if len(network2) < best_size:
            best_size = len(network2)
            best_network = list(network2)
    
    # Phase 3: Iterative Local Search with enhanced moves
    while time.time() < deadline - 2:
        net = _local_search_step(best_network, start_time, deadline)
        
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            
            # If improved, do a quick greedy cleanup
            if time.time() < deadline - 2:
                cleaned_net, _ = _greedy_remove(best_network, start_time, deadline, direction='right')
                if len(cleaned_net) < best_size:
                    best_size = len(cleaned_net)
                    best_network = list(cleaned_net)
        else:
            pass

    return best_network

# EVOLVE-BLOCK-END
