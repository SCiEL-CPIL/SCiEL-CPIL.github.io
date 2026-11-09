"""Size-efficient sorting networks on 17 wires."""
import time
import random
import sys

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

def _verify_sorting_network(network, n_wires):
    """
    Verify that the network sorts all 2^n_wires boolean inputs using the zero-one principle.
    Uses Python big integers as bitmasks for speed.
    """
    if not hasattr(_verify_sorting_network, '_initial_masks_cache'):
        _verify_sorting_network._initial_masks_cache = {}
    
    if n_wires not in _verify_sorting_network._initial_masks_cache:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _verify_sorting_network._initial_masks_cache[n_wires] = masks
    
    masks = _verify_sorting_network._initial_masks_cache[n_wires][:]
    
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

def _is_redundant(network, index, n_wires):
    """
    Check if the comparator at 'index' is redundant by simulating the network up to that point
    and checking if the comparator is a no-op on the current logical masks.
    Returns True if redundant, False otherwise.
    """
    if index < 0 or index >= len(network):
        return False
    
    # Get initial masks
    if not hasattr(_verify_sorting_network, '_initial_masks_cache'):
        _verify_sorting_network._initial_masks_cache = {}
    
    if n_wires not in _verify_sorting_network._initial_masks_cache:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _verify_sorting_network._initial_masks_cache[n_wires] = masks
    
    masks = _verify_sorting_network._initial_masks_cache[n_wires][:]
    
    # Apply comparators before the one in question
    for i in range(index):
        u, v = network[i]
        mu = masks[u]
        mv = masks[v]
        masks[u] = mu & mv
        masks[v] = mu | mv
        
    # Check the comparator in question
    u, v = network[index]
    mu = masks[u]
    mv = masks[v]
    
    # If mu is a subset of mv, then min(mu, mv) = mu and max(mu, mv) = mv.
    # The comparator does nothing.
    if (mu & mv) == mu:
        return True
        
    return False

def _simulate_network(network, n_wires):
    """
    Simulate the network and return the final masks.
    """
    if not hasattr(_verify_sorting_network, '_initial_masks_cache'):
        _verify_sorting_network._initial_masks_cache = {}
    
    if n_wires not in _verify_sorting_network._initial_masks_cache:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _verify_sorting_network._initial_masks_cache[n_wires] = masks
    
    masks = _verify_sorting_network._initial_masks_cache[n_wires][:]
    
    for i, j in network:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
        
    return masks

def _is_sorted_network(network, n_wires):
    """
    Check if the network is sorted using the final masks.
    """
    masks = _simulate_network(network, n_wires)
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            return False
    return True

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Start with seed network
    network = _build_seed_network()
    
    # Initial cleanup: remove redundant comparators using CDG check
    # We iterate until no more changes or time runs out
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        # Iterate backwards to avoid index shifting issues
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            if _is_redundant(network, i, N):
                network.pop(i)
                changed = True
        
    best_network = list(network)
    best_size = len(network)
    
    # Simulated Annealing
    current_network = list(best_network)
    current_size = len(current_network)
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    
    # Pre-generate some random comparators for reuse? No, generate on the fly.
    
    while time.time() < deadline - 2.0 and T > min_T:
        
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove a random comparator
            if current_size > 0:
                idx = random.randint(0, current_size - 1)
                # Check if removing it keeps it sorted (or if it's redundant)
                # Using _is_redundant is fast. But what if it's not redundant but removing it still sorts?
                # That's harder to check without full sim.
                # However, if it's not redundant, removing it likely breaks sorting.
                # Let's try removing it and checking full validity only if it looks promising?
                # Actually, just try it. If _is_redundant is True, it's definitely safe.
                # If False, it might still be safe (non-redundant but removable).
                # Let's just do a full check for removals to be safe, but limit frequency?
                # No, full check is 17 * 131072 ops ~ 2M ops. In Python, this is ~0.1s.
                # We can afford ~100-200 full checks per second? Maybe less.
                # Let's rely on _is_redundant for the fast path.
                
                if _is_redundant(current_network, idx, N):
                    new_network = current_network[:idx] + current_network[idx+1:]
                    new_size = current_size - 1
                    # Accept
                    current_network = new_network
                    current_size = new_size
                    if current_size < best_size:
                        best_network = list(current_network)
                        best_size = current_size
                else:
                    # Not redundant, so removing it will likely break sorting.
                    # We skip this move to save time.
                    pass
                    
        elif move_type < 0.8:
            # Swap adjacent comparators
            if current_size > 1:
                idx = random.randint(0, current_size - 2)
                new_network = current_network[:]
                new_network[idx], new_network[idx+1] = new_network[idx+1], new_network[idx]
                new_size = current_size
                
                # Check if still sorted
                if _is_sorted_network(new_network, N):
                    # Accept swap (even if same size)
                    current_network = new_network
                    # Update best if size improved (it didn't, but maybe it enables future removals)
                    # We don't update best_size here unless size < best_size
                    if new_size < best_size:
                        best_network = list(current_network)
                        best_size = new_size
                else:
                    # Reject with probability based on T
                    # Delta cost is 0 (size same), but it broke sorting? 
                    # Wait, SA usually allows bad moves. But "bad" means unsorted.
                    # If it's unsorted, it's invalid. We cannot keep an invalid network as current state?
                    # Actually, we CAN, if we plan to fix it. But it's risky.
                    # Let's only accept swaps that maintain validity.
                    # This makes it a hill-climbing on valid states.
                    pass
                    
        else:
            # Replace a comparator
            if current_size > 0:
                idx = random.randint(0, current_size - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_network = current_network[:]
                new_network[idx] = (i, j)
                new_size = current_size
                
                if _is_sorted_network(new_network, N):
                    current_network = new_network
                    if new_size < best_size:
                        best_network = list(current_network)
                        best_size = new_size

        # Cooling
        T *= cooling_rate
        
        # Periodic cleanup: remove redundant comparators
        if random.random() < 0.1:
            changed = True
            while changed and time.time() < deadline - 2.0:
                changed = False
                for i in range(len(current_network) - 1, -1, -1):
                    if _is_redundant(current_network, i, N):
                        current_network.pop(i)
                        current_size -= 1
                        changed = True
            if current_size < best_size:
                best_network = list(current_network)
                best_size = current_size

    # Final verification of best network
    if _verify_sorting_network(best_network, N):
        return best_network
    else:
        # Fallback to seed if something went wrong
        return _build_seed_network()

# EVOLVE-BLOCK-END
