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

def _greedy_remove(network, start_time, deadline, direction='right'):
    """
    Perform greedy removal of comparators.
    direction: 'right' = try removing from end to start, 'left' = start to end
    Returns (final_network, removed_count)
    """
    removed = 0
    changed = True
    while changed:
        changed = False
        if direction == 'right':
            indices = range(len(network) - 1, -1, -1)
        else:
            indices = range(len(network))
            
        for i in indices:
            if time.time() > deadline:
                return network, removed
            new_net = network[:i] + network[i+1:]
            if _verify_sorting_network(new_net, N):
                network = new_net
                removed += 1
                changed = True
                break
    return network, removed

def _local_search_with_swaps(network, start_time, deadline):
    """
    Try swapping adjacent comparators to potentially enable further removals.
    This is a more powerful local search move.
    """
    best_net = list(network)
    best_size = len(network)
    
    # Try swapping adjacent comparators
    for i in range(len(network) - 1):
        if time.time() > deadline:
            return best_net
        
        # Try swapping network[i] and network[i+1]
        new_net = network[:]
        new_net[i], new_net[i+1] = new_net[i+1], new_net[i]
        
        # After swap, try to remove comparators
        swapped_net, removed = _greedy_remove(new_net, start_time, deadline, direction='right')
        
        if len(swapped_net) < best_size:
            best_net = swapped_net
            best_size = len(swapped_net)
    
    return best_net

def _random_comparator(network, start_time, deadline):
    """
    Randomly replace one comparator with a different one, then try to remove.
    This allows escaping local minima by exploring the space of networks.
    """
    best_net = list(network)
    best_size = len(network)
    
    # Try random replacements
    for _ in range(50):
        if time.time() > deadline:
            return best_net
        
        # Pick a random index
        idx = random.randint(0, len(network) - 1)
        
        # Generate a random valid comparator (i, j) with i < j
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Replace the comparator
        new_net = network[:]
        new_net[idx] = (i, j)
        
        # Verify the network is still valid
        if _verify_sorting_network(new_net, N):
            # Try to remove comparators
            reduced_net, _ = _greedy_remove(new_net, start_time, deadline, direction='right')
            
            if len(reduced_net) < best_size:
                best_net = reduced_net
                best_size = len(reduced_net)
    
    return best_net

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    # Start with seed network
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    # Phase 1: Greedy removal from right
    network, _ = _greedy_remove(network, start_time, deadline, direction='right')
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Phase 2: Greedy removal from left on the result
    network2, _ = _greedy_remove(list(best_network), start_time, deadline, direction='left')
    if len(network2) < best_size:
        best_size = len(network2)
        best_network = list(network2)
    
    # Phase 3: Local search with swaps - try multiple random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Do several restarts with shuffled order
        num_restarts = min(10, int(remaining_time / 3))  # ~3s per restart
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Shuffle the network to try different removal orders
            shuffled = list(best_network)
            random.shuffle(shuffled)
            
            # First do greedy removal on shuffled
            net, _ = _greedy_remove(shuffled, start_time, deadline, direction='right')
            
            # Then try swaps
            net = _local_search_with_swaps(net, start_time, deadline)
            
            # Then try random replacements
            net = _random_comparator(net, start_time, deadline)
            
            if len(net) < best_size:
                best_size = len(net)
                best_network = list(net)
    
    return best_network

# EVOLVE-BLOCK-END
