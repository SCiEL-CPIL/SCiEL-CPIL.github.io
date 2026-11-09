import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N

# Pre-compute initial masks
# masks[w] is an integer where bit k is 1 if input k has a 1 on wire w
_INITIAL_MASKS = None

def _get_initial_masks():
    global _INITIAL_MASKS
    if _INITIAL_MASKS is None:
        _INITIAL_MASKS = [0] * N
        for w in range(N):
            mask = 0
            # Optimization: iterate through set bits?
            # Or just loop k. 131072 * 17 is ~2.2M ops. Fast enough once.
            for k in range(TOTAL_INPUTS):
                if (k >> w) & 1:
                    mask |= (1 << k)
            _INITIAL_MASKS[w] = mask
    return _INITIAL_MASKS

def _verify_network(network):
    """
    Verify if the network sorts all inputs.
    Returns True if valid, False otherwise.
    """
    if not network:
        return N <= 1 # Trivial case
    
    masks = _get_initial_masks()[:]
    
    for i, j in network:
        if i >= j:
            return False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
        
    # Check sortedness
    for w in range(N - 1):
        if masks[w] & ~masks[w + 1]:
            return False
            
    return True

def _build_seed_network():
    """
    Build a seed sorting network using odd-even transposition.
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

def _greedy_prune(network):
    """
    Remove any comparator that can be removed while keeping the network valid.
    Returns the pruned network and the number of removals.
    """
    pruned = list(network)
    removed = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(pruned) - 1, -1, -1):
            candidate = pruned[:i] + pruned[i+1:]
            if _verify_network(candidate):
                pruned = candidate
                removed += 1
                changed = True
                break
    return pruned, removed

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Start with seed
    network = _build_seed_network()
    
    # Initial greedy prune
    network, _ = _greedy_prune(network)
    best_network = list(network)
    best_size = len(network)
    
    # Simulated Annealing / Random Walk
    current_size = best_size
    temperature = 1.0
    cooling_rate = 0.999
    
    # We will perform random moves on the current network.
    # We only keep the network if it is valid.
    # We accept worse solutions (larger size) with probability exp(-delta/T).
    
    while time.time() < deadline:
        # Generate a random move
        move_type = random.random()
        
        new_network = None
        
        if move_type < 0.4:
            # Try to remove a random comparator
            if current_size > 0:
                idx = random.randint(0, current_size - 1)
                new_network = network[:idx] + network[idx+1:]
                
        elif move_type < 0.6:
            # Try to add a random comparator
            # Insert at random position
            idx = random.randint(0, current_size)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_network = network[:idx] + [(i, j)] + network[idx:]
            
        elif move_type < 0.8:
            # Swap two random comparators
            if current_size >= 2:
                i1 = random.randint(0, current_size - 1)
                i2 = random.randint(0, current_size - 1)
                if i1 != i2:
                    new_network = network[:]
                    new_network[i1], new_network[i2] = new_network[i2], new_network[i1]
                    
        else:
            # Move a random comparator to a random position
            if current_size >= 2:
                i1 = random.randint(0, current_size - 1)
                i2 = random.randint(0, current_size - 1)
                if i1 != i2:
                    new_network = network[:]
                    comp = new_network.pop(i1)
                    # Adjust index if i1 < i2
                    if i1 < i2:
                        i2 -= 1
                    new_network.insert(i2, comp)

        if new_network is None:
            continue
            
        # Verify new network
        if not _verify_network(new_network):
            continue
            
        new_size = len(new_network)
        delta = new_size - current_size
        
        # Acceptance criterion
        if delta <= 0:
            # Better or equal, always accept
            network = new_network
            current_size = new_size
            
            # Greedy prune after improvement
            network, _ = _greedy_prune(network)
            current_size = len(network)
            
            if current_size < best_size:
                best_size = current_size
                best_network = list(network)
                # Update temperature if we found a new best? 
                # Or just let it cool.
                
        else:
            # Worse, accept with probability exp(-delta / T)
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    network = new_network
                    current_size = new_size
                    # Do NOT prune here, as we want to allow temporary bloat
                    # to escape local minima. But maybe a light prune?
                    # Let's keep it unpruned to allow structural changes.
                    
        # Cool down
        temperature *= cooling_rate
        
    return best_network
