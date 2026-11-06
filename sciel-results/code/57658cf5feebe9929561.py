"""Size-efficient sorting networks on 17 wires – search-based construction."""

import time
import random
import math

N = 17

# Deadline guard: kill search well before 90s wall-clock limit
_START_TIME = time.time()
_DEADLINE = 85.0  # seconds; leaves 5s safety margin

def _now():
    return time.time()

def _time_exceeded():
    return (_now() - _START_TIME) >= _DEADLINE

def _odd_even_transposition(n=N):
    """Standard odd-even transposition sorting network."""
    comp = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comp.append((i, i + 1))
    return comp

def _verify_booleans(network):
    """
    Verify correctness via zero-one principle using big integers.
    Each wire is a big int of 2^N bits.
    """
    total = 1 << N
    mask_all = (1 << total) - 1
    
    # Initialize wires
    wires = [0] * N
    
    # Cache masks for initialization
    if not hasattr(_verify_booleans, '_masks'):
        _verify_booleans._masks = [0] * N
        for k in range(N):
            block = 1 << k
            period = 2 * block
            num_periods = total // period
            v = 0
            for p in range(num_periods):
                v |= ((1 << block) - 1) << (p * period + block)
            _verify_booleans._masks[k] = v
            
    for k in range(N):
        wires[k] = _verify_booleans._masks[k]
        
    # Apply comparators
    for (i, j) in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
        
    # Check sorted
    for k in range(N - 1):
        if wires[k] & ~wires[k + 1] & mask_all:
            return False
    return True

def _depth_of_network(network, n=N):
    """Compute the depth of the network."""
    if not network:
        return 0
    last_use = [-1] * n
    depth = [0] * len(network)
    for idx in range(len(network)):
        i, j = network[idx]
        d = 0
        if last_use[i] >= 0:
            d = max(d, depth[last_use[i]] + 1)
        if last_use[j] >= 0:
            d = max(d, depth[last_use[j]] + 1)
        depth[idx] = d
        last_use[i] = idx
        last_use[j] = idx
    return max(depth) + 1

def construct():
    """
    Construct a sorting network on N=17 wires.
    Uses Simulated Annealing to minimize comparator count.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with Odd-Even Transposition
    current_network = _odd_even_transposition(N)
    current_len = len(current_network)
    
    # Verify base
    if not _verify_booleans(current_network):
        # Should not happen
        return current_network
        
    best_network = list(current_network)
    best_len = current_len
    
    rng = random.Random(42)
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    # We will run SA until time is up.
    # Each step: generate neighbor, verify, accept/reject.
    
    while not _time_exceeded():
        # Generate neighbor
        move_type = rng.random()
        neighbor = list(current_network)
        
        if move_type < 0.7 and len(neighbor) > 0:
            # Remove a random comparator
            idx = rng.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
        elif move_type < 0.9 and len(neighbor) > 1:
            # Swap two random comparators
            i = rng.randint(0, len(neighbor) - 1)
            j = rng.randint(0, len(neighbor) - 1)
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Insert a random comparator
            # Choose random i, j
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(neighbor))
            neighbor.insert(pos, (i, j))
            
        # Verify neighbor
        if _verify_booleans(neighbor):
            new_len = len(neighbor)
            delta = new_len - len(current_network)
            
            if delta < 0:
                # Improvement
                current_network = neighbor
                if new_len < best_len:
                    best_network = neighbor
                    best_len = new_len
            else:
                # Worse or same, accept with probability
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        current_network = neighbor
        else:
            # Invalid, reject
            pass
            
        # Cool down
        T *= cooling_rate
        
    return best_network
