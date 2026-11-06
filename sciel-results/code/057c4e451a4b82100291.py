import time
import random
import numpy as np

N = 17
DEADLINE = time.time() + 85  # 5s margin before 90s limit

# Precompute all 2^N Boolean inputs
# Shape: (2^N, N)
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = _make_all_inputs()

def _is_valid(network):
    """
    Check if the network sorts all Boolean inputs.
    network: list of (i, j) tuples
    """
    # Copy inputs to avoid modifying the original
    current = _ALL_INPUTS.copy()
    
    for i, j in network:
        col_i = current[:, i]
        col_j = current[:, j]
        
        # Perform compare-exchange
        current[:, i] = np.minimum(col_i, col_j)
        current[:, j] = np.maximum(col_i, col_j)
        
    # Check if sorted
    diffs = np.diff(current, axis=1)
    return np.all(diffs >= 0)

def _reduce_network(network):
    """
    Greedily remove redundant comparators.
    Returns the reduced network.
    """
    net = list(network)
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        random.shuffle(net)
        
        i = 0
        while i < len(net):
            if time.time() > DEADLINE:
                return net
            
            comp = net[i]
            candidate = net[:i] + net[i+1:]
            
            if _is_valid(candidate):
                net = candidate
                improved = True
            else:
                i += 1
                
    return net

def construct():
    # 1. Generate Seed: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            seed_network.append((i, i + 1))
            
    # 2. Initial Reduction
    best_network = _reduce_network(seed_network)
    best_len = len(best_network)
    
    # 3. Local Search with Perturbation
    while time.time() < DEADLINE - 2:  # Keep 2s buffer
        # Perturb the current best
        perturbed = list(best_network)
        
        # Generate a random comparator
        i, j = random.sample(range(N), 2)
        if i > j: i, j = j, i
        
        # Insert at random position
        pos = random.randint(0, len(perturbed))
        perturbed.insert(pos, (i, j))
        
        # Reduce the perturbed network
        candidate = _reduce_network(perturbed)
        
        # Validate candidate before accepting
        if len(candidate) < best_len and _is_valid(candidate):
            best_network = candidate
            best_len = len(candidate)
            
        # Try a swap if time permits
        if len(best_network) > 1 and time.time() < DEADLINE - 2:
             p1 = random.randint(0, len(best_network) - 1)
             p2 = random.randint(0, len(best_network) - 1)
             if p1 != p2:
                 swap_net = list(best_network)
                 swap_net[p1], swap_net[p2] = swap_net[p2], swap_net[p1]
                 # Check if the swapped network is valid
                 if _is_valid(swap_net):
                     candidate = _reduce_network(swap_net)
                     if len(candidate) < best_len and _is_valid(candidate):
                         best_network = candidate
                         best_len = len(candidate)

    return best_network
