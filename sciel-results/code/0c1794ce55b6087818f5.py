"""Improved sorting network search for 17 wires using bitmask arithmetic."""
N = 17
import time

def _precompute_targets(n):
    """Precompute the target bitmask state for a sorted network.
    targets[i] is a bitmask where bit k is 1 if and only if the k-th input,
    when sorted, has a 1 at wire i.
    """
    targets = [0] * n
    for k in range(1 << n):
        c = bin(k).count('1')
        if c == 0:
            continue
        start_idx = n - c
        for i in range(start_idx, n):
            targets[i] |= (1 << k)
    return targets

def _initial_state(n):
    """Compute the initial bitmask state for all inputs.
    state[i] is a bitmask where bit k is 1 if and only if the k-th input
    has a 1 at wire i.
    """
    state = [0] * n
    for k in range(1 << n):
        for i in range(n):
            if (k >> i) & 1:
                state[i] |= (1 << k)
    return state

def _simulate_network(network, initial_state):
    """Simulate the network on the bitmask state."""
    state = initial_state[:]
    for i, j in network:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
    return state

def _is_sorting_network_fast(network, initial_state, targets):
    """Check if the network produces the target sorted state."""
    final_state = _simulate_network(network, initial_state)
    return final_state == targets

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Precompute constants
    targets = _precompute_targets(N)
    init_state = _initial_state(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, init_state, targets):
        return network
        
    # Local search: Greedy removal
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(network):
            if time.time() > deadline:
                return network
            
            # Try removing comparator at i
            new_network = network[:i] + network[i+1:]
            
            # Check if new network is valid
            if _is_sorting_network_fast(new_network, init_state, targets):
                network = new_network
                improved = True
            else:
                i += 1
                
    return network
