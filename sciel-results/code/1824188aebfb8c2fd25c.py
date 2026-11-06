import time
import numpy as np

N = 18
DEADLINE = 85.0  # Leave 5s margin for 90s limit

# EVOLVE-BLOCK-START

def _odd_even_transposition_network(n):
    """Generate an odd-even transposition sorting network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def construct():
    start_time = time.time()
    
    # Precompute all Boolean inputs
    all_inputs = np.arange(1 << N, dtype=np.int64)
    
    # Precompute sorted values
    sorted_values = np.array([(1 << N) - (1 << k) for k in range(N + 1)], dtype=np.int64)
    
    def is_sorted(inputs):
        """Check if all inputs in the array are sorted."""
        match = np.zeros(len(inputs), dtype=bool)
        for s in sorted_values:
            match |= (inputs == s)
        return np.all(match)

    def apply_network(inputs, comparators):
        """Apply a list of comparators to an array of bitmasks."""
        for (i, j) in comparators:
            mask_i = (inputs >> i) & 1
            mask_j = (inputs >> j) & 1
            swap_mask = mask_j & (~mask_i)
            inputs = inputs ^ (swap_mask << i) ^ (swap_mask << j)
        return inputs

    # Start with Odd-Even Transposition Network
    network = _odd_even_transposition_network(N)
    
    # Verify initial network
    if not is_sorted(apply_network(all_inputs.copy(), network)):
        # Should not happen
        return network

    best_network = network
    best_count = len(network)
    
    # Optimization Loop
    changed = True
    while changed:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            candidate = network[:idx] + network[idx+1:]
            
            result = apply_network(all_inputs.copy(), candidate)
            if is_sorted(result):
                network = candidate
                changed = True
                best_network = network
                best_count = len(network)
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    return best_network

# EVOLVE-BLOCK-END
