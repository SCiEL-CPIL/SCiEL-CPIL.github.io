"""Size-efficient sorting networks on 17 wires."""
import time
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131072

# Precompute all boolean inputs as a uint8 matrix (131072 x 17)
_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        # Generate all 2^17 bit patterns as a matrix
        # Each row is a 17-bit pattern, column i is bit i
        _ALL_INPUTS = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
        for i in range(N):
            # Bit i of each number from 0 to 2^17-1
            pattern = np.zeros(NUM_INPUTS, dtype=np.uint8)
            for k in range(0, NUM_INPUTS, 1 << (i + 1)):
                start = k + (1 << i)
                end = min(k + (1 << (i + 1)), NUM_INPUTS)
                pattern[start:end] = 1
            _ALL_INPUTS[:, i] = pattern
    return _ALL_INPUTS


def _simulate(network, inputs=None):
    """Simulate a sorting network on all boolean inputs.
    Returns True if sorted (each wire i <= wire i+1 for all i).
    """
    if inputs is None:
        inputs = _get_all_inputs()
    
    state = inputs.copy()
    for i, j in network:
        # Compare-exchange: min goes to i, max goes to j
        # For boolean: if state[:,i] > state[:,j], swap
        a = state[:, i]
        b = state[:, j]
        # min = a & b (for booleans, min is AND), max = a | b (OR)
        state[:, i] = a & b
        state[:, j] = a | b
    
    # Check if sorted: state[:, i] <= state[:, i+1] for all i
    # For booleans: not (state[:,i] == 1 and state[:,i+1] == 0)
    for i in range(N - 1):
        if np.any(state[:, i] > state[:, i + 1]):
            return False
    return True


def _is_sorted_network(network):
    """Check if a network sorts all boolean inputs."""
    return _simulate(network)


def _odd_even_transposition():
    """Generate odd-even transposition sorting network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network


def _systematic_deletion(network):
    """Try to remove each comparator in order, repeating until no more can be removed."""
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(network):
            # Try removing comparator at position i
            candidate = network[:i] + network[i+1:]
            if _is_sorted_network(candidate):
                network = candidate
                improved = True
                # Don't increment i; try to remove the next one at the same position
            else:
                i += 1
    return network


def _random_deletion(network, num_trials=2000):
    """Try random deletions to find more removable comparators."""
    import random
    improved = True
    while improved:
        improved = False
        for _ in range(num_trials):
            idx = random.randint(0, len(network) - 1)
            candidate = network[:idx] + network[idx+1:]
            if _is_sorted_network(candidate):
                network = candidate
                improved = True
                break
    return network


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    # Start with odd-even transposition network
    network = _odd_even_transposition()
    
    # First pass: systematic deletion
    while time.time() < deadline:
        network = _systematic_deletion(network)
        # After systematic deletion, try random deletion for a bit
        if time.time() < deadline - 5:
            network = _random_deletion(network, num_trials=500)
            # Then do another systematic pass
            network = _systematic_deletion(network)
        else:
            break
    
    return network
