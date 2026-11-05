"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17
import time
import numpy as np

def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)

def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs."""
    state = inputs.copy()
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorting_network_fast(network, all_inputs):
    """Faster check: verify each row is of form 0...01...1 using vectorized ops"""
    final = _simulate_network(network, all_inputs)
    diffs = np.diff(final, axis=1)
    invalid_rows = np.any(diffs < 0, axis=1)
    return not np.any(invalid_rows)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators."""
    network = list(network)
    improved = True
    while improved:
        improved = False
        for i in range(len(network)):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    all_inputs = _all_boolean_inputs(N)
    
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    network = _local_search_remove(network, all_inputs, deadline)
    
    return network
# EVOLVE-BLOCK-END
