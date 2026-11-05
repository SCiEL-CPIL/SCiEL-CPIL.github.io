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

def _is_sorted_fast(state):
    """Check if all rows are sorted (non-decreasing) using vectorized ops."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _simulate_partial(network, all_inputs, skip_idx=None):
    """Simulate network, optionally skipping one comparator. Returns final state."""
    state = all_inputs.copy()
    for k, (i, j) in enumerate(network):
        if k == skip_idx:
            continue
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators using efficient verification."""
    network = list(network)
    
    # Track best network found
    best_network = network[:]
    best_len = len(network)
    
    # Strategy: Repeatedly scan for removable comparators, removing them greedily
    # Use reverse order first (later comparators are often redundant)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try from end to start
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                if len(network) < best_len:
                    best_len = len(network)
                    best_network = network[:]
                break  # Restart scan after a removal
    
    # Second pass: try from start to end
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(network)):
            if time.time() > deadline:
                break
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                if len(network) < best_len:
                    best_len = len(network)
                    best_network = network[:]
                break  # Restart scan after a removal
    
    # Third pass: try all positions again (bidirectional)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                if len(network) < best_len:
                    best_len = len(network)
                    best_network = network[:]
                break
        if not improved:
            for i in range(len(network)):
                if time.time() > deadline:
                    break
                if _try_remove_comparator(network, i, all_inputs):
                    network.pop(i)
                    improved = True
                    if len(network) < best_len:
                        best_len = len(network)
                        best_network = network[:]
                    break
    
    return network if len(network) <= best_len else best_network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    # Additional optimization: try to swap adjacent comparators to enable further removals
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(network) - 1):
                if time.time() > deadline - 5.0:
                    break
                # Try swapping network[i] and network[i+1]
                if network[i] != network[i+1]:
                    new_network = network.copy()
                    new_network[i], new_network[i+1] = new_network[i+1], new_network[i]
                    if _is_sorting_network_fast(new_network, all_inputs):
                        # Check if this swap enables removal of one of them
                        for remove_idx in [i, i+1]:
                            test_network = new_network[:remove_idx] + new_network[remove_idx+1:]
                            if _is_sorting_network_fast(test_network, all_inputs):
                                network = test_network
                                improved = True
                                break
                        if improved:
                            break
    
    return network
# EVOLVE-BLOCK-END
