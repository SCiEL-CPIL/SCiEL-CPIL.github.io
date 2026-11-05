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

def _batch_try_removals(network, candidates, all_inputs):
    """Try removing each candidate index. Returns list of indices that can be removed."""
    removable = []
    for idx in candidates:
        if idx >= len(network):
            continue
        new_network = network[:idx] + network[idx+1:]
        final = _simulate_network(new_network, all_inputs)
        if _is_sorted_fast(final):
            removable.append(idx)
    return removable

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators using batch verification for efficiency."""
    network = list(network)
    n_comparators = len(network)
    
    # Strategy: Try removing multiple comparators at once when possible
    # First, do a single-removal pass to find removable comparators
    removable_set = set()
    
    # Try from the end first
    for i in range(len(network) - 1, -1, -1):
        if time.time() > deadline:
            break
        if _try_remove_comparator(network, i, all_inputs):
            removable_set.add(i)
    
    # Try from the beginning
    for i in range(len(network)):
        if time.time() > deadline:
            break
        if _try_remove_comparator(network, i, all_inputs):
            removable_set.add(i)
    
    # Now try to remove all identified removable comparators
    # Sort by index descending to avoid index shifting issues
    removable_indices = sorted(removable_set, reverse=True)
    removed_count = 0
    
    for idx in removable_indices:
        if time.time() > deadline:
            break
        if idx < len(network):
            # Verify it's still removable (network may have changed)
            if _try_remove_comparator(network, idx, all_inputs):
                network.pop(idx)
                removed_count += 1
    
    # Continue with iterative single removals if time permits
    improved = True
    while improved:
        improved = False
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break
        if not improved:
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
