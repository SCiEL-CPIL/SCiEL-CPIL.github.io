"""Improved sorting network search with structural changes."""
N = 17
import time
import numpy as np
import random

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

def _greedy_remove(network, all_inputs, deadline):
    """Aggressively remove any removable comparator in random order."""
    network = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            if _try_remove_comparator(network, idx, all_inputs):
                network.pop(idx)
                improved = True
                break
    return network

def _swap_remove_search(network, all_inputs, deadline):
    """Try swapping pairs of comparators and then removing one."""
    network = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(network)
        if n < 2:
            break
        # Try random pairs of swaps
        attempts = 0
        while attempts < 50 and time.time() < deadline:
            attempts += 1
            i = random.randint(0, n - 2)
            j = random.randint(0, n - 2)
            if i == j:
                continue
            new_network = network[:]
            new_network[i], new_network[j] = new_network[j], new_network[i]
            if _is_sorting_network_fast(new_network, all_inputs):
                # Try removing one of the swapped comparators
                for rem in [i, j]:
                    if rem >= len(new_network):
                        continue
                    test = new_network[:rem] + new_network[rem+1:]
                    if _is_sorting_network_fast(test, all_inputs):
                        network = test
                        improved = True
                        break
                if improved:
                    break
    return network

def _construct_base_network():
    """Construct a compact base sorting network using bitonic-like structure."""
    # Use a known efficient construction: 
    # Start with a small number of parallel compare-exchange stages
    # This is a simplified version of a batcher-like network
    
    network = []
    
    # Stage 1: Compare adjacent pairs (like first pass of odd-even)
    for i in range(0, N - 1, 2):
        network.append((i, i + 1))
    for i in range(1, N - 1, 2):
        network.append((i, i + 1))
    
    # Stage 2: Compare pairs with distance 2
    for i in range(0, N - 2, 2):
        network.append((i, i + 2))
    for i in range(1, N - 2, 2):
        network.append((i, i + 2))
    
    # Stage 3: Compare pairs with distance 4
    for i in range(0, N - 4, 2):
        network.append((i, i + 4))
    for i in range(1, N - 4, 2):
        network.append((i, i + 4))
    
    # Stage 4: Compare pairs with distance 8
    for i in range(0, N - 8, 2):
        network.append((i, i + 8))
    for i in range(1, N - 8, 2):
        network.append((i, i + 8))
    
    # Add more passes to ensure correctness
    # This is a heuristic; we'll verify and fix
    # Add several more rounds of adjacent comparisons
    for _ in range(4):
        for i in range(0, N - 1, 2):
            network.append((i, i + 1))
        for i in range(1, N - 1, 2):
            network.append((i, i + 1))
    
    return network

def _make_valid_base(all_inputs):
    """Create a guaranteed-valid base network (odd-even transposition)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safe margin
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Start with guaranteed valid odd-even transposition
    best_network = _make_valid_base(all_inputs)
    if not _is_sorting_network_fast(best_network, all_inputs):
        return best_network
    
    best_len = len(best_network)
    
    # First pass: aggressive greedy removal
    deadline1 = start_time + 15.0
    network = _greedy_remove(best_network, all_inputs, deadline1)
    if _is_sorting_network_fast(network, all_inputs) and len(network) < best_len:
        best_network = network
        best_len = len(network)
    
    # Second pass: swap-remove search
    deadline2 = start_time + 30.0
    network = _swap_remove_search(best_network, all_inputs, deadline2)
    if _is_sorting_network_fast(network, all_inputs) and len(network) < best_len:
        best_network = network
        best_len = len(network)
    
    # Third pass: more aggressive greedy removal
    deadline3 = start_time + 40.0
    network = _greedy_remove(best_network, all_inputs, deadline3)
    if _is_sorting_network_fast(network, all_inputs) and len(network) < best_len:
        best_network = network
        best_len = len(network)
    
    # Final pass: one more round of removal
    if time.time() < deadline - 2.0:
        deadline4 = deadline - 1.0
        network = _greedy_remove(best_network, all_inputs, deadline4)
        if _is_sorting_network_fast(network, all_inputs) and len(network) < best_len:
            best_network = network
            best_len = len(network)
    
    return best_network
# EVOLVE-BLOCK-END
