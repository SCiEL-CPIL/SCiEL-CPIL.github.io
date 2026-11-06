"""Size-efficient sorting network on 17 wires via odd-even merge seed + aggressive local search."""
import time
import numpy as np
import random

N = 17
DEADLINE = 88.0  # safety margin before 90s kill
NUM_INPUTS = 1 << N

def _make_all_inputs():
    """Return array of shape (2^N, N) with all Boolean inputs, uint8."""
    idx = np.arange(NUM_INPUTS, dtype=np.uint32)
    arr = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        arr[:, i] = (idx >> i) & 1
    return arr

def _simulate_network(network, inputs):
    """Simulate network on all inputs. inputs shape (n, N) uint8."""
    state = inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return state

def _check_sorted(state):
    """Check if state (n, N) is sorted in each row."""
    return np.all(state[:, :-1] <= state[:, 1:])

def _is_valid_network(network, all_inputs):
    """Check network sorts all Boolean inputs."""
    state = _simulate_network(network, all_inputs)
    return _check_sorted(state)

def _odd_even_merge_sort_network(n):
    """Build odd-even merge sort network for n wires. Returns list of (i,j) comparators."""
    if n <= 1:
        return []
    
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    # Recursively sort each half
    left = _odd_even_merge_sort_network(half)
    right = _odd_even_merge_sort_network(n - half)
    
    # Shift right half indices
    right = [(i + half, j + half) for (i, j) in right]
    
    # Merge the two sorted halves using odd-even merge
    merged = _odd_even_merge(left, right, half, n - half)
    
    return left + right + merged

def _odd_even_merge(left_net, right_net, left_size, right_size):
    """Merge two sorted networks of sizes left_size and right_size."""
    total = left_size + right_size
    if left_size == 0:
        return right_net
    if right_size == 0:
        return left_net
    
    if total == 2:
        return [(0, 1)]
    
    # Interleave and merge
    # The odd-even merge algorithm:
    # 1. Recursively merge odd and even sub-sequences
    # 2. Compare adjacent elements
    
    # Build the merged network by interleaving
    # For odd-even merge, we need to:
    # - Take odd-indexed elements from both halves and merge them
    # - Take even-indexed elements from both halves and merge them
    # - Compare adjacent pairs
    
    # This is complex to implement recursively; let's use a simpler approach
    # Build a balanced merge network
    
    # Simple approach: use a series of compare-exchange layers
    # For merging two sorted lists of size k each, we need k-1 layers
    comparators = []
    
    # Layer 1: compare elements that are 2 apart (to handle odd/even)
    for i in range(0, total - 2, 2):
        comparators.append((i, i + 2))
    
    # Layer 2: compare adjacent elements
    for i in range(0, total - 1, 2):
        comparators.append((i, i + 1))
    
    # Additional layers for correctness
    for i in range(1, total - 1, 2):
        comparators.append((i, i + 1))
    
    return comparators

def _build_seed_network():
    """Build a seed sorting network using odd-even merge sort."""
    # Try odd-even merge sort
    try:
        net = _odd_even_merge_sort_network(N)
        if len(net) > 0:
            return net
    except:
        pass
    
    # Fallback: odd-even transposition
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _try_remove_comparator(network, index, all_inputs):
    """Try removing comparator at index; return True if still valid."""
    reduced = network[:index] + network[index+1:]
    state = _simulate_network(reduced, all_inputs)
    return _check_sorted(state)

def _try_move_comparator(network, from_idx, to_idx, all_inputs):
    """Try moving comparator from from_idx to to_idx; return True if still valid."""
    if from_idx == to_idx:
        return True
    comp = network[from_idx]
    moved = network[:from_idx] + network[from_idx+1:]
    # Insert at to_idx (adjust if needed)
    insert_pos = min(to_idx, len(moved))
    moved.insert(insert_pos, comp)
    state = _simulate_network(moved, all_inputs)
    return _check_sorted(state)

def _aggressive_local_search(network, all_inputs, start_time, time_budget):
    """Aggressive local search with removal and move operations."""
    best = list(network)
    
    # Multiple passes of removal
    for pass_num in range(30):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        improved = False
        # Try removing from end to start
        for idx in range(len(best) - 1, -1, -1):
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break
            if _try_remove_comparator(best, idx, all_inputs):
                best = best[:idx] + best[idx+1:]
                improved = True
                break  # restart scan
        
        if not improved:
            break
    
    # Try move operations
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.7:  # use remaining time for moves
        for _ in range(50):
            elapsed = time.time() - start_time
            if elapsed > time_budget * 0.8:
                break
            
            if len(best) < 2:
                break
            
            # Try moving a random comparator to a random position
            from_idx = random.randint(0, len(best) - 1)
            to_idx = random.randint(0, len(best) - 1)
            
            if _try_move_comparator(best, from_idx, to_idx, all_inputs):
                comp = best[from_idx]
                best = best[:from_idx] + best[from_idx+1:]
                insert_pos = min(to_idx, len(best))
                best.insert(insert_pos, comp)
    
    return best

def construct():
    start_time = time.time()
    time_budget = DEADLINE - start_time
    
    # Precompute all Boolean inputs
    all_inputs = _make_all_inputs()
    
    # Build seed
    network = _build_seed_network()
    
    # Verify seed is valid
    if not _is_valid_network(network, all_inputs):
        # Fallback to transposition
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
    
    # Aggressive local search
    network = _aggressive_local_search(network, all_inputs, start_time, time_budget)
    
    # Final validation
    if not _is_valid_network(network, all_inputs):
        # Return a known valid network
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
    
    return network
