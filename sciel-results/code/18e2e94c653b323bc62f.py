"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _apply_comparator(arr, i, j):
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]

def _apply_network(arr, network):
    for i, j in network:
        _apply_comparator(arr, i, j)

def _is_sorted_network(network, n_wires=N, n_tests=200):
    # Quick check: test random inputs
    for _ in range(n_tests):
        arr = np.random.randint(0, n_wires * 2, size=n_wires)
        _apply_network(arr, network)
        if not np.all(np.diff(arr) >= 0):
            return False
    # Also test zero-one principle with a subset of Boolean inputs
    for _ in range(100):
        arr = np.random.randint(0, 2, size=n_wires)
        _apply_network(arr, network)
        if not np.all(np.diff(arr) >= 0):
            return False
    return True

def _is_fully_sorted_network(network, n_wires=N):
    # Full zero-one principle check: all 2^16 = 65536 Boolean inputs
    for mask in range(1 << n_wires):
        arr = np.array([(mask >> i) & 1 for i in range(n_wires)], dtype=np.int8)
        _apply_network(arr, network)
        if not np.all(np.diff(arr) >= 0):
            return False
    return True

def _depth(network):
    """Compute parallel depth of a comparator network."""
    if not network:
        return 0
    # Build layers: a comparator (i,j) can start after both i and j are free
    # Simple layering: greedy
    in_use = set()
    depth = 0
    remaining = list(network)
    while remaining:
        next_layer = []
        still_remaining = []
        for comp in remaining:
            if comp[0] in in_use or comp[1] in in_use:
                still_remaining.append(comp)
            else:
                next_layer.append(comp)
        if not next_layer:
            # All remaining are blocked; just take them all (shouldn't happen if valid)
            depth += 1
            in_use = set()
            # Force progress
            next_layer = [still_remaining[0]]
            still_remaining = still_remaining[1:]
        depth += 1
        in_use = set()
        for comp in next_layer:
            in_use.add(comp[0])
            in_use.add(comp[1])
        remaining = still_remaining
    return depth

def _batcher_odd_even_merge_sort(n):
    """Generate a sorting network for n wires using Batcher's odd-even merge sort."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    left = _batcher_odd_even_merge_sort(half)
    right = _batcher_odd_even_merge_sort(half)
    
    # Offset right half
    right = [(i + half, j + half) for i, j in right]
    
    # Odd-even merge
    merged = _odd_even_merge(half, half, n)
    
    return left + right + merged

def _odd_even_merge(m, k, n):
    """Odd-even merge of two sorted halves of sizes m and k."""
    if m == 1 and k == 1:
        return [(0, 1)]
    if m == 1:
        # Merge single element into sorted list of size k
        return [(i, i + 1) for i in range(k - 1, -1, -1) if True]  # placeholder
    if k == 1:
        # Merge sorted list of size m into single element
        return [(i, i + 1) for i in range(m - 1)]  # placeholder
    
    # Standard odd-even merge
    comparators = []
    # Odd stage
    for i in range(0, n - 1, 2):
        if i < m and i + 1 < n:
            comparators.append((i, i + 1))
    # Even stage
    for i in range(1, n - 1, 2):
        if i < n:
            comparators.append((i, i + 1))
    return comparators

def _generate_initial_network():
    """Generate a reasonable initial sorting network."""
    # Start with odd-even transposition (known to work)
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    return network

def _random_comparator():
    i, j = random.sample(range(N), 2)
    if i > j:
        i, j = j, i
    return (i, j)

def _local_search(network, max_iters=5000, time_limit=80.0):
    """Local search to minimize comparator count."""
    start_time = time.time()
    best_network = list(network)
    best_score = len(best_network)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try random removal
        if len(best_network) > 1:
            idx = random.randint(0, len(best_network) - 1)
            candidate = list(best_network)
            candidate.pop(idx)
            if _is_sorted_network(candidate, N, n_tests=50):
                best_network = candidate
                best_score = len(best_network)
                continue
        
        # Try random swap
        if len(best_network) > 1:
            i, j = random.sample(range(len(best_network)), 2)
            candidate = list(best_network)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _is_sorted_network(candidate, N, n_tests=50):
                best_network = candidate
                continue
        
        # Try random insertion
        new_comp = _random_comparator()
        if new_comp not in best_network:
            candidate = list(best_network)
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, new_comp)
            if _is_sorted_network(candidate, N, n_tests=50):
                # Only accept if it helps (shouldn't, but maybe enables later removals)
                # For now, skip insertions that increase size
                pass
        
        # Try random replacement
        if len(best_network) > 1:
            idx = random.randint(0, len(best_network) - 1)
            new_comp = _random_comparator()
            candidate = list(best_network)
            candidate[idx] = new_comp
            if _is_sorted_network(candidate, N, n_tests=50):
                best_network = candidate
    
    return best_network

def _optimize_network(initial_network, time_limit=80.0):
    """Optimize the network using local search."""
    start_time = time.time()
    
    # First, try to remove redundant comparators
    network = list(initial_network)
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > time_limit - 5:
                break
            candidate = list(network)
            candidate.pop(idx)
            if _is_sorted_network(candidate, N, n_tests=100):
                network = candidate
                improved = True
    
    # Local search for further improvements
    if time.time() - start_time < time_limit - 5:
        network = _local_search(network, max_iters=2000, time_limit=time_limit - 5)
    
    return network

def construct():
    """Construct a sorting network for 16 wires with minimal comparators."""
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Start with a known-good network
    network = _generate_initial_network()
    
    # Optimize
    network = _optimize_network(network, time_limit=deadline - start_time - 2)
    
    # Final verification with full zero-one principle
    if time.time() - start_time < deadline - 1:
        if not _is_fully_sorted_network(network):
            # Fallback to known-good network
            network = _generate_initial_network()
    
    return network
# EVOLVE-BLOCK-END
