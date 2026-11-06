"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _all_boolean_inputs(n):
    """Generate all 2^n Boolean inputs as a numpy array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.uint64)
    wires = np.zeros((total, n), dtype=np.uint8)
    for i in range(n):
        wires[:, i] = (indices >> i) & 1
    return wires

def _batcher_merge_sort(n):
    """Constructs Batcher's odd-merge sorting network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    
    comparators = []
    n_odd = (n + 1) // 2
    n_even = n // 2
    
    sub_odd = _batcher_merge_sort(n_odd)
    for i, j in sub_odd:
        comparators.append((2*i, 2*j))
    
    sub_even = _batcher_merge_sort(n_even)
    for i, j in sub_even:
        comparators.append((2*i + 1, 2*j + 1))
    
    if n % 2 == 0:
        m = n // 2
        for i in range(0, m, 2):
            comparators.append((i, i + m))
        
        sub_merge_odd = _batcher_merge_sort(m)
        for i, j in sub_merge_odd:
            comparators.append((2*i, 2*j))
        
        sub_merge_even = _batcher_merge_sort(m)
        for i, j in sub_merge_even:
            comparators.append((2*i + 1, 2*j + 1))
    else:
        m = n // 2
        for i in range(0, 2*m, 2):
            if i + m < n:
                comparators.append((i, i + m))
        
        sub_merge_odd = _batcher_merge_sort((n + 1) // 2)
        for i, j in sub_merge_odd:
            comparators.append((2*i, 2*j))
        
        sub_merge_even = _batcher_merge_sort(n // 2)
        for i, j in sub_merge_even:
            comparators.append((2*i + 1, 2*j + 1))
    
    return comparators

def _simulate_network(network, inputs):
    """Simulate a network on a set of inputs. Returns boolean array of whether each is sorted."""
    arr = inputs.copy()
    for i, j in network:
        a = arr[:, i].copy()
        b = arr[:, j].copy()
        swap_mask = a > b
        arr[:, i] = np.where(swap_mask, b, a)
        arr[:, j] = np.where(swap_mask, a, b)
    
    # Check if sorted: non-decreasing
    return np.all(arr[:, :-1] <= arr[:, 1:], axis=1)

def construct():
    start_time = time.time()
    deadline = 85
    
    # Start with Batcher's network
    network = _batcher_merge_sort(N)
    
    # Generate all Boolean inputs
    bool_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    is_sorted = _simulate_network(network, bool_inputs)
    if not np.all(is_sorted):
        # Should not happen, but fallback
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Try to remove redundant comparators
    # For each comparator, check if removing it still results in a correct network
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > deadline:
                break
            
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            
            # Check if still correct
            is_sorted = _simulate_network(test_network, bool_inputs)
            
            if np.all(is_sorted):
                # Can remove this comparator
                network = test_network
                improved = True
    
    return network
# EVOLVE-BLOCK-END

