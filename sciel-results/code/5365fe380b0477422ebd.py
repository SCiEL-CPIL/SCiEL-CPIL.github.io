"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import product

N = 16

# EVOLVE-BLOCK-START
def construct():
    """Construct a sorting network for 16 wires with minimal comparators."""
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Step 1: Generate Batcher's odd-even merge sort network
    network = _batcher_sort(N)
    
    # Step 2: Remove redundant comparators
    network = _remove_redundant_comparators(network, deadline)
    
    return network


def _batcher_sort(n):
    """Batcher's odd-even merge sort for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left_network = _batcher_sort(half)
    right_network = _batcher_sort(half)
    
    network = []
    for (i, j) in left_network:
        network.append((i, j))
    for (i, j) in right_network:
        network.append((i + half, j + half))
    
    merge_network = _odd_even_merge(half)
    network.extend(merge_network)
    
    return network


def _odd_even_merge(n):
    """Odd-even merge two sorted sequences of length n."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comparators = []
    
    if n % 2 == 0:
        for i in range(0, n - 1, 2):
            comparators.append((i, i + 1))
    
    half = n // 2
    left = _odd_even_merge(half)
    comparators.extend(left)
    
    right = _odd_even_merge(half)
    for (a, b) in right:
        comparators.append((a + half, b + half))
    
    if n % 2 == 0:
        for i in range(1, n - 1, 2):
            comparators.append((i, i + 1))
    
    return comparators


def _verify_network(network, deadline=None):
    """Verify network sorts all Boolean inputs. Returns True, False, or None (timeout)."""
    for bits in product([0, 1], repeat=N):
        if deadline is not None and time.time() > deadline:
            return None
        arr = list(bits)
        for (i, j) in network:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        if arr != sorted(arr):
            return False
    return True


def _remove_redundant_comparators(network, deadline):
    """Remove redundant comparators from network."""
    network = list(network)
    i = 0
    while i < len(network) and time.time() < deadline:
        candidate = network[:i] + network[i+1:]
        result = _verify_network(candidate, deadline)
        if result is True:
            network = candidate
        elif result is False:
            i += 1
        else:
            break
    
    return network

# EVOLVE-BLOCK-END
