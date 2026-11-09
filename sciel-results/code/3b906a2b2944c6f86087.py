"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import numpy as np

def _batcher_odd_even_merge_sort(n):
    """Batcher's odd-even merge sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    left = _batcher_odd_even_merge_sort(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(half)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge network for 2n wires (n from each half)."""
    if n == 1:
        return [(0, 1)]
    even_merge = _odd_even_merge(n // 2)
    odd_merge = _odd_even_merge(n // 2)
    even_translated = [(2*i, 2*j) for (i, j) in even_merge]
    odd_translated = [(2*i + 1, 2*j + 1) for (i, j) in odd_merge]
    final_merge = [(2*i + 1, 2*i + 2) for i in range(n - 1)]
    return even_translated + odd_translated + final_merge

def _verify_network_fast(comparators, n):
    """Fast verification using numpy vectorization over all 2^n inputs."""
    num_inputs = 1 << n
    # Create array of shape (num_inputs, n) with all binary inputs
    inputs = np.zeros((num_inputs, n), dtype=np.uint8)
    for i in range(n):
        inputs[:, i] = (np.arange(num_inputs) >> i) & 1
    
    wires = inputs.copy()
    for (i, j) in comparators:
        # Compare-exchange: min to wire i, max to wire j
        wi = wires[:, i].copy()
        wj = wires[:, j].copy()
        # After compare: wire i gets min, wire j gets max
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
    
    # Check if all rows are sorted
    for k in range(n - 1):
        if np.any(wires[:, k] > wires[:, k + 1]):
            return False
    return True

def _remove_redundant(comparators, n, deadline):
    """Remove redundant comparators using fast verification."""
    comparators = list(comparators)
    i = 0
    while i < len(comparators) and time.time() < deadline:
        candidate = comparators[:i] + comparators[i+1:]
        if _verify_network_fast(candidate, n):
            comparators = candidate
        else:
            i += 1
    return comparators

def _try_swap_order(comparators, n, deadline):
    """Try swapping adjacent comparators to see if it enables further pruning."""
    comparators = list(comparators)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(comparators) - 1):
            if time.time() >= deadline:
                break
            # Try swapping comparators[i] and comparators[i+1]
            candidate = comparators[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _verify_network_fast(candidate, n):
                # Check if this enables removal of a later comparator
                for j in range(i + 2, len(candidate)):
                    if time.time() >= deadline:
                        break
                    test = candidate[:j] + candidate[j+1:]
                    if _verify_network_fast(test, n):
                        comparators = test
                        improved = True
                        break
                if not improved:
                    # Just accept the swap if it's valid
                    comparators = candidate
                    improved = True
                    break
    return comparators

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Build Batcher's odd-even merge sort network
    network = _batcher_odd_even_merge_sort(N)
    
    # Verify it's correct
    if not _verify_network_fast(network, N):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Remove redundant comparators (multiple passes)
    pruned = network
    for _ in range(5):
        if time.time() >= deadline:
            break
        new_pruned = _remove_redundant(pruned, N, deadline)
        if len(new_pruned) == len(pruned):
            break
        pruned = new_pruned
    
    # Final verification
    if not _verify_network_fast(pruned, N):
        return network
    
    return pruned
# EVOLVE-BLOCK-END
