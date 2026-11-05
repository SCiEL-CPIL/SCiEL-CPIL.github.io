import numpy as np
import time

N = 18
NUM_BOOL_INPUTS = 2 ** N

def _apply_network(network, x):
    """Apply network to a single input vector x (list of 0/1)."""
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted(x):
    """Check if x is sorted in non-decreasing order."""
    for i in range(N - 1):
        if x[i] > x[i + 1]:
            return False
    return True

def _verify_network(network):
    """Verify network sorts all 2^N Boolean inputs."""
    for mask in range(NUM_BOOL_INPUTS):
        x = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, x)
        if not _is_sorted(result):
            return False
    return True

def _batcher_bitonic_sort(n):
    """Generate Batcher bitonic sort network for n wires."""
    def bitonic_sort(lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return []
        mid = (lo + hi) // 2
        left_comps = bitonic_sort(lo, mid, direction)
        right_comps = bitonic_sort(mid, hi, 1 - direction)
        merge_comps = bitonic_merge(lo, hi, direction)
        return left_comps + right_comps + merge_comps
    
    def bitonic_merge(lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return []
        step = length // 2
        comparators = []
        for i in range(step):
            a = lo + i
            b = lo + i + step
            if direction == 0:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        mid = lo + step
        left_comps = bitonic_merge(lo, mid, direction)
        right_comps = bitonic_merge(mid, hi, direction)
        return comparators + left_comps + right_comps
    
    return bitonic_sort(0, n, 0)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    # Start with Batcher bitonic sort
    network = _batcher_bitonic_sort(N)
    
    # Local search: try removing comparators
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx+1:]
            if _verify_network(candidate):
                network = candidate
                improved = True
                break
    
    return network
