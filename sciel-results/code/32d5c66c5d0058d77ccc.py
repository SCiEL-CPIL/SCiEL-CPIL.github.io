import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
import time

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle with NumPy."""
    if not network:
        return n <= 1
        
    num_inputs = 1 << n
    inputs = np.arange(num_inputs, dtype=np.uint32)
    
    masks = (1 << np.arange(n, dtype=np.uint32))
    vals = ((inputs[:, None] & masks[None, :]) > 0).astype(np.uint8)
    
    for i, j in network:
        if i < 0 or i >= n or j < 0 or j >= n or i >= j:
            return False
        vi = vals[:, i]
        vj = vals[:, j]
        
        vi_min = vi & vj
        vj_max = vi | vj
        
        vals[:, i] = vi_min
        vals[:, j] = vj_max
        
    diffs = vals[:, 1:] - vals[:, :-1]
    return np.all(diffs >= 0)

def _bitonic_merge(n, direction):
    """
    Generate comparators for bitonic merge of n wires.
    direction: 0 for ascending, 1 for descending.
    """
    if n <= 1:
        return []
    
    comparators = []
    half = n // 2
    
    # Compare elements i and i+half
    for i in range(half):
        if direction == 0:
            # Ascending: smaller goes to i
            comparators.append((i, i + half))
        else:
            # Descending: larger goes to i, so we compare and put max in i
            # Since comparator always puts min in first arg, we swap:
            # compare (i+half, i) -> min goes to i+half, max goes to i
            comparators.append((i + half, i))
    
    # Recursively merge the two halves
    # First half: wires 0 to half-1
    # Second half: wires half to n-1
    comparators.extend([(i, j) for i, j in _bitonic_merge(half, direction)])
    comparators.extend([(i + half, j + half) for i, j in _bitonic_merge(half, direction)])
    
    return comparators

def _bitonic_sort(n, direction=0):
    """
    Generate a bitonic sorting network for n wires.
    direction: 0 for ascending, 1 for descending.
    """
    if n <= 1:
        return []
    
    comparators = []
    half = n // 2
    
    # Sort first half in direction
    comparators.extend(_bitonic_sort(half, direction))
    
    # Sort second half in opposite direction
    comparators.extend([(i + half, j + half) for i, j in _bitonic_sort(half, 1 - direction)])
    
    # Merge the two halves
    comparators.extend(_bitonic_merge(n, direction))
    
    return comparators

def _remove_redundant_comparators(network, n=16, max_iterations=50):
    """
    Try to remove redundant comparators from the network.
    A comparator is redundant if the network still sorts correctly without it.
    Uses a greedy approach: try removing each comparator one at a time.
    """
    start_time = time.time()
    deadline = start_time + 40  # Leave time for final verification
    
    current_net = list(network)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try removing each comparator
        for idx in range(len(current_net)):
            if time.time() >= deadline:
                break
            candidate = current_net[:idx] + current_net[idx+1:]
            if _verify_network_fast(candidate, n):
                current_net = candidate
                improved = True
                break  # Restart from beginning after removal
    
    return current_net

def construct():
    """Return a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin (evaluator kills at 90s)
    
    # Generate Batcher's Bitonic Sort network for 16 wires
    net = _bitonic_sort(N, 0)
    
    # Verify the network
    if not _verify_network_fast(net, N):
        # Fallback to odd-even transposition if bitonic fails
        net_fallback = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        return net_fallback
    
    # Try to reduce comparator count by removing redundant comparators
    if time.time() < deadline - 10:  # Only if we have time
        optimized_net = _remove_redundant_comparators(net, N)
        # Final verification of optimized network
        if _verify_network_fast(optimized_net, N):
            return optimized_net
    
    return net

# EVOLVE-BLOCK-END
