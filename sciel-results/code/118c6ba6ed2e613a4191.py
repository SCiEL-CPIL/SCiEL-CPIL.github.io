import time
import numpy as np
import itertools

N = 16

# EVOLVE-BLOCK-START
import time

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle with NumPy."""
    # Generate all 2^n inputs as a bit matrix (65536, 16)
    # Use uint8 for speed
    num_inputs = 1 << n
    # Create a range of 0 to 2^n - 1
    inputs = np.arange(num_inputs, dtype=np.uint32)
    
    # Expand to binary representation
    # We need to extract bits.
    # Mask for each bit
    masks = (1 << np.arange(n, dtype=np.uint32))
    
    # inputs[:, None] & masks[None, :]
    # Result shape: (65536, 16)
    vals = ((inputs[:, None] & masks[None, :]) > 0).astype(np.uint8)
    
    for i, j in network:
        # Extract columns i and j
        vi = vals[:, i]
        vj = vals[:, j]
        
        # min = vi & vj
        # max = vi | vj
        vi_min = vi & vj
        vj_max = vi | vj
        
        vals[:, i] = vi_min
        vals[:, j] = vj_max
        
    # Check if sorted: vals[:, k] <= vals[:, k+1] for all k
    # Since it's 0/1, sorted means no 1 followed by 0.
    # i.e., vals[:, :-1] <= vals[:, 1:]
    diffs = vals[:, 1:] - vals[:, :-1]
    # If any diff is -1 (1 -> 0), it's not sorted.
    # Or simply check if all rows are non-decreasing.
    return np.all(diffs >= 0)

def _generate_bitonic(n):
    """Generate Batcher's Bitonic Sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort left half ascending
    left = _generate_bitonic(half)
    
    # Sort right half descending
    # We generate ascending sort for right half, then reverse wire indices
    right_asc = _generate_bitonic(half)
    # Shift to right half indices
    right_asc_shifted = [(i + half, j + half) for i, j in right_asc]
    # Convert to descending by reversing wire indices within the right half?
    # If we want descending sort on wires [half, n-1],
    # we can take the ascending sort network for [0, half-1] and map wire k to n-1-k?
    # Let's use the property: Net_desc = [(n-1-j, n-1-i) for i,j in Net_asc]
    # But we are sorting a sub-array.
    # Let's just generate the ascending sort for the right half, and then apply the "reverse wire" trick locally.
    # Local reverse: wire k in [0, half-1] maps to (half-1-k) in [half, n-1]?
    # No, that reverses the order of wires.
    
    # Actually, there is a simpler way to get descending sort:
    # Just use the same network, but the "merge" step of bitonic sort handles the direction?
    # No, Bitonic Merge requires one asc, one desc.
    
    # Let's try the standard construction:
    # 1. Sort left ascending.
    # 2. Sort right descending.
    # 3. Merge.
    
    # To get right descending:
    # Generate ascending sort for half.
    # Apply it to the right half.
    # Then, reverse the output of the right half?
    # We can't reverse output.
    
    # Alternative:
    # The network for descending sort of size n is the same as ascending sort of size n,
    # with all comparators (i, j) replaced by (n-1-j, n-1-i).
    # Let's assume this works for n > 2.
    right_desc = [(half + (half - 1 - j), half + (half - 1 - i)) for i, j in right_asc]
    # Note: i, j are in [0, half-1].
    # New indices: half + (half - 1 - j) and half + (half - 1 - i).
    # This reverses the order of wires in the right half.
    
    # Merge step
    merge = _bitonic_merge(n)
    
    return left + right_desc + merge

def _bitonic_merge(n):
    """Generate Bitonic Merge network for n wires (input is bitonic)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # First layer: compare (i, i+half)
    first_layer = [(i, i + half) for i in range(half)]
    
    # Then recursively merge the even and odd subsequences
    # Even indices: 0, 2, 4...
    # Odd indices: 1, 3, 5...
    
    # The subsequence of even indices forms a bitonic sequence of size half.
    # The subsequence of odd indices forms a bitonic sequence of size half.
    
    even_merge = _bitonic_merge(half)
    even_mapped = [(2 * i, 2 * j) for i, j in even_merge]
    
    odd_merge = _bitonic_merge(half)
    odd_mapped = [(2 * i + 1, 2 * j + 1) for i, j in odd_merge]
    
    return first_layer + even_mapped + odd_mapped

def _optimize_network(network, n=16, time_limit=45):
    """Optimize network by removing redundant comparators."""
    start_time = time.time()
    current_net = list(network)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        improved = False
        for idx in range(len(current_net)):
            # Try removing current_net[idx]
            test_net = current_net[:idx] + current_net[idx+1:]
            
            if _verify_network_fast(test_net, n):
                current_net = test_net
                improved = True
                break # Restart loop with new network
        
        if not improved:
            # Try a few random swaps to escape local minima?
            # Or just break if no improvement.
            # Given the time, simple greedy is often sufficient for Bitonic.
            break
            
    return current_net

def construct():
    """Return a sorting network for 16 wires."""
    # Start with Bitonic Sort (80 comparators)
    net = _generate_bitonic(16)
    
    # Verify initial network
    if not _verify_network_fast(net, 16):
        # Fallback to known valid network (Odd-Even Transposition)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # Optimize
    optimized_net = _optimize_network(net, 16, time_limit=50)
    
    # Final check
    if _verify_network_fast(optimized_net, 16):
        return optimized_net
    else:
        return net
# EVOLVE-BLOCK-END
