<<<<<<< SEARCH
def _apply_network(net, vals):
    """Apply network to values, return sorted check."""
    v = list(vals)
    for i, j in net:
        if v[i] > v[j]:
            v[i], v[j] = v[j], v[i]
    return v

def _is_sorted(vals):
    for k in range(len(vals) - 1):
        if vals[k] > vals[k+1]:
            return False
    return True

def _fast_check(net, samples):
    """Check network on a subset of random boolean inputs."""
    for s in samples:
        v = _apply_network(net, s)
        if not _is_sorted(v):
            return False, s
    return True, None

def _full_check(net):
    """Full verification over all 2^18 boolean inputs."""
    for mask in range(1 << N):
        vals = [(mask >> i) & 1 for i in range(N)]
        v = _apply_network(net, vals)
        if not _is_sorted(v):
            return False
    return True
=======
def _fast_full_check(net):
    """
    Vectorized full verification over all 2^18 boolean inputs.
    Returns True if the network sorts all inputs.
    """
    if not net:
        return N <= 1
    
    # Generate all 2^18 boolean inputs as a matrix of shape (2^18, N)
    # Using bit manipulation for speed
    num_inputs = 1 << N
    # Create an array of indices 0 to 2^N - 1
    indices = np.arange(num_inputs, dtype=np.int64)
    # Extract bits for each wire
    # input[i, j] is bit j of indices[i]
    # We can do this by shifting and masking, but vectorized over indices
    # Shape: (N, num_inputs)
    bits = np.zeros((N, num_inputs), dtype=np.int8)
    for j in range(N):
        bits[j] = (indices >> j) & 1
    
    # Transpose to (num_inputs, N) for easier application
    # Actually, applying the network is easier if we keep it as (num_inputs, N)
    # Let's transpose bits
    inputs = bits.T  # (num_inputs, N)
    
    # Apply network
    for i, j in net:
        # Compare wire i and wire j for all inputs
        # If inputs[:, i] > inputs[:, j], swap
        # Since inputs are 0/1, this is equivalent to:
        # min = inputs[:, i] & inputs[:, j]
        # max = inputs[:, i] | inputs[:, j]
        # But we need to do this conditionally? No, min/max works for 0/1.
        # However, standard compare-exchange swaps if a > b.
        # For 0/1:
        # if a=1, b=0 -> swap -> a=0, b=1
        # min(1,0)=0, max(1,0)=1.
        # if a=0, b=1 -> no swap -> a=0, b=1
        # min(0,1)=0, max(0,1)=1.
        # if a=1, b=1 -> no swap -> a=1, b=1
        # min(1,1)=1, max(1,1)=1.
        # if a=0, b=0 -> no swap -> a=0, b=0
        # min(0,0)=0, max(0,0)=0.
        # So we can just compute min and max directly.
        
        a = inputs[:, i]
        b = inputs[:, j]
        
        # Calculate min and max
        # min = a & b
        # max = a | b
        new_min = a & b
        new_max = a | b
        
        inputs[:, i] = new_min
        inputs[:, j] = new_max

    # Check if sorted
    # For each input row, check if inputs[k] <= inputs[k+1] for all k
    # diff = inputs[:, 1:] - inputs[:, :-1]
    # If any diff < 0, it's not sorted.
    diffs = inputs[:, 1:] - inputs[:, :-1]
    # If any element in diffs is -1, it's not sorted.
    return not np.any(diffs < 0)
>>>>>>> REPLACE

