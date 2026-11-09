<<<<<<< SEARCH
def _apply_network_vec(net, bits_array):
    """Apply a list of comparators to a 2D array of bit vectors (num_patterns x N).
    bits_array: shape (num_patterns, N) with values in {0,1}
    Returns the sorted result array.
    """
    b = bits_array.copy()
    for i, j in net:
        # For each pattern, compare wire i and wire j
        # min goes to i, max goes to j
        # Since values are 0/1: min = 1 - (a OR b) ... actually simpler:
        # if b[i] > b[j], swap
        # Vectorized: mask where b[i] > b[j]
        mask = b[:, i] > b[:, j]
        # Swap where mask is True
        tmp = b[:, i].copy()
        b[:, i] = np.where(mask, b[:, j], b[:, i])
        b[:, j] = np.where(mask, tmp, b[:, j])
    return b


def _is_sorted_net_vec(net):
    """Check if the network sorts all 2^18 Boolean inputs using vectorized ops."""
    # Generate all 2^18 bit patterns
    # Use a lookup approach: for each wire position, the bit is set based on the mask
    all_masks = np.arange(1 << N, dtype=np.int32)
    # bits[pattern, wire] = (pattern >> wire) & 1
    # Shape: (2^N, N)
    # Efficient: use broadcasting
    wire_positions = np.arange(N, dtype=np.int32)
    bits = ((all_masks[:, None] >> wire_positions[None, :]) & 1).astype(np.int8)
    
    out = _apply_network_vec(net, bits)
    
    # Check non-decreasing: out[k] <= out[k+1] for all k
    # For 0/1 values, this means no pattern has a 1 followed by a 0
    violations = np.any(out[:, :-1] > out[:, 1:], axis=1)
    return not np.any(violations)


def _is_sorted_net(net):
    """Check if the network sorts all 2^N Boolean inputs."""
    return _is_sorted_net_vec(net)
=======
def _apply_network_vec(net, bits_array):
    """Apply a list of comparators to a 2D array of bit vectors (num_patterns x N).
    bits_array: shape (num_patterns, N) with values in {0,1}
    Returns the sorted result array.
    """
    b = bits_array.copy()
    for i, j in net:
        mask = b[:, i] > b[:, j]
        if np.any(mask):
            tmp = b[:, i].copy()
            b[:, i] = np.where(mask, b[:, j], b[:, i])
            b[:, j] = np.where(mask, tmp, b[:, j])
    return b


def _check_sorted(net, bits):
    """Check if the network sorts the given bit patterns."""
    out = _apply_network_vec(net, bits)
    # Check non-decreasing: out[k] <= out[k+1] for all k
    # For 0/1 values, this means no pattern has a 1 followed by a 0
    # violations: boolean array of shape (num_patterns,)
    # out[:, :-1] > out[:, 1:] is shape (num_patterns, N-1)
    # We want to know if ANY row has a violation.
    # np.any(out[:, :-1] > out[:, 1:], axis=1) gives boolean per row.
    # np.any(...) gives boolean for whole array.
    return not np.any(out[:, :-1] > out[:, 1:])
>>>>>>> REPLACE

