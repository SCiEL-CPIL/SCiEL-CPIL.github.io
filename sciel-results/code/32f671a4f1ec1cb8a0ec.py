<<<<<<< SEARCH
def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    total = 1 << N
    # Generate all 2^N boolean inputs
    # Use a fast method: for each bit position, create the pattern
    inputs = np.empty((total, N), dtype=np.uint8)
    for i in range(N):
        # Bit i of x for x in 0..total-1
        inputs[:, i] = (np.arange(total, dtype=np.uint32) >> i) & 1
    
    for (i, j) in network:
        a = inputs[:, i].copy()
        b = inputs[:, j].copy()
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    
    # Check sorted: no 1 before 0 in any row
    # Equivalent: for each row, the sequence is non-decreasing
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True
=======
def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    total = 1 << N
    # Generate all 2^N boolean inputs
    inputs = np.empty((total, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(total, dtype=np.uint32) >> i) & 1
    
    for (i, j) in network:
        # Use in-place min/max with temporary storage to avoid copies
        tmp = inputs[:, i]
        inputs[:, i] = np.minimum(tmp, inputs[:, j])
        inputs[:, j] = np.maximum(tmp, inputs[:, j])
    
    # Check sorted: no 1 before 0 in any row
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True
>>>>>>> REPLACE

