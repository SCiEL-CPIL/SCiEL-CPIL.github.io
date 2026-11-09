<<<<<<< SEARCH
def _verify_network_fast(comps):
    """
    Verify if the network sorts all 2^N binary inputs using the zero-one principle.
    Since N=16, 2^16 = 65536 inputs. This is manageable.
    Returns True if valid, False otherwise.
    """
    # Pre-compute the permutation logic or just run it?
    # Running 65536 * len(comps) operations.
    # If len(comps) ~ 60, 65536 * 60 = 3.9M operations.
    # In Python, this might take 1-2 seconds.
    # In NumPy, we can vectorize if we treat inputs as a matrix.
    
    # Let's use NumPy for speed.
    # Inputs: 65536 x 16 matrix of 0s and 1s.
    # We can generate all 2^16 patterns efficiently.
    
    # Generate all 2^N patterns
    # np.arange(2**N).reshape(-1, 1) // np.power(2, N-1 - np.arange(N)) % 2
    patterns = np.arange(2**N).reshape(-1, 1) // np.power(2, N - 1 - np.arange(N)) % 2
    
    # Apply comparators
    # We need to update the matrix in place for each comparator.
    # Comparator (i, j): 
    #   min = np.minimum(patterns[:, i], patterns[:, j])
    #   max = np.maximum(patterns[:, i], patterns[:, j])
    #   patterns[:, i] = min
    #   patterns[:, j] = max
    
    for i, j in comps:
        min_val = np.minimum(patterns[:, i], patterns[:, j])
        max_val = np.maximum(patterns[:, i], patterns[:, j])
        patterns[:, i] = min_val
        patterns[:, j] = max_val
    
    # Check if all rows are sorted
    # A row is sorted if patterns[:, 0] <= patterns[:, 1] <= ... <= patterns[:, N-1]
    # Since binary, this means all 0s come before 1s.
    # Check: for each row, find the first 1. All elements before it must be 0, all after must be 1.
    # Or simply check if np.diff(patterns, axis=1) has any -1.
    # If sorted, diff is either 0 or 1. If unsorted, there is a 1 followed by 0, so diff is -1.
    
    diffs = np.diff(patterns, axis=1)
    # If any diff is -1, it's not sorted.
    if np.any(diffs < 0):
        return False
    return True
=======
def _verify_network_fast(comps):
    """
    Verify if the network sorts all 2^N binary inputs using the zero-one principle.
    Since N=16, 2^16 = 65536 inputs. This is manageable.
    Returns True if valid, False otherwise.
    """
    # Generate all 2^N patterns
    patterns = np.arange(2**N).reshape(-1, 1) // np.power(2, N - 1 - np.arange(N)) % 2
    
    # Apply comparators
    for i, j in comps:
        min_val = np.minimum(patterns[:, i], patterns[:, j])
        max_val = np.maximum(patterns[:, i], patterns[:, j])
        patterns[:, i] = min_val
        patterns[:, j] = max_val
    
    # Check if all rows are sorted
    diffs = np.diff(patterns, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _verify_network_with_state(comps, patterns):
    """
    Verify network given pre-computed patterns. Returns True if valid.
    This avoids regenerating patterns each time.
    """
    patterns = patterns.copy()
    for i, j in comps:
        min_val = np.minimum(patterns[:, i], patterns[:, j])
        max_val = np.maximum(patterns[:, i], patterns[:, j])
        patterns[:, i] = min_val
        patterns[:, j] = max_val
    
    diffs = np.diff(patterns, axis=1)
    if np.any(diffs < 0):
        return False
    return True
>>>>>>> REPLACE

