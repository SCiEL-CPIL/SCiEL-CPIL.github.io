import time
import numpy as np

N = 17
DEADLINE = time.time() + 85.0

def construct():
    # Generate the seed: Odd-Even Transposition Sort
    seed_net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            seed_net.append((i, i + 1))
            
    current_net = list(seed_net)
    
    def is_valid(net):
        # Generate all 131072 inputs as a (131072, 17) array of 0s and 1s
        # This is done efficiently using bit manipulation
        arr = ((np.arange(1 << N)[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
        
        # Apply network
        for i, j in net:
            # Compare-exchange: min to i, max to j
            # Using vectorized min/max
            min_val = np.minimum(arr[:, i], arr[:, j])
            max_val = np.maximum(arr[:, i], arr[:, j])
            arr[:, i] = min_val
            arr[:, j] = max_val
            
        # Check if sorted: no 1 before a 0 in any row
        # Equivalent to checking if rows are non-decreasing
        diffs = np.diff(arr, axis=1)
        if np.any(diffs < 0):
            return False
        return True

    # Verify seed is valid
    if not is_valid(current_net):
        return seed_net
        
    # Local Search: Try to remove each comparator
    i = len(current_net) - 1
    while i >= 0 and time.time() < DEADLINE:
        # Try removing current_net[i]
        test_net = current_net[:i] + current_net[i+1:]
        if is_valid(test_net):
            current_net = test_net
            # Do not decrement i, as the next element to test is now at index i
        else:
            i -= 1
            
    return current_net
