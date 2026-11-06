import numpy as np
import time
import random

N = 18

def _odd_even_transposition_network(n):
    """Generates the standard Odd-Even Transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _is_valid_network(network, inputs):
    """
    Checks if the network sorts all inputs in the provided 'inputs' array.
    'inputs' is a numpy array of shape (2^N, N).
    """
    state = inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    # Check if sorted: each wire k <= wire k+1 for all rows
    # state[:, :-1] <= state[:, 1:]
    return np.all(state[:, :-1] <= state[:, 1:])

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    # 1. Generate all Boolean inputs (2^18 rows, 18 cols)
    # Using np.indices or binary representation
    # 2^18 is 262144. 18 bits.
    # We can generate the matrix efficiently.
    # bits = np.arange(2**N)
    # inputs = np.zeros((2**N, N), dtype=np.uint8)
    # for k in range(N):
    #     inputs[:, k] = (bits >> k) & 1
    # This loop is fast enough for N=18.
    
    bits = np.arange(2**N, dtype=np.uint32)
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        inputs[:, k] = (bits >> k) & 1

    # 2. Start with the standard Odd-Even Transposition network
    current_network = _odd_even_transposition_network(N)
    
    # Verify initial network (should be valid)
    if not _is_valid_network(current_network, inputs):
        # Fallback: return standard network if something went wrong
        return current_network

    best_network = list(current_network)
    best_size = len(best_network)

    # 3. Greedy Deletion Search
    # We try to remove comparators one by one.
    # To escape local minima, we shuffle the order of attempts.
    
    # Pre-compute a list of indices to iterate over
    # We will iterate multiple times.
    
    max_attempts = 100000 # Upper bound on steps, but time is the real limit
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Shuffle the current network to try different removal orders
        # We work on a copy to see if removal is possible
        test_network = list(current_network)
        random.shuffle(test_network)
        
        removed_any = False
        
        for idx in range(len(test_network)):
            if time.time() > deadline:
                break
            
            # Try removing test_network[idx]
            # To avoid O(N^2) copy cost, we can just check validity of a modified list
            # But constructing the list is cheap. The validity check is the bottleneck.
            
            # Optimization: Instead of checking validity of the whole network every time,
            # we can just try to remove it.
            
            candidate = [c for i, c in enumerate(test_network) if i != idx]
            
            if _is_valid_network(candidate, inputs):
                # Valid! Keep the removal.
                current_network = candidate
                removed_any = True
                # Update test_network to reflect the removal? 
                # Actually, since we are iterating over indices of the shuffled list,
                # and we removed an element, the indices shift.
                # It's safer to restart the inner loop or handle indices carefully.
                # Given the cost of validity check, let's just break and restart outer loop 
                # with the new network if we found a removal.
                # This is simpler and avoids index bugs.
                break
        
        if removed_any:
            # If we removed something, update best if smaller
            if len(current_network) < best_size:
                best_size = len(current_network)
                best_network = list(current_network)
            # Restart outer loop with the new (smaller) network
            # This is a "greedy" step.
            continue
        else:
            # No removals found in this shuffle.
            # If we haven't improved in a while, we might be stuck.
            # For now, we just keep shuffling. 
            # A more advanced approach would be to try swapping comparators or adding new ones,
            # but deletion is the primary driver for reducing size from N^2.
            pass

    return best_network
