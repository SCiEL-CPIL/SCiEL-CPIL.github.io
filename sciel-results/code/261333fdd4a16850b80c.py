import numpy as np
import time

def construct():
    N = 16
    deadline = time.time() + 85.0  # Leave 5s margin for 90s limit
    
    # 1. Generate Odd-Even Transposition Network
    # This network has N*(N-1)/2 comparators. For N=16, that's 120.
    # It is correct.
    initial_net = []
    for k in range(N):
        for i in range(0, N - 1 - (k % 2), 2):
            initial_net.append((i, i + 1))
            
    # 2. Verification Function using Bit-Parallelism
    # We represent the state of all 2^N inputs as a matrix of shape (2^N, N)
    # But to be efficient, we can represent the values as integers if we just track min/max?
    # No, we need to track the values on each wire.
    # Since inputs are boolean, values are 0 or 1.
    # We can use a uint8 array of shape (65536, 16).
    # 65536 * 16 bytes = 1MB. This is very small and fast.
    
    # Pre-compute all boolean inputs
    # Input i has 1s in positions where the bit is set.
    # Actually, we can just generate the matrix.
    inputs = np.array([list(map(int, bin(i).fill(N, '0'))[::-1]) for i in range(1 << N)], dtype=np.uint8)
    # Note: bin(i) gives MSB first. We want LSB first or consistent order.
    # Let's stick to a standard order.
    # Let's regenerate properly to avoid confusion.
    # Wire 0 is LSB, Wire N-1 is MSB.
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(1 << N):
        for j in range(N):
            inputs[i, j] = (i >> j) & 1

    def verify(net):
        # Apply net to inputs
        # To avoid copying the large array 120 times, we can process in-place?
        # No, we need to check the original state for each removal attempt.
        # So we must copy or re-apply.
        # Copying 1MB is fast. 120 * 1MB copy is ~120MB memory traffic. Very fast.
        
        current = inputs.copy()
        for i, j in net:
            # compare-exchange: min to i, max to j
            # min_val = np.minimum(current[:, i], current[:, j])
            # max_val = np.maximum(current[:, i], current[:, j])
            # current[:, i] = min_val
            # current[:, j] = max_val
            
            # Optimized:
            # If current[:, i] > current[:, j], swap.
            # swap_mask = current[:, i] > current[:, j]
            # temp = current[:, i]
            # current[:, i] = np.where(swap_mask, current[:, j], temp)
            # current[:, j] = np.where(swap_mask, temp, current[:, j])
            
            # Even faster for booleans:
            # min is AND, max is OR.
            # current[:, i] = np.bitwise_and(current[:, i], current[:, j])
            # current[:, j] = np.bitwise_or(current[:, i], current[:, j])
            # Wait, if we update i first, j uses the new i?
            # No, we need original values for both.
            
            min_val = np.bitwise_and(current[:, i], current[:, j])
            max_val = np.bitwise_or(current[:, i], current[:, j])
            current[:, i] = min_val
            current[:, j] = max_val

        # Check if sorted: current[:, 0] <= current[:, 1] <= ... <= current[:, N-1]
        # Since they are 0/1, sorted means all 0s come before all 1s.
        # Equivalent to: for all i, current[:, i] >= current[:, i+1] is FALSE?
        # No, sorted ascending means current[:, i] <= current[:, i+1].
        # In binary, this means if current[:, i] is 1, current[:, i+1] MUST be 1.
        # Violation: current[:, i] == 1 AND current[:, i+1] == 0.
        
        for i in range(N - 1):
            violation = np.bitwise_and(current[:, i], np.bitwise_not(current[:, i + 1]))
            if np.any(violation):
                return False
        return True

    # 3. Pruning Loop
    # We try to remove comparators.
    # We iterate through the list. If we remove one, the indices shift?
    # It's better to work with a list and remove by index, or mark as removed.
    # Let's use a list of active indices or just a list of tuples.
    
    current_net = list(initial_net)
    
    # We will try to remove each comparator one by one.
    # To be efficient, we can try removing them in a specific order.
    # A common heuristic is to try removing comparators that are "redundant".
    # Greedy: iterate through the list. Try to remove comparator at index k.
    # If valid, remove it and restart the scan from the beginning? Or continue?
    # Restarting from the beginning ensures we don't miss new redundancies created by previous removals.
    # However, restarting is expensive.
    # Let's try a single pass first, then repeat passes until no change or time limit.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(current_net) and time.time() < deadline:
            # Try removing current_net[i]
            candidate_net = current_net[:i] + current_net[i+1:]
            
            if verify(candidate_net):
                current_net = candidate_net
                improved = True
                # Do NOT increment i, because the next element shifted into position i
                # might also be removable.
            else:
                i += 1
                
    return current_net
