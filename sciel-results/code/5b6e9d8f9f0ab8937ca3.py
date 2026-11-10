import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Leave 5s for overhead/return
    
    best_set = []
    
    def try_greedy(order):
        nonlocal best_set
        if time.time() - start_time > timeout:
            return
        
        S = []
        # sums[k] is a bitset where bit s is 1 if sum s is achievable with exactly k elements from S
        # Max elements 48, Max sum 48*512 = 24576
        max_k = 48
        sums = [0] * (max_k + 1)
        sums[0] = 1  # Sum 0 with 0 elements
        
        for x in order:
            # Check if x is the mean of any subset of current S
            # Condition: exists k >= 2 such that sum(subset of size k) == k * x
            valid = True
            for k in range(2, len(S) + 1):
                target_sum = k * x
                # Check if target_sum is achievable with k elements
                if (sums[k] >> target_sum) & 1:
                    valid = False
                    break
            
            if valid:
                # Add x to S
                S.append(x)
                # Update sums for future checks
                # We need to update sums[k] to include subsets containing x
                # Iterate backwards to avoid using x multiple times
                new_sums = sums[:]
                for k in range(len(S), 0, -1):
                    new_sums[k] |= (sums[k-1] << x)
                sums = new_sums
                
                if len(S) > len(best_set):
                    best_set = S[:]
                    if len(best_set) == 48:
                        return

    # 1. Deterministic Ascending
    try_greedy(range(1, 513))
    
    # 2. Deterministic Descending
    if time.time() - start_time < timeout:
        try_greedy(range(512, 0, -1))
        
    # 3. Randomized Restarts
    if time.time() - start_time < timeout:
        # Use a local random instance to avoid global state issues if called multiple times?
        # But construct() is called once per evaluation usually.
        rng = random.Random(42) 
        while time.time() - start_time < timeout:
            order = list(range(1, 513))
            rng.shuffle(order)
            try_greedy(order)
            
    return best_set
