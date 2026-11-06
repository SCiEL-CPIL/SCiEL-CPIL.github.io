"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    target_min = 128
    target_max = 256
    num_targets = target_max - target_min + 1  # 129
    
    # Initialize A with all integers [0, 128]
    # This guarantees coverage but has high peak multiplicity.
    A = list(range(0, 129))
    A_set = set(A)
    
    # Compute initial representation counts
    # counts[s] = number of pairs (a, b) in A with a <= b and a+b = s
    counts = np.zeros(target_max + 1, dtype=np.int32)
    k = len(A)
    for i in range(k):
        a = A[i]
        for j in range(i, k):
            b = A[j]
            s = a + b
            if s <= target_max:
                counts[s] += 1
            else:
                break # Since A is sorted, if a+b > max, further b will be larger
        
    def get_peak():
        return int(np.max(counts[target_min:target_max+1]))
    
    def is_valid():
        return np.all(counts[target_min:target_max+1] > 0)
    
    best_peak = get_peak()
    best_A = A[:]
    
    # Greedy removal loop
    # We try to remove elements one by one if it doesn't break coverage.
    # To avoid getting stuck in a specific order, we can randomize the order of attempts or use a priority queue.
    # For speed, we just iterate through the current A.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_peak = get_peak()
        
        # Iterate over a copy of the list to avoid modification during iteration issues, 
        # but we want to remove from the actual list.
        # We'll check each element in A.
        for idx in range(len(A) - 1, -1, -1): # Reverse order might help? Or random?
            if time.time() > deadline:
                break
                
            elem = A[idx]
            
            # Simulate removal of elem
            # We need to know which counts decrease.
            # Pairs involving elem:
            # 1. (elem, elem) -> sum 2*elem. Decrease by 1.
            # 2. (elem, y) for y in A, y > elem. Sum elem+y. Decrease by 1.
            
            # Check if any of these sums would become 0 in the target range.
            # If counts[s] == 1 and s is in [128, 256], removing it makes it 0 -> Invalid.
            
            valid_removal = True
            
            # Check (elem, elem)
            s_ee = 2 * elem
            if target_min <= s_ee <= target_max:
                if counts[s_ee] == 1:
                    valid_removal = False
            
            # Check (elem, y) for y > elem
            if valid_removal:
                # Find y > elem in A
                # Since A is sorted, we can binary search or just iterate from idx+1
                for j in range(idx + 1, len(A)):
                    y = A[j]
                    s = elem + y
                    if s > target_max:
                        break
                    if s >= target_min:
                        if counts[s] == 1:
                            valid_removal = False
                            break
            
            if valid_removal:
                # Perform removal
                # Update counts
                counts[2 * elem] -= 1
                
                for j in range(idx + 1, len(A)):
                    y = A[j]
                    s = elem + y
                    if s > target_max:
                        break
                    if s >= 0: # Always true
                        counts[s] -= 1
                
                # Remove from list and set
                A.pop(idx)
                A_set.discard(elem)
                
                new_peak = get_peak()
                # We only accept if it's valid (which we checked) and we track the best peak.
                # Since we are removing elements, the total number of pairs decreases, 
                # so the peak *should* generally decrease or stay same, but distribution matters.
                # Actually, removing an element *can* increase the relative peak if it was balancing things?
                # No, counts only decrease. So max(counts) can only decrease or stay same.
                # So any valid removal is an improvement or neutral.
                
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = A[:]
                
                improved = True
                # Restart the loop to re-evaluate with the new A
                break # Break the for loop, restart while loop
                
    return sorted(best_A)

# EVOLVE-BLOCK-END
