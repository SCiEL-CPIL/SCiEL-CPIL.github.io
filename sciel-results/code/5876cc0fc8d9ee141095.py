import time
import random
from itertools import combinations

def is_sidon_4(subset):
    """
    Check if a subset of size 4 is Sidon.
    """
    sums = set()
    # subset has 4 elements
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def num_sidon_4(S):
    """
    Count the number of Sidon subsets of size 4 in S.
    """
    count = 0
    for subset in combinations(S, 4):
        if is_sidon_4(subset):
            count += 1
    return count

def max_sidon_size(S):
    """
    Accurate max Sidon size. Used for final verification.
    """
    from itertools import combinations
    def is_sidon(subset):
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
        
    for k in range(12, 0, -1):
        for subset in combinations(S, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initialize
    best_set = list(range(12))
    best_cost = num_sidon_4(best_set)
    
    current_set = list(best_set)
    current_cost = best_cost
    
    # Local search
    while time.time() < deadline:
        i = random.randint(0, 11)
        old_val = current_set[i]
        new_val = random.randint(0, 128)
        
        if new_val == old_val or new_val in current_set:
            continue
            
        current_set[i] = new_val
        new_cost = num_sidon_4(current_set)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_set = list(current_set)
                if best_cost == 0:
                    # We found a set with no Sidon-4.
                    # Verify max Sidon size
                    true_max = max_sidon_size(best_set)
                    # If true_max is 3, this is optimal (likely).
                    # We can break or continue to see if we can get 2?
                    # Getting 2 is very hard.
                    # Let's break to save time and return this.
                    # But wait, we should ensure we return the BEST set.
                    # If we found 0 Sidon-4, the max Sidon size is 3.
                    # Can we get max Sidon size 2?
                    # That would require NO Sidon-3.
                    # Let's check if we can find a set with no Sidon-3.
                    # Cost function for that would be num_sidon_3.
                    # But 3 is a very good score.
                    # Let's assume 3 is the target.
                    break
        elif new_cost == current_cost:
            # Accept to explore
            pass
        else:
            current_set[i] = old_val
            
    return best_set
