import random
import itertools
import time

def is_sidon(subset):
    """Check if a tuple of integers is a Sidon set (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # We check from largest to smallest
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Initialize with a set known to have many APs
    # e.g. 0..11 or random
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    if best_score == 2:
        return best_set

    # Local search
    current = best_set[:]
    current_score = best_score
    
    for step in range(1500):
        if time.time() > deadline:
            break
        
        # Pick a random index
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Try to replace with a random value
        new_val = random.randint(0, 128)
        if new_val in current:
            continue
            
        current[idx] = new_val
        new_score = max_sidon_size(current)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                if best_score == 2:
                    break
        else:
            # Revert
            current[idx] = old_val
            
    return best_set
