import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity_bitset(sums, n):
    """
    Check if the bitsets in 'sums' represent a valid cardinality-separated set.
    sums[k] is an integer where bit s is 1 if sum s is achievable with k elements.
    n is the current size of the set.
    """
    # sums[0] is always 1 (sum 0)
    # We need sums[i] & sums[j] == 0 for all i != j
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def construct():
    best_set = []
    
    # Strategy 1: Greedy Ascending
    A = []
    sums = [0] * 61
    sums[0] = 1
    n = 0
    for x in range(1, 257):
        if is_done(): break
        # Try adding x
        new_sums = sums[:]
        for k in range(n, -1, -1):
            new_sums[k+1] |= (new_sums[k] << x)
        if check_validity_bitset(new_sums, n + 1):
            A.append(x)
            sums = new_sums
            n += 1
            if n > len(best_set):
                best_set = A[:]
    
    # Strategy 2: Greedy Descending
    A = []
    sums = [0] * 61
    sums[0] = 1
    n = 0
    for x in range(256, 0, -1):
        if is_done(): break
        new_sums = sums[:]
        for k in range(n, -1, -1):
            new_sums[k+1] |= (new_sums[k] << x)
        if check_validity_bitset(new_sums, n + 1):
            A.append(x)
            sums = new_sums
            n += 1
            if n > len(best_set):
                best_set = A[:]
                
    # Strategy 3: Randomized Greedy with Restarts
    # We try to add elements in a random order, restarting if we get stuck or to find better sets
    max_restarts = 50
    for _ in range(max_restarts):
        if is_done(): break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        A = []
        sums = [0] * 61
        sums[0] = 1
        n = 0
        for x in candidates:
            if is_done(): break
            # Optimization: if we already have a set larger than best, we can be more selective
            # But for now, just greedy add
            new_sums = sums[:]
            for k in range(n, -1, -1):
                new_sums[k+1] |= (new_sums[k] << x)
            if check_validity_bitset(new_sums, n + 1):
                A.append(x)
                sums = new_sums
                n += 1
                if n > len(best_set):
                    best_set = A[:]
                    # If we hit a very large size, we might want to break early?
                    # No, keep trying to add more.
        
        # Local Search: Try to swap elements to allow more additions?
        # Simple local search: try to replace an element with a smaller one to free up space?
        # This is complex. Random restarts usually suffice for this type of constraint.
        
    # Final check and return
    # Ensure types are correct
    return [int(x) for x in best_set]
