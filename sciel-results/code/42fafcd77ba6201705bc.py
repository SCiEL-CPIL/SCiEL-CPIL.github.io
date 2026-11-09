import random
import itertools
import time

def is_sidon(S):
    """Check if set S is a Sidon set (all unordered pair sums unique including diagonals)."""
    sums = set()
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            if i <= j:
                s = x + y
                if s in sums:
                    return False
                sums.add(s)
    return True

def has_sidon_subset_of_size_k(A, k):
    """Check if A contains a Sidon subset of size k."""
    # If k is small, we can iterate through combinations
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def _max_sidon_size(A):
    """Find the maximum size of a Sidon subset in A."""
    # Start from the largest possible size (len(A)) down to 1
    for k in range(len(A), 0, -1):
        if has_sidon_subset_of_size_k(A, k):
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave margin for evaluator overhead
    
    # Baseline: range(12) has max Sidon size 5
    best_set = list(range(12))
    best_max_size = 5
    
    # Verify baseline
    best_max_size = _max_sidon_size(best_set)
    
    # Random search for better sets
    # We want to minimize the max Sidon subset size
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        A = random.sample(range(129), 12)
        
        # Quick check: does it have a Sidon subset of size 5?
        # If yes, it's not better than baseline (which has 5)
        # We can skip it unless we find one with max size < 5
        # But wait, we want to minimize, so if it has 5, it's equal to baseline.
        # If it has < 5, it's better.
        
        # To be efficient, check for size 5 first.
        # If it has size 5, it's not an improvement over 5.
        # If it doesn't have size 5, check if it has size 4.
        # If it has size 4, max is 4 (improvement).
        # If it doesn't have size 4, max is <= 3 (even better).
        
        if not has_sidon_subset_of_size_k(A, 5):
            # Max size is <= 4
            # Check if it's exactly 4 or less
            if has_sidon_subset_of_size_k(A, 4):
                # Max size is 4
                if best_max_size > 4:
                    best_set = A
                    best_max_size = 4
                    # We can return early if we want, but let's keep searching for 3
                    # Actually, returning early is fine if we want to save time,
                    # but we might find a 3. Let's update best and continue.
            else:
                # Max size is <= 3
                # This is very good
                best_set = A
                best_max_size = 3
                # Return immediately as this is likely optimal or near optimal
                return A
        
    return best_set
