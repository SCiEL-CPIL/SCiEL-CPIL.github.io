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

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave margin for evaluator overhead
    
    # Try to find a set with no Sidon subset of size 5
    # If found, the max Sidon size is <= 4.
    # Baseline is likely 5.
    
    # Heuristic: Random search
    # We can also try to construct it.
    # A set of multiples of a large number might help?
    # Or a set that is "dense" in a small range?
    # range(12) has max Sidon 5.
    # Let's try random sets.
    
    best_set = list(range(12))
    best_max_size = 5 # Assume baseline is 5
    
    # First, verify baseline
    if not has_sidon_subset_of_size_k(best_set, 5):
        best_max_size = 4
        # Check if 4 is possible
        if not has_sidon_subset_of_size_k(best_set, 4):
            best_max_size = 3
            
    # Random search for better sets
    # We want to minimize the largest Sidon subset.
    # Current best is 5 (from range(12)).
    # If we find a set with max Sidon size 4, we return it immediately (score 1.25).
    # If we find a set with max Sidon size 3, we return it immediately (score 1.67).
    
    # To make the search more effective, we can try to construct sets that are "dense" 
    # in a way that creates many sum collisions, thus preventing large Sidon subsets.
    # One strategy is to pick numbers from a small range, but range(12) already gives 5.
    # Maybe a set with many duplicates in differences?
    
    # Let's try a mixed approach:
    # 1. Random search
    # 2. Structured search: e.g., sets with many elements in [0, 10]
    
    # We will iterate many times.
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        A = random.sample(range(129), 12)
        
        # Check for Sidon subset of size 5
        if has_sidon_subset_of_size_k(A, 5):
            continue
            
        # If no Sidon subset of size 5, max size is <= 4
        # Check if it has Sidon subset of size 4
        if has_sidon_subset_of_size_k(A, 4):
            # Max size is 4
            return A
        else:
            # Max size is <= 3
            return A

    # If we didn't find a set with max size < 5, return the best known set
    # which is the baseline with max size 5.
    return best_set
