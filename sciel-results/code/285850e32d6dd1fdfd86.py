import random
import time

def is_valid_subset_sums(S):
    sums = set()
    n = len(S)
    # Generate all non-empty subset sums
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    
    # Check divisibility
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % sorted_sums[i] == 0:
                return False
    return True

def construct():
    # Try to find a set of size 4
    # If found, return it. Otherwise return size 3.
    # We have 60 seconds.
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Known valid set of size 3
    best_set = [31, 32, 34]
    
    # Search for size 4
    # We can use random restarts
    # To make it faster, we can restrict the range or use heuristics
    # Let's try random sets of size 4
    # We can also try to build it incrementally
    
    # A better search: 
    # Pick 4 numbers.
    # To avoid x | 2x, we want the sums to be "dense" but not hitting multiples.
    # Or "sparse" but avoiding multiples.
    
    # Let's try a local search or random search.
    # Given the small size, random search might work.
    
    for _ in range(100000): # Limit iterations to avoid timeout
        if time.time() > deadline:
            break
        
        # Generate random set of size 4
        S = random.sample(range(1, 257), 4)
        if is_valid_subset_sums(S):
            return S
            
    # If no size 4 found, try to see if we can improve size 3? 
    # No, size 3 is the baseline.
    # Maybe we can find a size 4 with a more directed search?
    # Let's try to search for size 4 with a heuristic:
    # Pick numbers close to each other to keep sums close?
    # Or pick numbers such that they are "large" to push sums out of divisibility range?
    
    # Let's try one more time with a slightly different strategy:
    # Pick 4 numbers from a specific range, e.g., [100, 200]
    for _ in range(50000):
        if time.time() > deadline:
            break
        S = random.sample(range(100, 201), 4)
        if is_valid_subset_sums(S):
            return S

    return best_set
