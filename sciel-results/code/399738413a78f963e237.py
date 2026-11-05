import itertools
import random
import time

def is_sidon(s):
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                return False
            sums.add(total)
    return True

def has_sidon_subset(s, k):
    n = len(s)
    for subset in itertools.combinations(s, k):
        if is_sidon(subset):
            return True
    return False

def max_sidon_size(s):
    n = len(s)
    for k in range(n, 2, -1):
        if has_sidon_subset(s, k):
            return k
    return 2

def construct():
    # We want to minimize the maximum Sidon subset size.
    # Theoretical minimum for n=12 is likely 3 or 4.
    # Current best (range(12)) is likely 4.
    # We search for a set with max Sidon size 3.
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    # If we already have 3, we are done.
    if best_size == 3:
        return best_set
        
    # Random search for a set with max Sidon size < best_size
    # We use a time limit to ensure we finish within 60s.
    # But 1000 iterations is very fast.
    
    start_time = time.time()
    limit = 50 # seconds
    
    for _ in range(5000):
        if time.time() - start_time > limit:
            break
            
        # Generate a random set of 12 distinct integers in [0, 128]
        # Using small ranges might help create collisions, but random is fine.
        s = random.sample(range(129), 12)
        
        # Quick check: if it has a Sidon subset of size best_size, skip?
        # No, we need the MAX size.
        # We can check if it has a Sidon subset of size best_size.
        # If it does, its max size is >= best_size. We want < best_size.
        # So if has_sidon_subset(s, best_size) is True, it's not better.
        # If False, its max size is < best_size.
        
        if not has_sidon_subset(s, best_size):
            # Then max size is at most best_size - 1
            # Let's find the exact max size to be sure, or just update best_size
            # Since we want to minimize, any size < best_size is an improvement.
            # We can just set best_size to best_size - 1? No, it could be lower.
            # But we only care about minimizing.
            # Let's compute the exact max size for the new best.
            new_size = max_sidon_size(s)
            if new_size < best_size:
                best_size = new_size
                best_set = s
                if best_size == 3:
                    break # Can't get lower than 3 for n=12? 
                    # Actually, can we get 2?
                    # If no 3 Sidon subset, max is 2.
                    # Is it possible to have 12 integers with no Sidon subset of size 3?
                    # Sidon subset of size 3: a,b,c such that a+a, a+b, a+c, b+b, b+c, c+c are distinct.
                    # a+b = a+c => b=c (distinct, so no)
                    # a+a = a+b => a=b (distinct, so no)
                    # a+b = b+c => a=c (distinct, so no)
                    # a+a = b+b => a=b (distinct, so no)
                    # a+b = b+c is the only cross term collision?
                    # Wait, a+b = c+c is possible.
                    # So a,b,c is NOT Sidon if a+b = c+c or a+c = b+b or b+c = a+a.
                    # So if we avoid these, it IS Sidon.
                    # Can we have 12 integers such that for ANY a,b,c, one of these holds?
                    # This is a very strong condition.
                    # Likely max Sidon size 3 is the minimum.
                    pass

    return best_set
