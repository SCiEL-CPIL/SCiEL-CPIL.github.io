import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
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

def has_sidon_subset_of_size_4(candidate):
    """Check if there exists a Sidon subset of size 4."""
    for subset in combinations(candidate, 4):
        if is_sidon(subset):
            return True
    return False

def max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset of the given candidate set."""
    n = len(candidate_set)
    # Check from largest possible subset size down
    # Optimization: If we only care about distinguishing 3 vs 4+, 
    # we can just check 4. But for the final score, we need the exact value.
    # However, if we find a set with no 4-Sidon, the value is <= 3.
    # We can assume it's 3 (since 2 is unlikely).
    # Let's just do the full check but start from 4 downwards if we suspect small values?
    # No, just do the standard check but it's fast enough for 12 elements.
    for k in range(n, 0, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    import random
    random.seed(42)
    
    best_set = None
    best_score = 100 # We want to minimize max_sidon_size
    
    # Try to find a set with max Sidon size <= 3
    # This means no 4-element Sidon subset exists.
    found_three = False
    
    # Initial candidates
    candidates = [
        list(range(12)),
        [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51],
        [0, 2, 10, 12, 20, 22, 30, 32, 40, 42, 50, 52]
    ]
    
    for c in candidates:
        if not has_sidon_subset_of_size_4(c):
            best_set = c
            found_three = True
            break
        else:
            # Evaluate full size to keep track
            sz = max_sidon_size(c)
            if best_set is None or sz < best_score:
                best_score = sz
                best_set = c
                
    if found_three:
        return best_set
        
    # Random search for a set with no 4-Sidon subset
    while time.time() < deadline:
        s = random.sample(range(129), 12)
        if not has_sidon_subset_of_size_4(s):
            best_set = s
            found_three = True
            break
        else:
            # Optional: track best (lowest max size) if we don't find 3
            # But checking full max size is slower.
            # We can just keep the first one or random ones.
            # If we don't find 3, we return the best known.
            pass
            
    if found_three:
        return best_set
        
    # If we didn't find a set with max Sidon <= 3, 
    # we should try to find one with max Sidon = 4 (which we already have in candidates)
    # Or maybe try to minimize the number of 4-Sidon subsets?
    # But the score depends on the *maximum* size.
    # If all sets have max size >= 4, and we found one with 4, that's the best we can do if 3 is impossible.
    # Let's assume the candidate with max size 4 is the best.
    
    # Just in case, do a few more random checks with full max_sidon_size to see if we can find 4?
    # We already have candidates with 4.
    
    return best_set
