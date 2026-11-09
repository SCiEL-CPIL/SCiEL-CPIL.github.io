import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a subset has unique pair sums (a+b for a<=b)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        a = subset[i]
        for j in range(i, n):
            s = a + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest size down
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if is_sidon(comb):
                return k
    return 0

def has_sidon_4(S):
    """Check if S contains any Sidon subset of size 4."""
    for comb in combinations(S, 4):
        if is_sidon(comb):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)  # Should be 4
    
    # Try some structured candidates first
    structured_candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    
    for cand in structured_candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            # Quick check for score 3
            if not has_sidon_4(cand):
                s = max_sidon_size(cand)
                if s < best_score:
                    best_score = s
                    best_set = cand[:]
                    if best_score <= 3:
                        # We found a good one, but let's see if we can get 2
                        # Don't break immediately, maybe there's a better one?
                        # Actually, 3 is a huge jump from 4. 2 is unlikely.
                        # We can break if we want to save time, but let's continue random search briefly.
                        pass
    
    # Random search
    random.seed(42)
    
    while time.time() < deadline:
        # Generate random 12-element set
        S = sorted(random.sample(range(129), 12))
        
        # Check if it has any Sidon 4
        if not has_sidon_4(S):
            s = max_sidon_size(S)
            if s < best_score:
                best_score = s
                best_set = S[:]
                if best_score <= 2:
                    break # Found optimal or near optimal
    
    return best_set
