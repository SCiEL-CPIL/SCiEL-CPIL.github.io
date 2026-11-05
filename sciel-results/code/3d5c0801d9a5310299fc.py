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

def max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset of the given candidate set."""
    n = len(candidate_set)
    # Check from largest possible subset size down
    for k in range(n, 0, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Candidate 1: The "Pair" structure
    # Pairs of (x, x+1) spaced by 10
    cand1 = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
    
    # Candidate 2: Pairs of (x, x+2) spaced by 10
    cand2 = [0, 2, 10, 12, 20, 22, 30, 32, 40, 42, 50, 52]
    
    # Candidate 3: Seed
    cand3 = list(range(12))
    
    best_set = cand3
    best_size = max_sidon_size(cand3)
    
    for c in [cand1, cand2]:
        if len(set(c)) != 12: continue
        if any(x < 0 or x > 128 for x in c): continue
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = c
            
    # If we found size 3, we can stop or try to verify
    if best_size <= 3:
        return best_set

    # If not, do a quick random search to see if we can find better
    import random
    random.seed(42)
    while time.time() < deadline:
        s = random.sample(range(129), 12)
        sz = max_sidon_size(s)
        if sz < best_size:
            best_size = sz
            best_set = s
            if best_size <= 3:
                break
                
    return best_set
