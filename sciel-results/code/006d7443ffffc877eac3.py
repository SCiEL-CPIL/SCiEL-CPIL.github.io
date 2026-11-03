import itertools
import math
import time

def score_set(U):
    U = sorted(U)
    if 0 not in U:
        return -1
    max_u = max(U)
    # Calculate U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0:
        return -1
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    best_score = -1
    best_set = [0, 1, 3]
    
    # Search space parameters
    max_n = 6
    max_val = 30
    
    # We must include 0. So we pick n-1 elements from 1..max_val
    for n in range(3, max_n + 1):
        k = n - 1
        # If time is running out, break
        if time.time() - start_time > 110:
            break
            
        # Generate combinations
        # To be efficient, we can prune if max_val is large, but 30 is small
        for combo in itertools.combinations(range(1, max_val + 1), k):
            U = [0] + list(combo)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    # Optional: Local search around best_set
    # But exhaustive search up to 30 is likely sufficient for n<=6
    
    return best_set
