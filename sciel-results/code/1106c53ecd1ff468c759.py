import itertools
import math
import time

def score_set(U):
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1:
        # log(1) = 0, score 1. But we want max.
        # Actually if ratio < 1, log is negative, score < 1.
        pass
    val = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_score = 0.0
    best_set = [0, 1, 3] # Seed
    
    # 1. Geometric Progressions
    for r in [2, 3, 4, 5]:
        for n in range(3, 10):
            U = [0]
            curr = 1
            for _ in range(n - 1):
                U.append(curr)
                curr *= r
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            
    # 2. Exhaustive search on small range
    # Search subsets of {0..15} of size 3 to 6
    # Must contain 0.
    # So choose k-1 from 1..15.
    
    range_max = 15
    candidates = list(range(1, range_max + 1))
    
    for k in range(3, 7):
        if time.time() > deadline:
            break
        # Choose k-1 elements from 1..15
        for combo in itertools.combinations(candidates, k - 1):
            U = [0] + list(combo)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
        
    # 3. Maybe slightly larger range for smaller k
    range_max2 = 25
    candidates2 = list(range(1, range_max2 + 1))
    for k in range(3, 5):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(candidates2, k - 1):
            U = [0] + list(combo)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    return best_set
