import math
import time
import itertools

def score_of(U):
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
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_set = [0, 1, 3, 5, 9]
    best_score = score_of(best_set)
    
    # Local search
    # Try adding elements
    # Try removing elements
    # Try swapping elements
    
    # Candidate pool for additions
    pool = list(range(0, 30))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        for x in pool:
            if x in best_set:
                continue
            new_set = best_set + [x]
            new_set.sort()
            s = score_of(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set
                improved = True
                # Expand pool if needed
                if max(best_set) > 20:
                    pool = list(range(0, max(best_set) + 10))
        
        # Try removing
        for x in list(best_set):
            if len(best_set) <= 2:
                continue
            new_set = [y for y in best_set if y != x]
            s = score_of(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set
                improved = True
        
        # Try swapping
        for i in range(len(best_set)):
            for x in pool:
                if x in best_set:
                    continue
                new_set = best_set.copy()
                new_set[i] = x
                new_set.sort()
                s = score_of(new_set)
                if s > best_score:
                    best_score = s
                    best_set = new_set
                    improved = True

    return best_set
