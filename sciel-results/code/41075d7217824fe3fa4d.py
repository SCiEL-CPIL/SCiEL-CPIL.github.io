import time
import random

def _check_and_add(S, sums, x):
    """
    Returns True if x can be added to S, and updates S and sums if so.
    S is a list of sorted integers.
    sums is a set of all triple sums.
    """
    new_sums = set()
    
    # 1. 3x
    s = 3 * x
    if s in sums:
        return False
    new_sums.add(s)
    
    # 2. 2x + y
    for y in S:
        s = 2 * x + y
        if s in sums or s in new_sums:
            return False
        new_sums.add(s)
        
    # 3. x + y + z
    n = len(S)
    for i in range(n):
        y = S[i]
        for j in range(i, n):
            z = S[j]
            s = x + y + z
            if s in sums or s in new_sums:
                return False
            new_sums.add(s)
            
    # Valid
    S.append(x)
    sums.update(new_sums)
    return True

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = []
    
    # Strategy 1: Smallest valid greedy
    S1 = []
    sums1 = set()
    for x in range(1, 4097):
        if time.time() - start_time > time_limit:
            break
        _check_and_add(S1, sums1, x)
    if len(S1) > len(best_set):
        best_set = S1[:]

    # Strategy 2: Largest valid greedy
    S2 = []
    sums2 = set()
    for x in range(4096, 0, -1):
        if time.time() - start_time > time_limit:
            break
        _check_and_add(S2, sums2, x)
    if len(S2) > len(best_set):
        best_set = S2[:]

    # Strategy 3: Randomized greedy
    # We run a few iterations
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        S3 = []
        sums3 = set()
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        i = 0
        while i < len(candidates) and time.time() - start_time < time_limit:
            x = candidates[i]
            if _check_and_add(S3, sums3, x):
                # Success, keep x in candidates? No, it's used.
                # We don't need to remove it from candidates explicitly if we just advance i,
                # but we should remove invalid ones to speed up?
                # Actually, if x is valid, we add it.
                # If x is invalid, we skip it.
                # We can just iterate through the shuffled list once.
                pass
            i += 1
            
        if len(S3) > len(best_set):
            best_set = S3[:]

    return best_set
