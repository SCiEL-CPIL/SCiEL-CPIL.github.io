import time
import random

def _greedy_smallest():
    values = []
    sums = set()
    for x in range(1, 257):
        if values and x <= values[-1]:
            continue
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_largest():
    values = []
    sums = set()
    last = 0
    while True:
        found_c = None
        found_tails = None
        for c in range(256, last, -1):
            tails = [c]
            total = c
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                found_c = c
                found_tails = tails
                break
        if found_c is None:
            break
        values.append(found_c)
        sums.update(found_tails)
        last = found_c
    return values

def _greedy_random():
    values = []
    sums = set()
    last = 0
    while True:
        valid_candidates = []
        for c in range(last + 1, 257):
            tails = [c]
            total = c
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                valid_candidates.append((c, tails))
        
        if not valid_candidates:
            break
        
        # Pick from the upper half of valid candidates
        half = len(valid_candidates) // 2
        # Ensure index is within bounds
        start_idx = half if half < len(valid_candidates) else len(valid_candidates) - 1
        end_idx = len(valid_candidates) - 1
        if start_idx > end_idx:
            start_idx = end_idx
            
        chosen_idx = random.randint(start_idx, end_idx)
        chosen_c, chosen_tails = valid_candidates[chosen_idx]
        
        values.append(chosen_c)
        sums.update(chosen_tails)
        last = chosen_c
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    candidates = []
    
    # 1. Greedy Smallest
    v1 = _greedy_smallest()
    candidates.append(v1)
    
    if time.time() > deadline:
        return max(candidates, key=len)
        
    # 2. Greedy Largest
    v2 = _greedy_largest()
    candidates.append(v2)
    
    if time.time() > deadline:
        return max(candidates, key=len)
        
    # 3. Randomized Greedy (Upper Half)
    random.seed(42)
    v3 = _greedy_random()
    candidates.append(v3)
    
    # 4. Randomized Greedy (Lower Half) - just in case
    random.seed(123)
    # Modify _greedy_random to take a bias parameter? 
    # For now, just run one more random variant with different seed
    v4 = _greedy_random()
    candidates.append(v4)

    return max(candidates, key=len)
