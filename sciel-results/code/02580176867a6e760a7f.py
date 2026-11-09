import time
import random

START_TIME = time.time()
DEADLINE = 85.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums_bitset(arr):
    n = len(arr)
    sums = [0] * (n + 1)
    sums[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            sums[k] |= (sums[k - 1] << x)
    return sums

def is_valid(arr):
    if not arr:
        return True
    sums = compute_sums_bitset(arr)
    n = len(arr)
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def greedy_extend(arr):
    arr = list(arr)
    used = set(arr)
    for c in range(1, 257):
        if is_done():
            break
        if c in used:
            continue
        new_arr = arr + [c]
        if is_valid(new_arr):
            arr = new_arr
            used.add(c)
    return arr

def construct():
    # Initial: powers of 2
    best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = greedy_extend(best)
    
    # Random restarts with local search
    for _ in range(50):
        if is_done():
            break
        if len(best) >= 60:
            break
        
        # Random initial set: pick 5-10 random distinct values
        size = random.randint(5, 10)
        trial = random.sample(range(1, 257), size)
        trial = greedy_extend(trial)
        
        if len(trial) > len(best):
            best = trial
    
    # Final extension
    best = greedy_extend(best)
    
    return [int(x) for x in best]
