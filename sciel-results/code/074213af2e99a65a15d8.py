import time
from itertools import combinations
import random

def is_sidon(subset):
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

def count_sidon_4(candidate):
    count = 0
    for subset in combinations(candidate, 4):
        if is_sidon(subset):
            count += 1
    return count

def max_sidon_size(candidate_set):
    n = len(candidate_set)
    for k in range(n, 0, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    random.seed(999)
    
    best_set = None
    best_count = float('inf')
    
    # Phase 1: Random search
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        s = random.sample(range(129), 12)
        cnt = count_sidon_4(s)
        if cnt == 0:
            return s
        if cnt < best_count:
            best_count = cnt
            best_set = list(s)
        
        # Periodically check elapsed time
        if iterations % 100 == 0 and time.time() > deadline:
            break
    
    # Phase 2: Local search on best found
    if best_set is not None and time.time() < deadline - 10:
        current = list(best_set)
        current_count = best_count
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            i = random.randint(0, 11)
            old_val = current[i]
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            current[i] = new_val
            new_count = count_sidon_4(current)
            if new_count < current_count:
                current_count = new_count
                if current_count == 0:
                    return current
                if current_count < best_count:
                    best_count = current_count
                    best_set = list(current)
            else:
                current[i] = old_val
    
    if best_set is not None:
        return best_set
    return list(range(12))
