import numpy as np
import random
import time

def construct():
    N = 1000
    max_sum = 2 * N
    
    def greedy_random(order):
        S_list = []
        S_set = set()
        S_bits = 0
        BadSums = 0
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        for x in order:
            if x in S_set:
                continue
            # Check validity
            if counts[2 * x] >= 2:
                continue
            if (S_bits & (BadSums >> x)) != 0:
                continue
            
            # Add x
            S_list.append(x)
            S_set.add(x)
            S_bits |= (1 << x)
            
            # Update counts
            if len(S_list) > 1:
                others = np.array(S_list[:-1])
                sums = np.concatenate(([2 * x], x + others))
            else:
                sums = np.array([2 * x])
                
            old_counts = counts[sums].copy()
            counts[sums] += 1
            
            became_bad = sums[old_counts == 1]
            if len(became_bad) > 0:
                for idx in became_bad:
                    BadSums |= (1 << idx)
                    
        return S_list, S_set, S_bits, BadSums, counts

    best_S_list = []
    best_S_set = set()
    best_S_bits = 0
    best_BadSums = 0
    best_counts = None
    best_size = 0
    
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Greedy
    for i in range(50):
        if time.time() - start_time > time_limit - 30:
            break
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S_list, S_set, S_bits, BadSums, counts = greedy_random(candidates)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S_list = S_list
            best_S_set = S_set
            best_S_bits = S_bits
            best_BadSums = BadSums
            best_counts = counts.copy()
            
    # Phase 2: Local Search
    S_list = best_S_list
    S_set = best_S_set
    S_bits = best_S_bits
    BadSums = best_BadSums
    counts = best_counts
    
    def can_add(x):
        if counts[2 * x] >= 2:
            return False
        if (S_bits & (BadSums >> x)) != 0:
            return False
        return True

    def add(x):
        nonlocal S_bits, BadSums
        S_list.append(x)
        S_set.add(x)
        S_bits |= (1 << x)
        
        if len(S_list) > 1:
            others = np.array(S_list[:-1])
            sums = np.concatenate(([2 * x], x + others))
        else:
            sums = np.array([2 * x])
            
        old_counts = counts[sums].copy()
        counts[sums] += 1
        
        became_bad = sums[old_counts == 1]
        if len(became_bad) > 0:
            for idx in became_bad:
                BadSums |= (1 << idx)

    def remove(x):
        nonlocal S_bits, BadSums
        S_list.remove(x)
        S_set.remove(x)
        S_bits &= ~(1 << x)
        
        if len(S_list) > 0:
            others = np.array(S_list)
            sums = np.concatenate(([2 * x], x + others))
        else:
            sums = np.array([2 * x])
            
        old_counts = counts[sums].copy()
        counts[sums] -= 1
        
        became_good = sums[old_counts == 2]
        if len(became_good) > 0:
            for idx in became_good:
                BadSums &= ~(1 << idx)

    # Local search loop
    while time.time() - start_time < time_limit:
        # Try add
        if len(S_list) < N:
            x = random.randint(1, N)
            if x not in S_set:
                if can_add(x):
                    add(x)
                    continue
        
        # Try swap
        if S_list:
            y = random.choice(S_list)
            x = random.randint(1, N)
            if x == y or x in S_set:
                continue
            
            # Attempt swap
            remove(y)
            if can_add(x):
                add(x)
            else:
                add(y)

    return S_list
