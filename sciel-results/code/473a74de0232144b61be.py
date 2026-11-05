import random
import itertools
import time

def is_ap_free(subset):
    s = set(subset)
    for a, c in itertools.combinations(s, 2):
        if (a + c) % 2 == 0:
            b = (a + c) // 2
            if b in s and b != a and b != c:
                return False
    return True

def max_ap_free_size(S):
    S = list(S)
    for k in range(len(S), 2, -1):
        for subset in itertools.combinations(S, k):
            if is_ap_free(subset):
                return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 60
    
    best_set = list(range(12))
    best_size = max_ap_free_size(best_set)
    
    if best_size <= 3:
        return best_set

    # Local search
    current_set = list(range(12))
    current_size = best_size
    
    for _ in range(10000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random element to replace
        i = random.randint(0, 11)
        old_val = current_set[i]
        new_val = random.randint(0, 128)
        if new_val in current_set and new_val != old_val:
            continue
            
        current_set[i] = new_val
        new_size = max_ap_free_size(current_set)
        
        if new_size < current_size:
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_set = current_set.copy()
                if best_size <= 3:
                    break
        else:
            # Revert or keep? Keep for random walk
            # Revert to avoid getting stuck
            current_set[i] = old_val
            
    return best_set
