import time
import random
from itertools import combinations

def is_sidon(subset):
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    S = sorted(S)
    n = len(S)
    # Check from n down to 3
    # If we find a Sidon subset of size k, return k
    # We know size 3 always exists for n>=4? No, we proved size 2 is impossible for n>=4, 
    # meaning size 3 is the minimum possible max size.
    # Wait, if max size is 3, it means there IS a Sidon subset of size 3, 
    # and NO Sidon subset of size 4.
    
    for k in range(n, 2, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 3 # Since n>=4, there is always a Sidon subset of size 3? 
             # Let's verify. 
             # Is it possible that NO 3-subset is Sidon? 
             # That would mean every 3-subset is a 3-AP. 
             # We proved this is impossible for n=4. 
             # So for n=12, there is always a Sidon subset of size 3. 
             # So the return value is at least 3.

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    if best_score == 3:
        return best_set

    current_set = best_set[:]
    current_score = best_score
    
    for _ in range(200000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        new_val = random.randint(0, 128)
        
        if new_val == old_val or new_val in current_set:
            continue
            
        current_set[idx] = new_val
        new_score = max_sidon_size(current_set)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
                if best_score == 3:
                    break
        else:
            current_set[idx] = old_val
            
    return best_set
