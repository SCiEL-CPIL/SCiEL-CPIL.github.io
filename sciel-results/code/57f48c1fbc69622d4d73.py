"""Program for maximizing 3-AP differences in a 32-element set within 1..512."""

import random
import time
import sys

def count_aps(s):
    """
    Calculate the number of distinct differences d > 0 such that
    a, a+d, a+2d are all in the set s.
    """
    s_set = set(s)
    diffs = set()
    # Optimize by checking pairs or triples?
    # With 32 elements, O(N^3) is 32^3 = 32768 operations.
    # This is very fast.
    for i in range(len(s)):
        a = s[i]
        for j in range(len(s)):
            if i == j: continue
            b = s[j]
            if b <= a: continue # a must be smallest
            d = b - a
            c = a + 2 * d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def construct():
    """
    Use simulated annealing to find a 32-element set in [1, 512]
    that maximizes the number of distinct 3-AP differences.
    """
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Initialize with a random set
    current_set = set(random.sample(range(1, MAX_VAL + 1), N))
    current_score = count_aps(list(current_set))
    
    best_set = list(current_set)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.01
    
    # Helper to get a random element and a new random value
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random element to change
        idx = random.randint(0, N - 1)
        old_val = list(current_set)[idx]
        
        # Generate a new value
        # Sometimes pick a random value, sometimes pick a value close to existing ones
        if random.random() < 0.5:
            new_val = random.randint(1, MAX_VAL)
        else:
            # Pick a value near an existing element to maintain structure
            ref_val = random.choice(list(current_set))
            offset = random.randint(-50, 50)
            new_val = ref_val + offset
            if new_val < 1:
                new_val = 1
            elif new_val > MAX_VAL:
                new_val = MAX_VAL
        
        if new_val in current_set and new_val != old_val:
            continue
            
        # Apply change
        current_set.remove(old_val)
        current_set.add(new_val)
        
        new_score = count_aps(list(current_set))
        
        # Acceptance probability
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(current_set)
        else:
            # Reject: revert change
            current_set.remove(new_val)
            current_set.add(old_val)
            
        T *= cooling_rate
        
    # Final check
    if count_aps(best_set) > best_score:
        best_score = count_aps(best_set)
        
    return best_set
