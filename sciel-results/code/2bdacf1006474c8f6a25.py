import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    def is_valid_set(S):
        if not S:
            return True
        S_set = set(S)
        for x in S:
            others = [y for y in S if y != x]
            n_others = len(others)
            if n_others < 2:
                continue
                
            max_sum = sum(others)
            # Bitset DP
            dp = [0] * (n_others + 1)
            dp[0] = 1
            
            for a in others:
                # Update in reverse
                for k in range(n_others, 0, -1):
                    dp[k] |= (dp[k-1] << a)
            
            # Check violations
            for k in range(2, n_others + 1):
                target = k * x
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    candidates = list(range(1, 513))
    
    # Strategy 1: Ascending
    current_set = []
    for x in candidates:
        if time.time() > deadline:
            break
        new_set = current_set + [x]
        if is_valid_set(new_set):
            current_set = new_set
            if len(current_set) > len(best_set):
                best_set = current_set[:]
                
    # Strategy 2: Descending
    current_set = []
    for x in reversed(candidates):
        if time.time() > deadline:
            break
        new_set = current_set + [x]
        if is_valid_set(new_set):
            current_set = new_set
            if len(current_set) > len(best_set):
                best_set = current_set[:]
                
    # Strategy 3: Random
    # We will do multiple random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = []
        for x in candidates:
            if time.time() > deadline:
                break
            new_set = current_set + [x]
            if is_valid_set(new_set):
                current_set = new_set
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                    
    return best_set
