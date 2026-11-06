import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Try target sizes from 48 down to 1
    for target_k in range(48, 0, -1):
        if time.time() > deadline - 5:
            break
            
        found = None
        
        # Adaptive restarts
        # Start with more restarts, decrease if time is low
        while time.time() < deadline - 5:
            # Initialize
            initial_set = set(random.sample(range(1, 513), target_k))
            
            # Local Search
            current_set = initial_set
            current_conflicts = count_conflicts(current_set)
            
            if current_conflicts == 0:
                found = current_set
                break
                
            best_set = current_set.copy()
            best_conflicts = current_conflicts
            
            # Search loop
            # Limit steps based on remaining time?
            # Just use a fixed limit, the outer loop will break if time runs out
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                to_remove = random.choice(list(current_set))
                candidates = [x for x in range(1, 513) if x not in current_set]
                if not candidates:
                    break
                to_add = random.choice(candidates)
                
                new_set = current_set.copy()
                new_set.remove(to_remove)
                new_set.add(to_add)
                
                new_conflicts = count_conflicts(new_set)
                
                if new_conflicts < current_conflicts:
                    current_set = new_set
                    current_conflicts = new_conflicts
                    if current_conflicts == 0:
                        found = current_set
                        break
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        best_set = current_set.copy()
                else:
                    if random.random() < 0.05:
                        current_set = new_set
                        current_conflicts = new_conflicts
                        
            if found:
                break
            # If we didn't find it, and time is low, break outer loop too?
            # The outer while condition handles it.

        if found:
            return list(found)
            
    # Fallback: Greedy
    selected = []
    for x in range(1, 513):
        candidate = set(selected) | {x}
        if count_conflicts(candidate) == 0:
            selected.append(x)
        if time.time() > deadline:
            break
    return selected

def count_conflicts(V):
    if not V:
        return 0
    
    V_list = list(V)
    n = len(V_list)
    conflicts = 0
    
    OFFSET = 25000
    
    for a in V_list:
        diffs = [x - a for x in V_list if x != a]
        m = len(diffs)
        if m < 2:
            continue
            
        dp = [0] * (m + 1)
        dp[0] = 1 << OFFSET
        
        for d in diffs:
            shift = d + OFFSET
            for c in range(m - 1, -1, -1):
                if dp[c]:
                    dp[c + 1] |= (dp[c] << shift)
                    
        mask = 1 << OFFSET
        for c in range(2, m + 1):
            if dp[c] & mask:
                conflicts += 1
                break
                
    return conflicts
