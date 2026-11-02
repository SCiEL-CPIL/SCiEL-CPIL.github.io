import math
import time

def score(U):
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
        
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # 1. Brute force small sets
    # Size 3: {0, a, b}
    for a in range(1, 20):
        for b in range(a + 1, 20):
            U = [0, a, b]
            s = score(U)
            if s > best_score:
                best_score = s
                best_set = U
            if time.time() > deadline:
                return best_set
                
    # Size 4: {0, a, b, c}
    for a in range(1, 15):
        for b in range(a + 1, 15):
            for c in range(b + 1, 15):
                U = [0, a, b, c]
                s = score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
            if time.time() > deadline:
                return best_set
                
    # 2. Local search from best_set
    # Try adding elements
    current_set = best_set[:]
    current_score = best_score
    
    # Simple local search
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Try adding a random element
        new_set = current_set[:]
        x = max(current_set) + 1
        # Or random
        import random
        x = random.randint(1, max(current_set) * 2)
        if x not in new_set:
            new_set.append(x)
            s = score(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
        
        # Try removing an element (if size > 1)
        if len(current_set) > 1:
            new_set = current_set[:]
            idx = random.randint(1, len(new_set) - 1) # Don't remove 0
            new_set.pop(idx)
            s = score(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
                
        # Try replacing an element
        if len(current_set) > 1:
            new_set = current_set[:]
            idx = random.randint(1, len(new_set) - 1)
            old_val = new_set[idx]
            new_val = random.randint(1, max(new_set) * 2)
            if new_val not in new_set:
                new_set[idx] = new_val
                s = score(new_set)
                if s > current_score:
                    current_set = new_set
                    current_score = s

    if current_score > best_score:
        best_set = current_set
        best_score = current_score

    return sorted(best_set)
