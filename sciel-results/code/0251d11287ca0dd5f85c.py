import time
import random

def _is_valid_range(s):
    n = len(s)
    if n < 2:
        return True
    s_sorted = sorted(s)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + s_sorted[i]
        
    for k in range(1, n):
        max_k = prefix[n] - prefix[n - k]
        min_k1 = prefix[k + 1]
        if max_k >= min_k1:
            return False
    return True

def _get_conflicts_fast(s):
    n = len(s)
    if n < 2:
        return True, 0
    
    sums = [0] * (n + 1)
    sums[0] = 1
    
    for x in s:
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << x)
            
    conflict_pairs = 0
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                conflict_pairs += 1
                
    is_valid = (conflict_pairs == 0)
    return is_valid, conflict_pairs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    for n in range(60, 0, -1):
        if time.time() > deadline - 10.0:
            break
        for a in range(1, 257 - n + 1):
            s = list(range(a, a + n))
            if _is_valid_range(s):
                if len(s) > len(best_set):
                    best_set = s
                break
        
    if not best_set:
        best_set = [1, 2]
        
    current_set = best_set[:]
    current_valid, current_conflicts = _get_conflicts_fast(current_set)
    current_score = len(current_set) - 5.0 * current_conflicts
    
    best_score = current_score
    best_valid_set = current_set[:] if current_valid else []
    
    T = 50.0
    cooling_rate = 0.995
    min_T = 0.1
    
    s_set = set(current_set)
    available = [x for x in range(1, 257) if x not in s_set]
    
    while time.time() < deadline - 2.0:
        if T < min_T:
            T = 20.0
            
        move_type = random.random()
        
        new_set = current_set[:]
        new_set_set = s_set.copy()
        new_available = available[:] # Copying list of ~200 ints is fast
        
        if move_type < 0.5:
            if new_available:
                idx = random.randint(0, len(new_available) - 1)
                x = new_available[idx]
                new_set.append(x)
                new_set_set.add(x)
                new_available.pop(idx)
        elif move_type < 0.7:
            if len(new_set) > 1:
                idx = random.randint(0, len(new_set) - 1)
                x = new_set[idx]
                new_set.pop(idx)
                new_set_set.remove(x)
                new_available.append(x)
        else:
            if len(new_set) > 0 and new_available:
                idx = random.randint(0, len(new_set) - 1)
                old_x = new_set[idx]
                avail_idx = random.randint(0, len(new_available) - 1)
                new_x = new_available[avail_idx]
                
                new_set[idx] = new_x
                new_set_set.remove(old_x)
                new_set_set.add(new_x)
                
                new_available[avail_idx] = old_x
                
        new_valid, new_conflicts = _get_conflicts_fast(new_set)
        new_score = len(new_set) - 5.0 * new_conflicts
        
        delta = new_score - current_score
        
        if delta > 0:
            current_set = new_set
            s_set = new_set_set
            available = new_available
            current_score = new_score
            current_valid = new_valid
            current_conflicts = new_conflicts
            
            if current_valid and len(current_set) > len(best_valid_set):
                best_score = current_score
                best_valid_set = current_set[:]
        else:
            if random.random() < pow(2.71828, delta / T):
                current_set = new_set
                s_set = new_set_set
                available = new_available
                current_score = new_score
                current_valid = new_valid
                current_conflicts = new_conflicts
                
        T *= cooling_rate
        
    if best_valid_set:
        return list(best_valid_set)
    else:
        return [1, 2]
