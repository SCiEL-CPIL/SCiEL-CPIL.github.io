def construct():
    global _START
    _START = time.time()
    
    L = reduce(_lcm, range(1, 25))
    weights = [0] + [L // i for i in range(1, 25)]
    
    best_set = _greedy_decreasing_int()
    best_size = len(best_set)
    
    current_set = []
    
    def dfs(current_a, current_sums):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
            
        if len(current_set) + current_a <= best_size:
            return
            
        if current_a == 0:
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(current_set)
            return
        
        target = weights[current_a]
        
        # Option 1: Include current_a
        if target not in current_sums:
            new_sums = set(current_sums)
            for s in current_sums:
                ns = s + target
                if ns <= L:
                    new_sums.add(ns)
            
            current_set.append(current_a)
            dfs(current_a - 1, new_sums)
            current_set.pop()
            
            if _deadline_hit():
                return
            
        # Option 2: Exclude current_a
        if len(current_set) + (current_a - 1) > best_size:
            dfs(current_a - 1, current_sums)
            
    initial_sums = {0}
    dfs(24, initial_sums)
    
    if not best_set:
        best_set = [24]
        
    return best_set
