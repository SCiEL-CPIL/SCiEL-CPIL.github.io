import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    initial_set = [1, 2, 5, 11]
    initial_sums = set()
    for i, x in enumerate(initial_set):
        for y in initial_set[i:]:
            initial_sums.add(x + y)
    
    def make_bitset(sums):
        b = 0
        for s in sums:
            b |= (1 << s)
        return b
        
    current_sums_bitset = make_bitset(initial_sums)
    candidates = list(range(12, 285))
    n_cands = len(candidates)
    
    best_set = initial_set[:]
    best_size = len(best_set)
    node_count = 0
    
    def dfs(current_set, current_sums_bitset, start_idx, target_size):
        nonlocal best_set, best_size, node_count
        
        node_count += 1
        if node_count % 500 == 0:
            if time.time() > deadline:
                return False

        if len(current_set) == target_size:
            if len(current_set) > best_size:
                best_set = current_set[:]
                best_size = len(current_set)
            return True
            
        if time.time() > deadline:
            return False
            
        # Pruning: if not enough candidates left
        if n_cands - start_idx < target_size - len(current_set):
            return False
            
        for i in range(start_idx, n_cands):
            x = candidates[i]
            
            # Calculate new sums
            new_sums = 0
            for y in current_set:
                s = x + y
                new_sums |= (1 << s)
            s2 = 2 * x
            new_sums |= (1 << s2)
            
            # Check conflict
            if current_sums_bitset & new_sums:
                continue
                
            # No conflict, recurse
            next_set = current_set + [x]
            next_sums_bitset = current_sums_bitset | new_sums
            
            if dfs(next_set, next_sums_bitset, i + 1, target_size):
                return True
                
        return False

    # Try target sizes from 19 down to 10
    for target_size in range(19, 9, -1):
        if time.time() > deadline:
            break
            
        found = dfs(initial_set, current_sums_bitset, 0, target_size)
        
        if found:
            return best_set
            
    return best_set
