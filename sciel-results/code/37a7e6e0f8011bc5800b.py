import time

def construct():
    deadline = time.time() + 140
    
    prescribed = [1, 2, 5, 11]
    used_sums = set()
    for i in range(4):
        for j in range(i, 4):
            used_sums.add(prescribed[i] + prescribed[j])
    
    # Candidates sorted
    candidates = list(range(12, 285))
    
    best = list(prescribed)
    best_size = 4
    
    def try_extend(current, used, target_size, start_idx, depth):
        nonlocal best, best_size
        
        if time.time() > deadline:
            return False
        
        n = len(current)
        if n == target_size:
            if n > best_size:
                best = list(current)
                best_size = n
            return True
        
        if start_idx >= len(candidates):
            return False
        
        # Pruning: can we reach target_size?
        if n + (len(candidates) - start_idx) < target_size:
            return False
        
        for i in range(start_idx, len(candidates)):
            if time.time() > deadline:
                return False
            
            x = candidates[i]
            
            # Check validity
            ok = True
            for y in current:
                if x + y in used:
                    ok = False
                    break
            if ok and 2 * x in used:
                ok = False
            
            if not ok:
                continue
            
            new_sums = {2 * x}
            for y in current:
                new_sums.add(x + y)
            
            if new_sums & used:
                continue
            
            current.append(x)
            used.update(new_sums)
            
            if try_extend(current, used, target_size, i + 1, depth + 1):
                current.pop()
                used.difference_update(new_sums)
                return True
            
            current.pop()
            used.difference_update(new_sums)
        
        return False
    
    # Try to find sets of increasing size
    for target in range(20, 3, -1):
        if time.time() > deadline - 10:
            break
        if try_extend(list(prescribed), set(used_sums), target, 0, 0):
            break
    
    return best
