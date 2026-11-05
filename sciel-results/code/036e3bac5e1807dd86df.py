import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    LCM = 232792560
    elements = list(range(1, 23))
    weights = [LCM // x for x in elements]
    n = 22
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    def try_add(subset_sums, w):
        shifted = set(s + w for s in subset_sums)
        if shifted & subset_sums:
            return None
        return subset_sums | shifted
    
    current_elements = []
    
    def dfs(idx, current_sums):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if len(current_elements) + (n - idx) <= best_size:
            return
        
        if idx == n:
            if len(current_elements) > best_size:
                best_size = len(current_elements)
                best_set = list(current_elements)
                # If we found 16 or more, we can stop
                if best_size >= 16:
                    return
            return
        
        # Try include first
        w = weights[idx]
        new_sums = try_add(current_sums, w)
        if new_sums is not None:
            current_elements.append(elements[idx])
            dfs(idx + 1, new_sums)
            current_elements.pop()
        
        if time.time() > deadline:
            return
        
        # Try skip
        dfs(idx + 1, current_sums)
    
    dfs(0, {0})
    
    return best_set
