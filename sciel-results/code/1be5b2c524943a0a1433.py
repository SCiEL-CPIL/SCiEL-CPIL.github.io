import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = 1.0
    min_d = 5
    max_d = 10000000
    max_terms = 256
    
    best_solution = None
    best_k = max_terms + 1
    
    def max_sum(d, m):
        if m <= 0:
            return 0.0
        s = 0.0
        for i in range(m):
            s += 1.0 / (d + i)
        return s

    def search(current_sum, last_d, terms_used, path):
        nonlocal best_solution, best_k, deadline
        
        if time.time() > deadline:
            return
        
        if terms_used >= best_k:
            return
            
        if abs(current_sum - target) < 1e-12:
            if terms_used < best_k:
                best_k = terms_used
                best_solution = list(path)
            return
        
        if current_sum > target:
            return
            
        rem_terms = max_terms - terms_used
        if rem_terms <= 0:
            return
            
        inv_rem = 1.0 / (target - current_sum)
        low_d = max(last_d + 1, int(math.ceil(inv_rem)))
        
        if low_d > max_d:
            return

        max_possible_add = 1.0 / low_d + max_sum(low_d + 1, rem_terms - 1)
        if current_sum + max_possible_add < target:
            return

        d = low_d
        while d <= max_d:
            if time.time() > deadline:
                return
            
            if current_sum + 1.0/d + max_sum(d+1, rem_terms-1) < target:
                break
            
            path.append(d)
            search(current_sum + 1.0/d, d, terms_used + 1, path)
            path.pop()
            
            d += 1

    search(0.0, min_d - 1, 0, [])
    
    if best_solution is None:
        # Fallback
        target_dp, reached = 2520, {0: []}
        for d in range(5, target_dp + 1):
            if target_dp % d:
                continue
            w = target_dp // d
            for total, values in list(reached.items()):
                if total + w <= target_dp and total + w not in reached:
                    reached[total + w] = values + [d]
            if target_dp in reached:
                return reached[target_dp]
        raise RuntimeError("no baseline representation")

    return best_solution
