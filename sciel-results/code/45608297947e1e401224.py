import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for 90s limit
    
    N = 24
    candidates = list(range(N, 0, -1))
    
    best_set = []
    
    def is_subset_sum(target, items):
        if not items:
            return False
        if target in items:
            return True
        total = sum(items)
        if total < target:
            return False
        if min(items) > target:
            return False
            
        items_sorted = sorted(items, reverse=True)
        
        def helper(idx, current_sum):
            if time.time() > deadline:
                return False
            if current_sum == target:
                return True
            if idx == len(items_sorted):
                return False
            if current_sum > target:
                return False
            
            if helper(idx + 1, current_sum + items_sorted[idx]):
                return True
            if helper(idx + 1, current_sum):
                return True
            return False
        
        return helper(0, Fraction(0))

    def solve(idx, current_set):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if len(current_set) + (N - idx) <= len(best_set):
            return
            
        if idx == N:
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            return
            
        x = candidates[idx]
        
        valid_to_add = True
        if current_set:
            rec_x = Fraction(1, x)
            others_recs = [Fraction(1, y) for y in current_set]
            
            if is_subset_sum(rec_x, others_recs):
                valid_to_add = False
            
            if valid_to_add:
                for y in current_set:
                    rec_y = Fraction(1, y)
                    if rec_y > rec_x:
                        target_diff = rec_y - rec_x
                        others_for_y = [Fraction(1, z) for z in current_set if z != y]
                        if is_subset_sum(target_diff, others_for_y):
                            valid_to_add = False
                            break
        
        if valid_to_add:
            current_set.append(x)
            solve(idx + 1, current_set)
            current_set.pop()
            
        solve(idx + 1, current_set)

    solve(0, [])
    
    if not best_set:
        return [24]
        
    return best_set
