import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    TIME_LIMIT = 100  # Seconds
    
    best_solution = [0]
    best_max_val = float('inf')
    
    target_diffs = set(range(1, 31))
    
    def is_valid_insertion(a, x, used_diffs):
        """
        Check if inserting x into sorted list a is valid.
        Returns True if valid.
        """
        if x in a:
            return False
        
        # Check differences with all elements
        for y in a:
            d = abs(x - y)
            if d == 0:
                continue
            if d in used_diffs:
                return False
        
        return True

    def dfs(current_set, used_diffs):
        nonlocal best_solution, best_max_val, start_time, TIME_LIMIT
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Prune
        if current_set[-1] >= best_max_val:
            return
            
        if used_diffs.issuperset(target_diffs):
            if current_set[-1] < best_max_val:
                best_max_val = current_set[-1]
                best_solution = list(current_set)
            return

        if len(current_set) >= 32:
            return

        missing = target_diffs - used_diffs
        if not missing:
            return
            
        d_min = min(missing)
        
        # Generate candidates
        candidates = set()
        for a in current_set:
            x1 = a + d_min
            if x1 > 0: # Elements must be distinct, 0 is in set. x1=0 is invalid if 0 in set.
                candidates.add(x1)
            x2 = a - d_min
            if x2 >= 0:
                candidates.add(x2)
        
        # Filter and sort candidates
        # Priority: 
        # 1. x < max(current_set) (does not increase span)
        # 2. x > max(current_set) (increases span, prefer smallest)
        
        max_val = current_set[-1]
        internal_candidates = [x for x in candidates if x < max_val and x not in current_set]
        external_candidates = [x for x in candidates if x > max_val and x not in current_set]
        
        # Sort internal? Doesn't matter much for max, but might matter for future. 
        # Let's just iterate.
        # Sort external by value (smallest first) to minimize span growth.
        external_candidates.sort()
        
        # Try internal first
        for x in internal_candidates:
            if not is_valid_insertion(current_set, x, used_diffs):
                continue
            
            new_diffs = set(used_diffs)
            valid = True
            for y in current_set:
                d = abs(x - y)
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Insert x into sorted set
            # Since we are in Python, list insertion is O(n), but n is small (<=32)
            new_set = current_set + [x]
            new_set.sort()
            
            dfs(new_set, new_diffs)
            
        # Try external if no internal worked (or to explore further)
        for x in external_candidates:
            if not is_valid_insertion(current_set, x, used_diffs):
                continue
            
            new_diffs = set(used_diffs)
            valid = True
            for y in current_set:
                d = abs(x - y)
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            new_set = current_set + [x]
            # x > max, so it's already sorted if we append
            # new_set.sort() # not needed
            
            dfs(new_set, new_diffs)
            
    dfs([0], set())
    
    return best_solution
