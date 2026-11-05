import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit
    
    N = 22
    candidates = list(range(1, N + 1))
    
    def is_dissociated_fast(vals):
        """
        Fast check if all subset sums of vals are distinct.
        Uses incremental set building with early exit.
        """
        k = len(vals)
        if k == 0:
            return True
        if k == 1:
            return True
            
        sums = {0}
        for i, v in enumerate(vals):
            # Check if v creates a collision
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
            if len(sums) < (1 << (i + 1)):
                return False
        return True

    def backtrack(current_vals, current_elems, available, target_size, time_limit):
        """
        Recursive backtracking with incremental dissociation check.
        current_vals: list of L/x values for current set
        current_elems: list of actual elements
        """
        if time.time() - start_time > time_limit:
            return None
            
        if len(current_elems) == target_size:
            return current_elems
            
        # Pruning: if not enough elements left
        if len(current_elems) + len(available) < target_size:
            return None
            
        # Heuristic ordering: try elements that are "farther apart" in reciprocal space first
        # This tends to avoid collisions
        for i, x in enumerate(available):
            new_elem = x
            new_val = current_vals[0] // x if current_vals else x  # placeholder
            
            # Incremental check: add x to current set
            # Compute new LCM and new value
            L = 1
            for e in current_elems:
                L = math.lcm(L, e)
            if current_elems:
                L_new = math.lcm(L, x)
            else:
                L_new = x
                
            # Build new vals list
            if current_elems:
                new_vals = [L_new // e for e in current_elems]
            else:
                new_vals = []
            new_vals.append(L_new // x)
            
            # Fast dissociation check on new_vals
            if is_dissociated_fast(new_vals):
                result = backtrack(new_vals, current_elems + [x], available[i+1:], target_size, time_limit)
                if result is not None:
                    return result
                    
        return None

    def backtrack_optimized(available, target_size, time_limit):
        """
        Optimized backtracking that maintains state incrementally.
        """
        if time.time() - start_time > time_limit:
            return None
            
        if len(available) < target_size:
            return None
            
        # Try adding each element from available
        for i, x in enumerate(available):
            # Check if {x} alone is dissociated (trivially true for single element)
            # Then try to extend
            
            # Use a helper that builds up the set
            def build_set(current_elems, current_L, remaining, target):
                if time.time() - start_time > time_limit:
                    return None
                    
                if len(current_elems) == target:
                    return current_elems
                    
                if len(remaining) < target - len(current_elems):
                    return None
                    
                for j, y in enumerate(remaining):
                    # Try adding y
                    L_new = math.lcm(current_L, y) if current_L else y
                    
                    # Build vals for current_elems + [y]
                    vals = []
                    for e in current_elems:
                        vals.append(L_new // e)
                    vals.append(L_new // y)
                    
                    # Fast check
                    if is_dissociated_fast(vals):
                        result = build_set(current_elems + [y], L_new, remaining[j+1:], target)
                        if result is not None:
                            return result
                            
                return None
            
            result = build_set([x], x, available[i+1:], target_size)
            if result is not None:
                return result
                
        return None

    best_result = None
    best_size = 0
    
    # Try size 17 with multiple random orders
    target_size = 17
    attempts = 0
    max_attempts = 10000  # Cap attempts
    
    while time.time() - start_time < timeout - 3.0:
        attempts += 1
        if attempts > max_attempts:
            break
            
        # Shuffle candidates to change search path
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        # Run backtracking with this order
        result = backtrack_optimized(shuffled_candidates, target_size, timeout - 3.0)
        
        if result is not None:
            best_result = result
            best_size = target_size
            break  # Found the maximum we are targeting, return immediately
    
    # If 17 was not found, try 16 (should be very fast to find one)
    if best_size < 16:
        target_size = 16
        for _ in range(50):
            if time.time() - start_time > timeout - 1.0:
                break
            shuffled_candidates = candidates[:]
            random.shuffle(shuffled_candidates)
            result = backtrack_optimized(shuffled_candidates, target_size, timeout - 1.0)
            if result is not None:
                best_result = result
                best_size = target_size
                break
        
        # Fallback to greedy construction if search fails
        if best_result is None:
            S = []
            for x in range(22, 0, -1):  # Descending: smaller reciprocals first
                # Check if adding x keeps the set dissociated
                if not S:
                    S.append(x)
                else:
                    L = 1
                    for e in S:
                        L = math.lcm(L, e)
                    L_new = math.lcm(L, x)
                    vals = [L_new // e for e in S] + [L_new // x]
                    if is_dissociated_fast(vals):
                        S.append(x)
            best_result = S
            best_size = len(S)
            
    return best_result
