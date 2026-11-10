"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def check_valid(arr):
        """Check if all subset sums are distinct."""
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            if new_sums & sums:
                return False
            sums = sums | new_sums
        return len(sums) == 4096
    
    def fast_check(arr):
        """Fast collision check using set operations."""
        sums = {0}
        for x in arr:
            shifted = {s + x for s in sums}
            if shifted & sums:
                return False
            sums |= shifted
        return True
    
    def find_greedy_optimized(deadline):
        """Optimized greedy with better upper bound and collision detection."""
        arr = [1]
        sums = {0}
        current_max = 1
        
        for step in range(11):
            if time.time() > deadline:
                return None
            
            found = False
            # The next element must be > current_max and <= sum(arr) + 1
            # Actually, for distinct subset sums, we need x > max element so far
            # and x must not create collisions
            upper = sum(arr) + 1
            
            for x in range(current_max + 1, upper + 1):
                if time.time() > deadline:
                    return None
                
                # Quick collision check: x must not be representable as difference of two existing sums
                # More efficiently: check if any s + x is in sums
                collides = False
                for s in sums:
                    if (s + x) in sums:
                        collides = True
                        break
                
                if not collides:
                    new_sums = {s + x for s in sums}
                    sums |= new_sums
                    arr.append(x)
                    current_max = x
                    found = True
                    break
            
            if not found:
                # Fallback: use sum + 1 which always works
                x = sum(arr) + 1
                new_sums = {s + x for s in sums}
                sums |= new_sums
                arr.append(x)
                current_max = x
        
        return arr
    
    def backtracking_search(arr, sums, idx, deadline, best_result):
        """
        Backtracking search to minimize the largest element.
        arr: current array
        sums: set of all subset sums so far
        idx: next index to fill
        deadline: time limit
        best_result: best solution found so far (list or None)
        """
        if time.time() > deadline:
            return best_result
        
        if idx == 12:
            if best_result is None or max(arr) < max(best_result):
                return list(arr)
            return best_result
        
        # Lower bound on what the next element must be
        min_next = arr[-1] + 1 if arr else 1
        
        # Upper bound: we can't exceed current best max
        if best_result is not None:
            max_allowed = max(best_result)
        else:
            max_allowed = 1048576
        
        # Try values from min_next up to max_allowed
        # But we can prune: the remaining elements must fit in the range
        # For efficiency, try smaller values first
        
        for x in range(min_next, max_allowed + 1):
            if time.time() > deadline:
                break
            
            # Check collision: x must not create any duplicate subset sums
            collides = False
            for s in sums:
                if (s + x) in sums:
                    collides = True
                    break
            
            if collides:
                continue
            
            new_sums = {s + x for s in sums}
            new_arr = arr + [x]
            
            # Pruning: if x >= current best max, no point continuing with this branch
            # unless we can still improve with remaining elements
            if best_result is not None and x >= max(best_result):
                # Still try, but we know we won't improve on this branch's max
                # unless remaining elements are smaller (impossible since they must be > x)
                pass
            
            result = backtracking_search(new_arr, new_sums, idx + 1, deadline, best_result)
            if result is not None:
                best_result = result
        
        return best_result
    
    def local_search_improved(arr, deadline):
        """Improved local search to reduce the largest element."""
        arr = list(arr)
        best_max = max(arr)
        no_improve_count = 0
        
        while time.time() < deadline:
            improved = False
            
            # Strategy 1: Try reducing the largest element
            idx_max = max(range(12), key=lambda i: arr[i])
            old_val = arr[idx_max]
            
            # Try values from (previous element + 1) up to old_val - 1
            min_val = arr[idx_max - 1] + 1 if idx_max > 0 else 1
            
            for new_val in range(min_val, old_val):
                if time.time() > deadline:
                    break
                
                candidate = arr[:]
                candidate[idx_max] = new_val
                
                # Check distinctness
                if len(set(candidate)) != 12:
                    continue
                
                if fast_check(candidate):
                    arr = candidate
                    best_max = max(arr)
                    improved = True
                    no_improve_count = 0
                    break
            
            # Strategy 2: Try reducing other elements
            if not improved:
                for i in range(11, -1, -1):  # Try from largest index first
                    if time.time() > deadline:
                        break
                    
                    old_val = arr[i]
                    min_val = arr[i - 1] + 1 if i > 0 else 1
                    max_val = arr[i + 1] - 1 if i < 11 else best_max
                    
                    if max_val <= min_val:
                        continue
                    
                    # Try a few random values in the range
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        
                        new_val = random.randint(min_val, max_val)
                        if new_val == old_val:
                            continue
                        
                        candidate = arr[:]
                        candidate[i] = new_val
                        
                        if len(set(candidate)) != 12:
                            continue
                        
                        if fast_check(candidate):
                            arr = candidate
                            if max(arr) < best_max:
                                best_max = max(arr)
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
        
        return arr
    
    # Phase 1: Quick greedy to get a valid solution
    result = find_greedy_optimized(deadline - 15)
    
    if result is None or not check_valid(result):
        # Fallback: powers of 2
        return [1 << i for i in range(12)]
    
    if time.time() > deadline - 10:
        return result
    
    # Phase 2: Local search to improve
    improved = local_search_improved(result, deadline - 5)
    if improved and check_valid(improved) and max(improved) < max(result):
        result = improved
    
    # Phase 3: If time remains, try more targeted search
    if time.time() < deadline - 2:
        # Try to specifically reduce the maximum element
        current_max = max(result)
        idx_max = max(range(12), key=lambda i: result[i])
        
        for new_max in range(current_max - 1, 0, -1):
            if time.time() > deadline - 2:
                break
            
            # Try to find a valid set with this maximum
            # Use backtracking with this constraint
            # This is expensive, so only try a few values
            if new_max < current_max - 10:
                break
        
        # Final local search pass
        if time.time() < deadline - 1:
            improved2 = local_search_improved(result, deadline - 1)
            if improved2 and check_valid(improved2) and max(improved2) < max(result):
                result = improved2
    
    return result
# EVOLVE-BLOCK-END
