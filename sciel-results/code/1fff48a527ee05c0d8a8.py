import time
from itertools import combinations

def construct():
    """
    Return exactly 12 distinct positive integers <= 1048576 such that all 2^12 = 4096
    subset sums (including the empty subset) are distinct. Minimize the largest element.
    
    Uses backtracking search with pruning to find a set with minimal maximum element.
    The key insight: we don't need superincreasing sequences (each > sum of all previous).
    We just need all subset sums to be distinct, which allows much smaller max values.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s limit
    
    n = 12
    MAX_VAL = 1048576
    
    # Fallback: powers of 2 (guaranteed valid, max = 2048)
    fallback = [1 << i for i in range(n)]
    best_solution = fallback
    best_max = max(fallback)
    
    # Backtracking search
    # elements: list of chosen elements so far
    # sums: set of all subset sums of elements
    # pos: current position (0-indexed)
    
    def backtrack(elements, sums, pos, current_max):
        nonlocal best_solution, best_max, deadline, start_time
        
        # Check time
        if time.time() > deadline:
            return False
        
        # If we've placed all n elements, we have a valid solution
        if pos == n:
            if current_max < best_max:
                best_max = current_max
                best_solution = list(elements)
            return True
        
        # Pruning: if current_max is already >= best_max, no point continuing
        # (since we're adding more positive elements, max will only stay or increase)
        # Actually, current_max is the max so far; new elements could be smaller
        # But we want to minimize the final max, so if current_max >= best_max,
        # the final max will be >= current_max >= best_max, so prune.
        if current_max >= best_max:
            return False
        
        # Determine range for next element
        # Must be positive, distinct from existing, and <= some reasonable bound
        # Lower bound: at least 1, but also must not create duplicate sums
        # Upper bound: we can try up to best_max - 1 (since we want to improve)
        # Actually, the new element just needs to be <= best_max - 1 to potentially improve
        
        # For efficiency, try smaller values first
        # The new element a must satisfy: a + s1 != s2 for all s1, s2 in sums
        # Equivalently, a is not in {s2 - s1 : s1, s2 in sums, s2 > s1}
        # And a must not be in elements (distinctness)
        
        # Compute forbidden values: a such that a + s1 == s2 for some s1, s2 in sums
        # i.e., a = s2 - s1 where s2 > s1, and a > 0
        # Also a must not equal any existing element
        
        existing = set(elements)
        
        # Start from 1 and go up, but skip forbidden values
        # To be efficient, we can compute the set of forbidden differences
        # But for small pos, sums is small, so direct check is fine
        
        found = False
        # Try values from 1 upward, up to best_max - 1
        # Optimization: we know the minimum possible value
        # The new element must be at least 1
        # And it must not create collisions
        
        # For early positions, try a bounded range
        # Position 0: try 1
        # Position 1: try 2, 3, ...
        # etc.
        
        # Upper bound for this position: best_max - 1
        upper = best_max - 1
        if upper < 1:
            return False
        
        # Try candidate values
        # To speed up, we can skip values that are clearly forbidden
        # Compute forbidden set: {s2 - s1 for s1 in sums for s2 in sums if s2 > s1}
        # But this could be large. For small pos, it's manageable.
        
        if len(sums) <= 256:  # Early positions, compute forbidden set
            forbidden = set()
            sum_list = list(sums)
            for s1 in sum_list:
                for s2 in sum_list:
                    if s2 > s1:
                        d = s2 - s1
                        if d > 0 and d <= upper and d not in existing:
                            forbidden.add(d)
        
        for a in range(1, upper + 1):
            if time.time() > deadline:
                return False
            
            if a in existing:
                continue
            
            # Check if a creates any duplicate subset sum
            # New sums would be: sums union (a + s for s in sums)
            # Collision if a + s1 == s2 for some s1, s2 in sums, s1 != s2
            # Or equivalently, a is in {s2 - s1 : s1, s2 in sums}
            
            valid = True
            if len(sums) <= 512:
                # Direct check
                new_sums = set()
                for s in sums:
                    ns = a + s
                    if ns in sums:
                        valid = False
                        break
                    new_sums.add(ns)
                if valid:
                    # Also check no duplicates in new_sums themselves
                    # But since a + s1 == a + s2 implies s1 == s2, no issue
                    # And we checked a + s not in sums
                    elements.append(a)
                    new_sums_full = sums.union(new_sums)
                    found = backtrack(elements, new_sums_full, pos + 1, max(current_max, a))
                    elements.pop()
                    if found:
                        return True
            else:
                # For larger sums, use the forbidden difference check
                # a is valid if a is not in {s2 - s1 : s1, s2 in sums, s1 != s2}
                # and a + s not in sums for all s in sums
                # The second condition is equivalent to a not in {s2 - s1 : s1, s2 in sums}
                # since if a + s1 = s2, then a = s2 - s1
                # So just check if a is a difference of two elements in sums
                # But we need to be careful: a + s1 = s2 means a = s2 - s1
                # This is the same as a being in the difference set
                
                # Check by iterating
                valid = True
                for s in sums:
                    if (a + s) in sums:
                        valid = False
                        break
                if valid:
                    new_sums = {a + s for s in sums}
                    new_sums_full = sums.union(new_sums)
                    elements.append(a)
                    found = backtrack(elements, new_sums_full, pos + 1, max(current_max, a))
                    elements.pop()
                    if found:
                        return True
        
        return False
    
    # Start the search
    # Initial: no elements, sums = {0}, current_max = 0
    backtrack([], {0}, 0, 0)
    
    return best_solution
