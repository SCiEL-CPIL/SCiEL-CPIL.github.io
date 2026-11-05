from fractions import Fraction
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    # Precompute reciprocals as Fractions
    recip = {x: Fraction(1, x) for x in range(1, 25)}
    
    # Key insight: We need to find the largest subset S of [1,24] such that
    # for no x in S does 1/x = sum of 1/y for any non-empty subset of S\{x}
    
    # Strategy: Use backtracking with pruning
    # Order elements by "difficulty" - smaller elements are harder to include
    # because their reciprocals are larger and easier to match
    
    # Precompute: for each element x, which subsets of other elements could potentially sum to 1/x
    # This is expensive, so we'll use a smarter approach
    
    def is_subsum(target, available_set):
        """Check if target is a sum of a non-empty subset of available_set."""
        if not available_set:
            return False
        
        # If target is negative, impossible
        if target < 0:
            return False
            
        lst = list(available_set)
        n = len(lst)
        
        if n == 0:
            return False
        
        # Quick check: if any single element equals target
        for x in lst:
            if recip[x] == target:
                return True
        
        if n <= 12:
            for mask in range(1, 1 << n):
                s = Fraction(0, 1)
                for i in range(n):
                    if mask & (1 << i):
                        s += recip[lst[i]]
                        if s > target:
                            break
                if s == target:
                    return True
            return False
        else:
            mid = n // 2
            left = lst[:mid]
            right = lst[mid:]
            
            left_sums = [Fraction(0, 1)]
            for elem in left:
                new_sums = []
                for s in left_sums:
                    ns = s + recip[elem]
                    if ns <= target:
                        new_sums.append(ns)
                left_sums.extend(new_sums)
                
            right_sums = [Fraction(0, 1)]
            for elem in right:
                new_sums = []
                for s in right_sums:
                    ns = s + recip[elem]
                    if ns <= target:
                        new_sums.append(ns)
                right_sums.extend(new_sums)
                
            right_set = set(right_sums)
            
            for ls in left_sums:
                needed = target - ls
                if needed in right_set:
                    if not (ls == Fraction(0, 1) and needed == Fraction(0, 1)):
                        return True
            return False

    def can_add(x, current_set):
        """Check if x can be added to current_set while maintaining validity."""
        # 1. Check if 1/x is a subsum of current_set
        if is_subsum(recip[x], current_set):
            return False
        
        # 2. Check if adding x invalidates any existing y in current_set
        # y is invalidated if 1/y is a subsum of (current_set U {x}) \ {y}
        # This is equivalent to: 1/y - 1/x is a subsum of current_set \ {y, x}
        # (assuming the subsum for 1/y uses x; if it doesn't, it was already invalid in current_set)
        for y in current_set:
            diff = recip[y] - recip[x]
            if diff < 0:
                continue
            remaining = current_set - {y, x}
            if is_subsum(diff, remaining):
                return False
        
        return True
    
    def is_valid_set(s):
        """Check if the entire set s is reciprocal-subsum-free."""
        for x in s:
            others = s - {x}
            if is_subsum(recip[x], others):
                return False
        return True
    
    # Greedy approach with multiple orderings
    best_set = set()
    
    # Try greedy from different starting points
    orderings = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        [24, 1, 23, 2, 22, 3, 21, 4, 20, 5, 19, 6, 18, 7, 17, 8, 16, 9, 15, 10, 14, 11, 13, 12],  # Alternating
    ]
    
    for ordering in orderings:
        candidate = set()
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if can_add(x, candidate):
                candidate.add(x)
        
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Try to improve by adding excluded elements
    excluded = set(range(1, 25)) - best_set
    for x in sorted(excluded, reverse=True):
        if time.time() > deadline - 5:
            break
        if can_add(x, best_set):
            best_set.add(x)
    
    # Try local search: swap elements to potentially increase size
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_list = sorted(best_set)
        excluded = sorted(set(range(1, 25)) - best_set, reverse=True)
        
        for i, x in enumerate(current_list):
            if time.time() > deadline - 10:
                break
            for y in excluded:
                if y in best_set:
                    continue
                # Try removing x and adding y
                new_set = best_set - {x} | {y}
                if len(new_set) > len(best_set):
                    if is_valid_set(new_set):
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break
    
    # Final attempt: try to add any remaining excluded elements
    excluded = sorted(set(range(1, 25)) - best_set, reverse=True)
    for x in excluded:
        if time.time() > deadline - 3:
            break
        if can_add(x, best_set):
            best_set.add(x)
    
    # Ensure non-empty
    if not best_set:
        best_set = {24}
    
    return sorted(best_set)
