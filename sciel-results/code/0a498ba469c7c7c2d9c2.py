import time
import math
import itertools
import random

def score_set(U):
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 1.0
    return 1 + math.log(num) / den

def fast_score_set(U):
    """
    Optimized scoring for small sets.
    """
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    n = len(U)
    for i in range(n):
        a = U[i]
        for j in range(n):
            b = U[j]
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 1.0
    return 1 + math.log(num) / den

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_set = [0, 1, 3, 7, 13]
    best_score = score_set(best_set)
    
    # Phase 1: Exhaustive/Backtracking Search for small sets (size 3 to 5)
    # This ensures we find the global optimum for small dimensions.
    # We restrict max element to a reasonable bound to keep it fast.
    max_val_bound = 30
    
    # Pre-compute all subsets of {0..max_val_bound} of size 3, 4, 5?
    # C(31, 5) = 169911. This is small enough to iterate over directly if scoring is fast.
    # Let's just iterate over combinations.
    
    candidates = []
    for size in range(3, 6):
        # We must include 0. So we choose size-1 elements from 1..max_val_bound
        for combo in itertools.combinations(range(1, max_val_bound + 1), size - 1):
            s = [0] + list(combo)
            sc = fast_score_set(s)
            if sc > best_score:
                best_score = sc
                best_set = s[:]
            if time.time() > deadline:
                return best_set
                
    # Phase 2: Local Search starting from the best found
    current = best_set[:]
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swapping each element with a new value
        # Expand search range slightly for larger sets
        max_search_val = max(current) + 10
        if max_search_val > 100:
            max_search_val = 100
            
        for i in range(len(current)):
            old_val = current[i]
            # Try values in range 0..max_search_val
            for new_val in range(0, max_search_val + 1):
                if new_val == old_val:
                    continue
                # Ensure 0 is in set
                if old_val == 0 and new_val != 0:
                    continue
                if 0 not in current and new_val != 0:
                    continue
                
                # Create new set
                new_set = current[:]
                new_set[i] = new_val
                # Remove duplicates and sort
                new_set = sorted(set(new_set))
                
                if 0 not in new_set:
                    continue
                
                # Check size
                if len(new_set) < 3 or len(new_set) > 8:
                    continue
                
                s = fast_score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
        
        # Try adding an element
        if len(current) < 8:
            for new_val in range(0, max_search_val + 1):
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                if 0 not in new_set:
                    continue
                s = fast_score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
                        
        # Try removing an element (if size > 3)
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                new_set = current[:i] + current[i+1:]
                s = fast_score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]

    return best_set
