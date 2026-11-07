import time
import random
from itertools import combinations

def _is_valid_set(S):
    """Check if S satisfies all constraints: distinct ints in 1..256,
    all nonempty subset sums are distinct, and no two distinct sums divide one another."""
    n = len(S)
    if n == 0:
        return False
    if len(set(S)) != n:
        return False
    if any(x < 1 or x > 256 for x in S):
        return False
    
    # Generate all nonempty subset sums
    sums = set()
    for r in range(1, n + 1):
        for combo in combinations(S, r):
            s = sum(combo)
            if s in sums:
                return False
            sums.add(s)
    
    # Check divisibility: no two distinct sums divide one another
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % sorted_sums[i] == 0:
                return False
    return True

def _try_greedy_with_pruning(elements, current_set, best_set, deadline):
    """Try to extend current_set using remaining elements, with pruning."""
    if time.time() > deadline:
        return
    if len(current_set) > len(best_set):
        best_set.extend(current_set)
    
    if not elements:
        return
    
    for i, elem in enumerate(elements):
        # Quick check: can this element be added?
        new_set = current_set + [elem]
        # Check distinct subset sums and divisibility incrementally
        if _is_valid_set(new_set):
            remaining = elements[i+1:]
            _try_greedy_with_pruning(remaining, new_set, best_set, deadline)
        # Also try skipping this element
        if i < len(elements) - 1:
            _try_greedy_with_pruning(elements[i+1:], current_set, best_set, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Start with a known valid small set
    best_set = [31, 32, 34]
    
    # Randomized search: try to build larger sets
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Start from a random subset of candidates
        # Focus on numbers that are "far apart" to reduce divisibility conflicts
        candidates = list(range(1, 257))
        
        # Shuffle and try greedy construction
        random.shuffle(candidates)
        
        current = []
        used_sums = set()
        
        for elem in candidates:
            if time.time() > deadline:
                break
            
            # Try adding elem
            new_current = current + [elem]
            
            # Check if all subset sums are distinct and no divisibility
            valid = True
            new_sums = set()
            for r in range(1, len(new_current) + 1):
                for combo in combinations(new_current, r):
                    s = sum(combo)
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            
            if valid:
                # Check divisibility among all sums
                sorted_sums = sorted(new_sums)
                for i in range(len(sorted_sums)):
                    for j in range(i + 1, len(sorted_sums)):
                        if sorted_sums[j] % sorted_sums[i] == 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                current = new_current
                used_sums = new_sums
        
        if len(current) > len(best_set):
            best_set = current[:]
        
        # Try to improve best_set by swapping elements
        if len(best_set) >= 3:
            # Try replacing one element with a larger one
            for pos in range(len(best_set)):
                if time.time() > deadline:
                    break
                old_elem = best_set[pos]
                for new_elem in range(256, old_elem + 1, -1):
                    if new_elem == old_elem:
                        continue
                    candidate = best_set[:pos] + [new_elem] + best_set[pos+1:]
                    if _is_valid_set(candidate):
                        best_set = candidate
                        break
    
    # Ensure best_set is valid
    if not _is_valid_set(best_set):
        best_set = [31, 32, 34]
    
    return list(best_set)
