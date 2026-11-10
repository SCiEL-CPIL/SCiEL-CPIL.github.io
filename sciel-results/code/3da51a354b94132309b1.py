"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = 256
    # Precompute: for each pair (a, b) with a < b, the value c = 2*b - a
    # If c is in the set and c > b, then (a, b, c) is a 3-AP
    # We can check: a, b, c form AP iff a + c = 2*b
    
    # Greedy: start with all numbers, then remove those causing APs
    # Better: start empty, add numbers in order of decreasing 1/x (i.e., ascending x)
    # but check if adding creates any 3-AP
    
    def has_3ap_with_candidate(selected_set, candidate):
        """Check if adding candidate to selected_set creates any 3-term AP."""
        s = selected_set
        for a in s:
            if a == candidate:
                continue
            # Check if (a, candidate, c) or (candidate, a, c) or (a, b, candidate) forms AP
            # Case 1: candidate is the middle: a + c = 2*candidate => c = 2*candidate - a
            c = 2 * candidate - a
            if c in s and c != candidate:
                return True
            # Case 2: candidate is the end: a + candidate = 2*b => b = (a + candidate)/2
            if (a + candidate) % 2 == 0:
                b = (a + candidate) // 2
                if b in s and b != candidate and b != a:
                    return True
            # Case 3: a is the middle: candidate + c = 2*a => c = 2*a - candidate
            c = 2 * a - candidate
            if c in s and c != candidate and c != a:
                return True
        return False
    
    # Greedy construction: add numbers from 1 to 256, skipping those that create APs
    selected = set()
    for x in range(1, n + 1):
        if not has_3ap_with_candidate(selected, x):
            selected.add(x)
    
    def reciprocal_sum(s):
        return sum(1.0 / x for x in s)
    
    best_set = set(selected)
    best_score = reciprocal_sum(best_set)
    
    # Local search: try swapping elements to improve the score
    # A swap: remove r from set, add a to set, if valid
    # Priority: try to add small numbers (high reciprocal) and remove large ones
    
    def is_valid_3ap_free(s):
        """Verify set has no 3-term AP."""
        arr = sorted(s)
        s_set = set(arr)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                a, b = arr[i], arr[j]
                c = 2 * b - a
                if c in s_set and c > b:
                    return False
        return True
    
    # Verify initial solution
    assert is_valid_3ap_free(best_set), "Initial greedy solution has a 3-AP!"
    
    # Local search with swaps
    # Try to improve by: removing a large element and adding a small one
    for iteration in range(10000):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        # Try all possible swaps: remove r, add a
        # Prioritize: a is small (not in set), r is large (in set)
        candidates_to_add = [x for x in range(1, n + 1) if x not in best_set]
        candidates_to_remove = sorted(best_set, reverse=True)  # try removing large ones first
        
        for a in candidates_to_add:
            if time.time() - start_time > DEADLINE:
                break
            for r in candidates_to_remove:
                # Skip if a == r
                if a == r:
                    continue
                # Try the swap
                new_set = best_set - {r}
                new_set.add(a)
                if is_valid_3ap_free(new_set):
                    new_score = reciprocal_sum(new_set)
                    if new_score > best_score + 1e-12:
                        best_set = new_set
                        best_score = new_score
                        improved = True
                        # Update candidates
                        candidates_to_add = [x for x in range(1, n + 1) if x not in best_set]
                        candidates_to_remove = sorted(best_set, reverse=True)
                        break
            if improved:
                break
        
        if not improved:
            # No single swap improves; try double swaps or break
            # Try adding one element without removing (if possible)
            added = False
            for a in candidates_to_add:
                if time.time() - start_time > DEADLINE:
                    break
                new_set = set(best_set)
                new_set.add(a)
                if is_valid_3ap_free(new_set):
                    new_score = reciprocal_sum(new_set)
                    if new_score > best_score + 1e-12:
                        best_set = new_set
                        best_score = new_score
                        added = True
                        candidates_to_add = [x for x in range(1, n + 1) if x not in best_set]
                        break
            if not added:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
