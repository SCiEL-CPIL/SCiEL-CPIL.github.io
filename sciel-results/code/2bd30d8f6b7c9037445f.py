"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _all_subset_sums_valid(sums):
    """Check that no two distinct sums divide each other, and all sums are unique."""
    if not sums:
        return True
    # Check uniqueness - if any two subsets give same sum, invalid
    if len(sums) != len(set(sums)):
        return False
    # Check divisibility: for any pair, neither divides the other
    s = sorted(sums)
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            if s[j] % s[i] == 0:
                return False
    return True

def _get_all_nonempty_subset_sums(elements):
    """Generate all non-empty subset sums."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid_set(elements):
    """Check if the set satisfies all constraints."""
    if len(elements) != len(set(elements)):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    sums = _get_all_nonempty_subset_sums(elements)
    return _all_subset_sums_valid(list(sums))

def _try_add_element(current_set, candidate, deadline):
    """Try adding candidate to current_set and check validity."""
    if time.time() > deadline:
        return None
    new_set = current_set + [candidate]
    if not _is_valid_set(new_set):
        return None
    return new_set

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_set = [31, 32, 34]
    
    # Strategy: Try to build larger sets using greedy + randomized restarts
    # Key insight: use numbers that are "incomparable" under divisibility of subset sums
    
    # First, try a deterministic greedy approach with multiple starting points
    candidates = list(range(256, 0, -1))  # Start with larger numbers
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates
        random.shuffle(candidates)
        
        current = []
        for c in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
            new_set = _try_add_element(current, c, deadline)
            if new_set is not None:
                current = new_set
        
        if len(current) > len(best_set):
            best_set = current
            if len(best_set) >= 30:
                break
    
    # Try to improve best_set by local search
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            # Try swapping elements
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for c in range(1, 257):
                    if c in best_set:
                        continue
                    new_set = best_set[:i] + [c] + best_set[i+1:]
                    if _is_valid_set(new_set) and len(new_set) > len(best_set):
                        best_set = new_set
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
