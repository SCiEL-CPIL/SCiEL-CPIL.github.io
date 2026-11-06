"""Improved program: fast greedy + local search with deadline guard."""
import time
import random
from itertools import combinations

DEADLINE_SECONDS = 85.0  # safety margin before 90s kill

def _valid_addition(a, existing):
    """Check if adding 'a' to 'existing' keeps the set valid.
    We need:
      1. For every x in existing: sum of any 3 distinct elements from (existing + [a]) excluding x is not divisible by x.
         Since existing was valid, we only need to check sums that include 'a'.
         So for each x in existing: sum of any 2 distinct elements from existing (excluding x) + a is not divisible by x.
      2. For 'a' itself: sum of any 3 distinct elements from existing is not divisible by a.
    """
    # Check condition for 'a': no 3-sum from existing divisible by a
    n = len(existing)
    if n >= 3:
        for c in combinations(existing, 3):
            if sum(c) % a == 0:
                return False
    
    # Check condition for each existing element x: no 2-sum from existing (excluding x) + a divisible by x
    for i, x in enumerate(existing):
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i:
                    continue
                s = existing[j] + existing[k] + a
                if s % x == 0:
                    return False
    return True

def _is_valid_set(selected):
    """Check if the entire set is valid."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) >= 3:
            for c in combinations(others, 3):
                if sum(c) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Phase 1: Greedy construction (try multiple orders)
    best_set = []
    
    # Try descending order first (largest numbers are "safer" as they're less likely to divide sums)
    candidates = list(range(192, 0, -1))
    selected = []
    for x in candidates:
        if time.time() > deadline - 2.0:
            break
        if _valid_addition(x, selected):
            selected.append(x)
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Try a few random orders to see if we get larger sets
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        candidates = list(range(1, 193))
        rng.shuffle(candidates)
        selected = []
        for x in candidates:
            if _valid_addition(x, selected):
                selected.append(x)
        if len(selected) > len(best_set):
            best_set = list(selected)
    
    # Phase 2: Local search - try to add more elements or swap
    if time.time() > deadline - 3.0:
        return best_set
    
    # Try adding elements not in the set
    in_set = set(best_set)
    for x in range(192, 0, -1):
        if time.time() > deadline - 3.0:
            break
        if x in in_set:
            continue
        # Try adding x
        if _valid_addition(x, best_set):
            best_set.append(x)
            in_set.add(x)
    
    # Phase 3: Local search with swaps - try replacing one element with another to allow more additions
    if time.time() > deadline - 3.0:
        return best_set
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        current_size = len(best_set)
        
        # Try removing one element and adding a different one (or two)
        for i in range(len(best_set)):
            if time.time() > deadline - 3.0:
                break
            removed = best_set[i]
            temp_set = best_set[:i] + best_set[i+1:]
            
            # Try adding any element not in temp_set
            in_temp = set(temp_set)
            for x in range(192, 0, -1):
                if x in in_temp:
                    continue
                if _valid_addition(x, temp_set):
                    new_set = temp_set + [x]
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        improved = True
                        break
            # Also try just removing (if it helps subsequent additions, but that's complex)
            # For now, only accept if size increases
    
    return best_set
