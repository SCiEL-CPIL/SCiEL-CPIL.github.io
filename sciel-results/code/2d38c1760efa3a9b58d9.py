# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints.
    Only need to check:
    1. x doesn't divide sum of any 3 other elements in selected
    2. No element in selected divides sum of any 2 other selected elements + x
    """
    n = len(selected)
    # Check 1: x doesn't divide sum of any 3 distinct others in selected
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = selected[i] + selected[j] + selected[k]
                if s % x == 0:
                    return False
    # Check 2: No element a in selected divides sum of any 2 other selected + x
    # For each a in selected, check all pairs (b,c) from selected\{a}
    # where b != c, and b,c != a, and check if (b+c+x) % a == 0
    # But we also need b and c to be distinct from a and from each other
    for a in selected:
        for i in range(n):
            for j in range(i+1, n):
                b = selected[i]
                c = selected[j]
                if b == a or c == a:
                    continue
                s = b + c + x
                if s % a == 0:
                    return False
    return True

def _is_valid(selected):
    """Full validity check."""
    n = len(selected)
    for idx, a in enumerate(selected):
        others = [selected[i] for i in range(n) if i != idx]
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
    return True

def _greedy_descending(deadline):
    """Greedy: try adding from 192 down to 1."""
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_ascending(deadline):
    """Greedy: try adding from 1 up to 192."""
    selected = []
    for x in range(1, 193):
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_randomized(deadline, seed=0):
    """Greedy: random order."""
    rng = random.Random(seed)
    candidates = list(range(1, 193))
    rng.shuffle(candidates)
    selected = []
    for x in candidates:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline, max_rounds=50):
    """Local search: try swaps and additions to improve size."""
    if not _is_valid(selected):
        # Fix: remove invalid elements
        selected = [x for x in selected if _can_add(x, [y for y in selected if y != x])]
    
    best = list(selected)
    best_size = len(best)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding elements not in best
        missing = [x for x in range(1, 193) if x not in best]
        random.shuffle(missing)
        for x in missing:
            if time.time() > deadline:
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                if len(best) > best_size:
                    best_size = len(best)
        
        # Try swaps: remove one, add one or two
        if len(best) < 48:
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                a = best[i]
                temp = best[:i] + best[i+1:]
                # Try to add up to 2 new elements
                missing = [x for x in range(1, 193) if x not in temp]
                random.shuffle(missing)
                added = 0
                for x in missing:
                    if added >= 2:
                        break
                    if _can_add(x, temp):
                        temp.append(x)
                        added += 1
                if len(temp) > len(best):
                    best = temp
                    best_size = len(best)
                    improved = True
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s margin
    
    # Try multiple greedy strategies
    candidates = []
    
    # Descending greedy
    s1 = _greedy_descending(deadline)
    candidates.append(s1)
    
    # Ascending greedy
    if time.time() < deadline:
        s2 = _greedy_ascending(deadline)
        candidates.append(s2)
    
    # Randomized greedy with different seeds
    for seed in range(5):
        if time.time() > deadline:
            break
        s = _greedy_randomized(deadline, seed=seed)
        candidates.append(s)
    
    # Pick the best initial solution
    best = max(candidates, key=len)
    
    # Local search to improve
    if time.time() < deadline:
        best = _local_search(best, deadline)
    
    # Final validation
    if not _is_valid(best):
        # Fallback: build from scratch with validation
        selected = []
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if _can_add(x, selected):
                selected.append(x)
        best = selected
    
    return best
# EVOLVE-BLOCK-END
