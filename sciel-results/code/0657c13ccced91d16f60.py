"""Improved program: fast greedy with deadline guard and local search polish."""
import time
import random
from itertools import combinations

def _valid(selected):
    """Check if selected list satisfies the no-dividing condition."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for i, a in enumerate(selected):
        # For each a, check all triples of distinct OTHER elements
        others = [selected[j] for j in range(n) if j != i]
        # We need sum of any 3 distinct others not divisible by a
        # Efficient check: precompute all triple sums? For small n this is fine.
        # But let's do it smartly: for a given a, compute all triple sums of others
        # and check if any is 0 mod a.
        m = len(others)
        if m < 3:
            continue
        # Generate all triple sums
        found_divisible = False
        for x_idx in range(m - 2):
            x = others[x_idx]
            for y_idx in range(x_idx + 1, m - 1):
                y = others[y_idx]
                for z_idx in range(y_idx + 1, m):
                    z = others[z_idx]
                    if (x + y + z) % a == 0:
                        found_divisible = True
                        break
                if found_divisible:
                    break
            if found_divisible:
                break
        if found_divisible:
            return False
    return True

def _can_add(selected, candidate, deadline):
    """Check if adding candidate to selected keeps validity. Fast path: only check new constraints."""
    # New constraints:
    # 1. candidate does not divide sum of any 3 distinct others in selected
    # 2. For each a in selected, a does not divide sum of any 3 distinct from (selected \ {a}) ∪ {candidate}
    #    This means: for each a, check triples that include candidate and 2 from selected\{a}, 
    #    and triples that are 3 from selected\{a} (already valid, skip)
    
    sel = selected
    cand = candidate
    n = len(sel)
    
    # Check 1: cand does not divide sum of any 3 distinct from sel
    if n >= 3:
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if (sel[i] + sel[j] + sel[k]) % cand == 0:
                        return False
    
    # Check 2: for each a in sel, a does not divide sum of any triple including cand
    # Triples including cand: {cand, x, y} where x, y are distinct from sel, x != a, y != a
    # Also need x != y, and x, y != cand (cand is new, so x,y from sel)
    for a in sel:
        # Triples with cand and two others from sel, neither being a
        # So we need x, y in sel, x != a, y != a, x != y
        # And check (cand + x + y) % a != 0
        # Also, triples without cand but with 3 from sel\{a} are already valid.
        # But wait: the condition is about ALL triples of 3 distinct OTHER elements.
        # When we add cand, the "others" for a now include cand.
        # So we need to check triples from (sel \ {a}) ∪ {cand}.
        # Triples that don't include cand: already checked (valid).
        # Triples that include cand: {cand, x, y} where x, y ∈ sel \ {a}, x != y.
        # So for each a, check all pairs (x, y) from sel \ {a} and see if (cand + x + y) % a == 0.
        
        # Also, what about triples that are {cand, x, y} where one of x or y is a? No, a is excluded.
        # So x, y must be from sel, x != a, y != a, x != y.
        
        # Let's collect sel without a
        others = [v for v in sel if v != a]
        m = len(others)
        if m < 2:
            continue
        for i in range(m - 1):
            x = others[i]
            for j in range(i + 1, m):
                y = others[j]
                if (cand + x + y) % a == 0:
                    return False
    
    return True

def _greedy_descending(deadline):
    """Greedy: try to add numbers from 192 down to 1."""
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if _can_add(selected, x, deadline):
            selected.append(x)
    return selected

def _greedy_ascending(deadline):
    """Greedy: try to add numbers from 1 up to 192."""
    selected = []
    for x in range(1, 193):
        if time.time() > deadline:
            break
        if _can_add(selected, x, deadline):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve by removing one element and adding a better one."""
    best = list(selected)
    best_len = len(best)
    
    while time.time() < deadline - 5:
        improved = False
        # Try removing each element and see if we can add a new one
        for i in range(len(best)):
            if time.time() > deadline - 5:
                break
            trial = [best[j] for j in range(len(best)) if j != i]
            removed = best[i]
            # Try adding all numbers not in trial
            for x in range(1, 193):
                if time.time() > deadline - 5:
                    break
                if x in trial:
                    continue
                if _can_add(trial, x, deadline):
                    new_set = trial + [x]
                    if len(new_set) > best_len:
                        best = new_set
                        best_len = len(best)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline for 90s wall-clock limit
    
    # Start with descending greedy
    result = _greedy_descending(deadline)
    best = list(result)
    
    # Try local search
    if time.time() < deadline - 10:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Try ascending greedy as alternative
    if time.time() < deadline - 10:
        alt = _greedy_ascending(deadline)
        if len(alt) > len(best):
            best = alt
            if time.time() < deadline - 5:
                improved = _local_search(best, deadline)
                if len(improved) > len(best):
                    best = improved
    
    return best
