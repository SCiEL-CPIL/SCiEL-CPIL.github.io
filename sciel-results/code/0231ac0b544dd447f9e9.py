import random
import time
import numpy as np
from scipy.optimize import linprog

N = 256

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _compute_sums(selected):
    """Compute all pairwise sums."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _greedy_maximize(selected, sums, deadline):
    """Greedily add elements until maximal, choosing max-blocking element."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        # Find unblocked elements
        sel_set = set(selected)
        unblocked = []
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if 2 * x in sums:
                continue
            blocked = False
            for a in selected:
                if x + a in sums:
                    blocked = True
                    break
            if not blocked:
                unblocked.append(x)
        
        if not unblocked:
            break
        
        # Choose element that blocks the most other unblocked elements
        best_x = None
        best_score = -1
        for x in unblocked:
            new_sums = set(sums)
            valid = True
            if 2 * x in new_sums:
                valid = False
            else:
                new_sums.add(2 * x)
                for a in selected:
                    s = x + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            if not valid:
                continue
            
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    score += 1
                    continue
                y_blocked = False
                for a in selected + [x]:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if y_blocked:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        new_sums = set(sums)
        if 2 * best_x not in new_sums:
            new_sums.add(2 * best_x)
            for a in selected:
                s = best_x + a
                if s not in new_sums:
                    new_sums.add(s)
            selected.append(best_x)
            sums = new_sums
        else:
            break
    
    return selected, sums

def _local_search_reduce(selected, deadline):
    """Try to remove elements while maintaining Sidon + maximality."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            if _is_valid_sidon(candidate):
                sums = _compute_sums(candidate)
                if _is_maximal(candidate, sums):
                    selected = candidate
                    improved = True
                    break
    return selected

def _lp_based_construct(rng, deadline):
    """Use LP relaxation of hitting set to guide construction."""
    # First, build a candidate pool using greedy
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # Build initial Sidon set
    target_size = rng.randint(10, 20)
    for x in candidates:
        if len(selected) >= target_size:
            break
        new_sums = set(sums)
        valid = True
        if 2 * x in new_sums:
            valid = False
        else:
            new_sums.add(2 * x)
            for a in selected:
                s = x + a
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
        if valid:
            selected.append(x)
            sums = new_sums
    
    if not selected:
        return None
    
    # Greedy maximize to get a maximal set
    selected, sums = _greedy_maximize(selected, sums, deadline)
    
    if not _is_valid_sidon(selected) or not _is_maximal(selected, sums):
        return None
    
    # Now try LP-based reduction
    # For each element in selected, check if removing it still maintains maximality
    # But first, let's try a different approach: use LP to find which elements are most critical
    
    # Build constraint matrix for hitting set
    # Variables: x_i for each i in [1, N], 1 if selected
    # For each missing element m, we need: sum of x_i over i that block m >= 1
    # An element i blocks m if 2i = 2m (i=m, not possible since m missing) 
    # or i + j = 2m for some j in selected, or i + m = s for some s in sums
    
    # This is complex. Let's use a simpler heuristic:
    # For each element in selected, compute how many missing elements it uniquely blocks
    sel_set = set(selected)
    unique_block_count = {}
    for i in selected:
        count = 0
        for m in range(1, N + 1):
            if m in sel_set:
                continue
            # Check if i is needed to block m
            # m is blocked if 2m in sums OR m+a in sums for some a in selected
            # Check if removing i would unblock m
            sums_without_i = set(sums)
            # Remove sums involving i
            for a in selected:
                if a == i:
                    continue
                s = i + a
                if s in sums_without_i and s != 2 * i:
                    # Check if this sum is still present from other pairs
                    still_present = False
                    for b_idx, b in enumerate(selected):
                        if b == i:
                            continue
                        for c_idx, c in enumerate(selected[b_idx+1:], start=b_idx+1):
                            if c == i:
                                continue
                            if b + c == s:
                                still_present = True
                                break
                        if still_present:
                            break
                    if not still_present:
                        sums_without_i.discard(s)
            # Also remove 2*i
            sums_without_i.discard(2 * i)
            
            # Check if m is still blocked
            if 2 * m in sums_without_i:
                continue
            blocked = False
            for a in selected:
                if a == i:
                    continue
                if m + a in sums_without_i:
                    blocked = True
                    break
            if not blocked:
                count += 1
        unique_block_count[i] = count
    
    # Try removing elements with lowest unique block count
    sorted_elements = sorted(selected, key=lambda x: unique_block_count.get(x, 0))
    for elem in sorted_elements:
        if time.time() > deadline:
            break
        candidate = [x for x in selected if x != elem]
        if not candidate:
            continue
        if _is_valid_sidon(candidate):
            sums = _compute_sums(candidate)
            if _is_maximal(candidate, sums):
                selected = candidate
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Strategy 1: Random initial Sidon set + greedy maximize + local search
    for attempt in range(150):
        if time.time() > deadline:
            break
        
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        target_size = rng.randint(10, 22)
        for x in candidates:
            if len(selected) >= target_size:
                break
            new_sums = set(sums)
            valid = True
            if 2 * x in new_sums:
                valid = False
            else:
                new_sums.add(2 * x)
                for a in selected:
                    s = x + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            if valid:
                selected.append(x)
                sums = new_sums
        
        if not selected:
            continue
        
        selected, sums = _greedy_maximize(selected, sums, deadline)
        
        if _is_valid_sidon(selected) and _is_maximal(selected, sums):
            if len(selected) < best_size:
                best_size = len(selected)
                best = list(selected)
            
            if time.time() < deadline:
                reduced = _local_search_reduce(selected, deadline)
                if _is_valid_sidon(reduced) and _is_maximal(reduced, _compute_sums(reduced)):
                    if len(reduced) < best_size:
                        best_size = len(reduced)
                        best = list(reduced)
        
        if best_size <= 14:
            break
    
    # Strategy 2: LP-guided construction
    if time.time() < deadline:
        for attempt in range(50):
            if time.time() > deadline:
                break
            result = _lp_based_construct(rng, deadline)
            if result is not None:
                if _is_valid_sidon(result) and _is_maximal(result, _compute_sums(result)):
                    if len(result) < best_size:
                        best_size = len(result)
                        best = list(result)
                        reduced = _local_search_reduce(result, deadline)
                        if _is_valid_sidon(reduced) and _is_maximal(reduced, _compute_sums(reduced)):
                            if len(reduced) < best_size:
                                best_size = len(reduced)
                                best = list(reduced)
            if best_size <= 14:
                break
    
    # Strategy 3: Structured initial sets
    if time.time() < deadline:
        for base in range(2, 12):
            if time.time() > deadline:
                break
            selected = []
            sums = set()
            for x in range(base, N + 1, base * 2 + 1):
                new_sums = set(sums)
                valid = True
                if 2 * x in new_sums:
                    valid = False
                else:
                    new_sums.add(2 * x)
                    for a in selected:
                        s = x + a
                        if s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                if valid:
                    selected.append(x)
                    sums = new_sums
            
            if selected:
                selected, sums = _greedy_maximize(selected, sums, deadline)
                if _is_valid_sidon(selected) and _is_maximal(selected, sums):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best = list(selected)
                    if time.time() < deadline:
                        reduced = _local_search_reduce(selected, deadline)
                        if _is_valid_sidon(reduced) and _is_maximal(reduced, _compute_sums(reduced)):
                            if len(reduced) < best_size:
                                best_size = len(reduced)
                                best = list(reduced)
    
    # Fallback
    if best is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            new_sums = set(sums)
            valid = True
            if 2 * x in new_sums:
                valid = False
            else:
                new_sums.add(2 * x)
                for a in selected:
                    s = x + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            if valid:
                selected.append(x)
                sums = new_sums
        return selected
    
    return best
