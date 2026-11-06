"""
Minimum maximal no-isosceles set in an N x N grid.

Strategy:
1. Randomized greedy construction with varied point orderings.
2. Aggressive minimality reduction (remove points while maintaining maximality).
3. Coverage-based greedy to construct sets that block many points.
4. Local search: try swapping one point for another to reduce set size.
5. MILP-based refinement: formulate as a set cover problem to find minimal
   subset that maintains maximality.
"""
import time
import random
import itertools
import numpy as np

N = 16
DEADLINE = 320.0

# Precompute all points
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
N_POINTS = len(ALL_POINTS)

# Precompute distance squared matrix
def _precompute_dist2():
    d2 = np.zeros((N_POINTS, N_POINTS), dtype=np.int32)
    for i in range(N_POINTS):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, N_POINTS):
            x2, y2 = ALL_POINTS[j]
            d = (x1 - x2)**2 + (y1 - y2)**2
            d2[i, j] = d
            d2[j, i] = d
    return d2

DIST2 = _precompute_dist2()

def _dist2_idx(i, j):
    return DIST2[i, j]

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _point_to_idx(p):
    return p[1] * N + p[0]

def _isosceles_idx(i, j, k):
    d1 = DIST2[i, j]
    d2 = DIST2[i, k]
    d3 = DIST2[j, k]
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add_idx(p_idx, chosen_indices):
    """Check if point p_idx can be added to chosen without creating isosceles triangle."""
    for i in chosen_indices:
        for j in chosen_indices:
            if i == j:
                continue
            if _isosceles_idx(p_idx, i, j):
                return False
    return True

def _is_maximal_idx(chosen_set):
    """Check if chosen_set is maximal (no point can be added)."""
    chosen_set = set(chosen_set)
    for p in range(N_POINTS):
        if p in chosen_set:
            continue
        if _can_add_idx(p, list(chosen_set)):
            return False
    return True

def _reduce_to_minimal(chosen_set, all_indices):
    """
    Iteratively remove points from chosen_set if the remaining set is still maximal.
    """
    chosen_list = list(chosen_set)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen_list)):
            p = chosen_list[i]
            S_test = chosen_list[:i] + chosen_list[i+1:]
            
            if len(S_test) < 2:
                continue
            
            S_test_set = set(S_test)
            # Check maximality
            maximal = True
            for q in all_indices:
                if q in S_test_set:
                    continue
                if _can_add_idx(q, S_test):
                    maximal = False
                    break
            
            if maximal:
                chosen_list = S_test
                changed = True
                break
                
    return chosen_list

def _greedy_random(order, rng):
    """
    Greedy construction: iterate through points in given order, add if valid.
    """
    chosen = []
    chosen_set = set()
    for p in order:
        if p in chosen_set:
            continue
        if _can_add_idx(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _coverage_score(p_idx, chosen, all_indices, chosen_set):
    """
    Score how many points p_idx would block if added to chosen.
    """
    if not _can_add_idx(p_idx, chosen):
        return 0
    
    new_blocked = 0
    for q in all_indices:
        if q in chosen_set or q == p_idx:
            continue
        # Check if q is already blocked by chosen
        already_blocked = False
        for a in chosen:
            if _isosceles_idx(q, a, chosen[0]) if len(chosen) > 0 else False:
                already_blocked = True
                break
        # More careful check
        already_blocked = False
        for i in range(len(chosen)):
            a = chosen[i]
            for j in range(i + 1, len(chosen)):
                b = chosen[j]
                if _isosceles_idx(q, a, b):
                    already_blocked = True
                    break
            if already_blocked:
                break
        
        if already_blocked:
            continue
        
        # Check if adding p_idx would block q
        for a in chosen:
            if _isosceles_idx(p_idx, q, a):
                new_blocked += 1
                break
                
    return new_blocked

def _greedy_coverage(all_indices, rng):
    """
    Greedy construction using coverage heuristic.
    """
    remaining = list(all_indices)
    chosen = []
    chosen_set = set()
    
    if not remaining:
        return chosen
    
    start = rng.choice(remaining)
    remaining.remove(start)
    chosen.append(start)
    chosen_set.add(start)
    
    while remaining:
        best_p = None
        best_score = -1
        
        candidates = remaining
        if len(candidates) > 150:
            candidates = rng.sample(candidates, 150)
        
        for p in candidates:
            score = _coverage_score(p, chosen, all_indices, chosen_set)
            if score > best_score:
                best_score = score
                best_p = p
        
        if best_p is None or best_score <= 0:
            break
            
        remaining.remove(best_p)
        chosen.append(best_p)
        chosen_set.add(best_p)
        
    return chosen

def _local_search(chosen, all_indices, time_limit=30.0):
    """
    Try to improve chosen by swapping one point for another.
    """
    start_time = time.time()
    chosen_list = list(chosen)
    n = len(chosen_list)
    
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        
        p_remove = chosen_list[i]
        S_without = chosen_list[:i] + chosen_list[i+1:]
        S_without_set = set(S_without)
        
        for p_add in all_indices:
            if p_add in S_without_set:
                continue
            if p_add == p_remove:
                continue
            
            if not _can_add_idx(p_add, S_without):
                continue
            
            S_new = S_without + [p_add]
            
            # Check if maximal
            maximal = True
            for q in all_indices:
                if q in S_new:
                    continue
                if _can_add_idx(q, S_new):
                    maximal = False
                    break
            
            if maximal:
                reduced = _reduce_to_minimal(S_new, all_indices)
                if len(reduced) < n:
                    return reduced
                    
    return None

def _milp_minimize(chosen, all_indices, time_limit=30.0):
    """
    Use MILP to find minimal subset of chosen that is still maximal.
    For each point q not in chosen, we need at least one triple (q, a, b) with a,b in chosen
    that is isosceles. This is a set cover problem.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    start_time = time.time()
    chosen_set = set(chosen)
    n_chosen = len(chosen)
    
    # For each point q not in chosen, find all triples (q, a, b) with a,b in chosen that are isosceles
    # We need to select a subset of chosen such that for every q, at least one such triple exists.
    
    # Build constraint matrix
    # Variables: x_i for i in chosen (1 if kept, 0 if removed)
    # For each q not in chosen: sum of x_a * x_b over isosceles triples >= 1
    # This is quadratic, so we linearize with auxiliary variables.
    
    # Simpler approach: just try all subsets of chosen of size k, starting from small k.
    # This is exponential, so use a heuristic instead.
    
    # Heuristic: iteratively remove points and check maximality
    best = chosen[:]
    best_set = set(best)
    
    for round in range(10):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(len(best)):
            p = best[i]
            candidate = best[:i] + best[i+1:]
            if len(candidate) < 2:
                continue
            cand_set = set(candidate)
            maximal = True
            for q in all_indices:
                if q in cand_set:
                    continue
                if _can_add_idx(q, candidate):
                    maximal = False
                    break
            if maximal:
                best = candidate
                best_set = set(best)
                improved = True
                break
        if not improved:
            break
            
    return best

def construct():
    start_time = time.time()
    
    all_indices = list(range(N_POINTS))
    
    best_set = None
    best_size = N_POINTS + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Random greedy + reduction
    for attempt in range(300):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        order = all_indices[:]
        rng.shuffle(order)
        
        chosen = _greedy_random(order, rng)
        
        # Ensure maximality by adding more points if needed
        chosen_set = set(chosen)
        changed = True
        while changed:
            changed = False
            for p in all_indices:
                if p in chosen_set:
                    continue
                if _can_add_idx(p, chosen):
                    chosen.append(p)
                    chosen_set.add(p)
                    changed = True
        
        reduced = _reduce_to_minimal(chosen, all_indices)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 7:
            break
    
    # Strategy 2: Coverage greedy + reduction
    for attempt in range(150):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        chosen = _greedy_coverage(all_indices, rng)
        
        if not chosen:
            continue
            
        chosen_set = set(chosen)
        # Ensure maximality
        changed = True
        while changed:
            changed = False
            for p in all_indices:
                if p in chosen_set:
                    continue
                if _can_add_idx(p, chosen):
                    chosen.append(p)
                    chosen_set.add(p)
                    changed = True
        
        reduced = _reduce_to_minimal(chosen, all_indices)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 7:
            break
    
    # Strategy 3: Local search on best found so far
    if best_set is not None:
        for ls_round in range(3):
            if time.time() - start_time > DEADLINE - 15:
                break
            improved = _local_search(best_set, all_indices, time_limit=20.0)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            if best_size <= 7:
                break
    
    # Strategy 4: MILP-based minimization
    if best_set is not None:
        if time.time() - start_time < DEADLINE - 10:
            minimized = _milp_minimize(best_set, all_indices, time_limit=15.0)
            if len(minimized) < best_size:
                best_size = len(minimized)
                best_set = minimized[:]
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = _greedy_random(all_indices[:], rng)
        return [ALL_POINTS[i] for i in chosen]
    
    return [ALL_POINTS[i] for i in best_set]
