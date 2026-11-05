"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time
import itertools

# EVOLVE-BLOCK-START

def _sq_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _build_iso_pairs(all_points):
    """For each point i, precompute the set of unordered pairs {j,k} (j<k) such that (i,j,k) form an isosceles triangle."""
    n = len(all_points)
    iso_pairs = [set() for _ in range(n)]
    for i in range(n):
        pi = all_points[i]
        for j in range(i+1, n):
            pj = all_points[j]
            for k in range(j+1, n):
                if i == j or i == k or j == k:
                    continue
                pk = all_points[k]
                d1 = _sq_dist(pi, pj)
                d2 = _sq_dist(pi, pk)
                d3 = _sq_dist(pj, pk)
                if d1 == d2 or d1 == d3 or d2 == d3:
                    iso_pairs[i].add((j, k))
    return iso_pairs

def _can_add(p_idx, chosen_set, iso_pairs):
    """Check if point p_idx can be added to chosen_set."""
    for (a, b) in iso_pairs[p_idx]:
        if a in chosen_set and b in chosen_set:
            return False
    return True

def _is_maximal(chosen_set, n, iso_pairs):
    """Check if the set is maximal."""
    for i in range(n):
        if i not in chosen_set:
            if _can_add(i, chosen_set, iso_pairs):
                return False
    return True

def _greedy_build(order, n, iso_pairs):
    """Build a maximal set by greedily adding points in given order."""
    chosen = []
    chosen_set = set()
    for p_idx in order:
        if _can_add(p_idx, chosen_set, iso_pairs):
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    # Ensure maximality
    for p_idx in range(n):
        if p_idx not in chosen_set and _can_add(p_idx, chosen_set, iso_pairs):
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    return chosen

def _minimize_deletion(chosen, n, iso_pairs):
    """Repeatedly remove points if the set remains maximal after removal."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            trial_set = set(chosen)
            trial_set.discard(chosen[i])
            if _is_maximal(trial_set, n, iso_pairs):
                chosen.remove(chosen[i])
                chosen_set = trial_set
                changed = True
                break
    return chosen

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n = len(all_points)
    start = time.time()
    deadline = start + 285  # safety margin
    
    iso_pairs = _build_iso_pairs(all_points)
    
    random.seed(42)
    
    best = None
    best_size = n + 1
    
    # Strategy 1: Heuristic ordering - points with more constraints first
    constraint_counts = [len(iso_pairs[i]) for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda i: -constraint_counts[i])
    chosen = _greedy_build(order, n, iso_pairs)
    chosen = _minimize_deletion(chosen, n, iso_pairs)
    best = chosen
    best_size = len(chosen)
    
    # Strategy 2: Random restarts
    for attempt in range(300):
        if time.time() > deadline - 20:
            break
        
        order = list(range(n))
        random.shuffle(order)
        chosen = _greedy_build(order, n, iso_pairs)
        
        if len(chosen) < best_size:
            chosen = _minimize_deletion(chosen, n, iso_pairs)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen
                if best_size <= 8:
                    break
    
    # Strategy 3: Additional deletion passes
    if best is not None and time.time() < deadline - 15:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            trial = list(best)
            random.shuffle(trial)
            trial = _minimize_deletion(trial, n, iso_pairs)
            if len(trial) < best_size:
                best_size = len(trial)
                best = trial
    
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END

