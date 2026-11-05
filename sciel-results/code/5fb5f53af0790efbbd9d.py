"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

DEADLINE = 325  # seconds, safety margin before 330s kill

def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

def _is_iso(a, b, c):
    d1 = _dist_sq(a, b)
    d2 = _dist_sq(a, c)
    d3 = _dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _creates_iso(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    for i in range(len(chosen)):
        a = chosen[i]
        da = _dist_sq(p, a)
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            db = _dist_sq(p, b)
            ab = _dist_sq(a, b)
            if da == db or da == ab or db == ab:
                return True
    return False

def _is_maximal(chosen, n=N):
    """Check if chosen is maximal: no point can be added without creating iso."""
    chosen_set = set(chosen)
    for x in range(n):
        for y in range(n):
            p = (x, y)
            if p in chosen_set:
                continue
            if not _creates_iso(p, chosen):
                return False
    return True

def _greedy_complete(chosen, order=None, n=N):
    """Greedily add points to make the set maximal."""
    if order is None:
        order = list(itertools.product(range(n), repeat=2))
    chosen = list(chosen)
    chosen_set = set(chosen)
    for p in order:
        if p in chosen_set:
            continue
        if not _creates_iso(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _try_remove(chosen, idx, n=N):
    """Try removing point at idx, then complete greedily. Return new set if smaller."""
    remaining = chosen[:idx] + chosen[idx+1:]
    completed = _greedy_complete(remaining)
    if len(completed) < len(chosen):
        return completed
    return None

def construct():
    start_time = time.time()
    n = N
    
    # Precompute all grid points
    all_points = list(itertools.product(range(n), repeat=2))
    
    # Phase 1: Get an initial maximal set using greedy with random order
    best_set = None
    best_len = n * n + 1
    
    # Try several random orderings for greedy construction
    for attempt in range(50):
        if time.time() - start_time > DEADLINE:
            break
        random_order = all_points[:]
        random.shuffle(random_order)
        candidate = _greedy_complete([], random_order, n)
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_set = candidate
            best_len = len(candidate)
    
    if best_set is None:
        # Fallback: simple row-major greedy
        best_set = _greedy_complete([], all_points, n)
        best_len = len(best_set)
    
    # Phase 2: Local search - try removing points and re-completing
    current_set = best_set[:]
    current_len = best_len
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < DEADLINE and no_improve_count < max_no_improve:
        improved = False
        # Shuffle indices for diversity
        indices = list(range(len(current_set)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > DEADLINE:
                break
            new_set = _try_remove(current_set, idx, n)
            if new_set is not None:
                current_set = new_set
                current_len = len(new_set)
                improved = True
                no_improve_count = 0
                break
        if not improved:
            no_improve_count += 1
    
    # Phase 3: Try different completion orders after removals
    if time.time() - start_time < DEADLINE - 30:
        for attempt in range(20):
            if time.time() - start_time > DEADLINE:
                break
            # Take current best, remove a random subset, re-complete with random order
            size_to_keep = max(0, len(current_set) - random.randint(1, 3))
            keep_indices = random.sample(range(len(current_set)), size_to_keep)
            subset = [current_set[i] for i in keep_indices]
            random_order = all_points[:]
            random.shuffle(random_order)
            candidate = _greedy_complete(subset, random_order, n)
            if len(candidate) < current_len:
                current_set = candidate
                current_len = len(candidate)
    
    # Verify maximality (should be true by construction, but just in case)
    if not _is_maximal(current_set, n):
        current_set = _greedy_complete(current_set, all_points, n)
    
    return current_set
# EVOLVE-BLOCK-END
