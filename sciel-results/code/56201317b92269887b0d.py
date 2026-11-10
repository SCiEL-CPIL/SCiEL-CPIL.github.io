import itertools
import random
import time

N = 16
DEADLINE = 280.0  # seconds, safety margin before 300s kill

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_blocked(p, chosen):
    """Check if p is blocked by chosen: for every q not in chosen, exists a,b in chosen with {p,a,b} isosceles."""
    # Actually, p is blocked if for every pair (a,b) in chosen, adding p doesn't create isosceles... 
    # No: p can be added if for ALL pairs (a,b) in chosen, {p,a,b} is NOT isosceles.
    # So p is NOT blocked (can be added) iff for all a,b in chosen, not isosceles(p,a,b).
    # p IS blocked iff there exists at least one pair (a,b) in chosen such that isosceles(p,a,b).
    # Wait, that's not right either.
    
    # p can be added to chosen iff for every pair (a,b) in chosen, {p,a,b} is not isosceles.
    # So p is blocked (cannot be added) iff there exists a pair (a,b) in chosen such that {p,a,b} IS isosceles.
    for a, b in itertools.combinations(chosen, 2):
        if _is_isosceles(p, a, b):
            return True
    return False

def _can_add(p, chosen):
    """Can p be added to chosen without creating isosceles?"""
    for a, b in itertools.combinations(chosen, 2):
        if _is_isosceles(p, a, b):
            return False
    return True

def _is_maximal(chosen, all_points):
    """Check if chosen is maximal: every point not in chosen is blocked."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _greedy(order, chosen=None):
    """Build a maximal set by greedily adding points in the given order."""
    if chosen is None:
        chosen = []
    chosen_set = set()
    for p in order:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    # Now make it maximal by adding any point that can still be added
    for p in order:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _try_remove_points(chosen, all_points):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            trial = chosen[:i] + chosen[i+1:]
            if _is_maximal(trial, all_points):
                chosen = trial
                changed = True
                break
    return chosen

def construct():
    start_time = time.time()
    all_points = list(itertools.product(range(N), repeat=2))
    all_points_set = set(all_points)
    
    best = None
    
    # First, try the seed's approach as baseline
    seed_order = list(itertools.product(range(N), repeat=2))
    baseline = _greedy(seed_order)
    baseline = _try_remove_points(baseline, all_points)
    best = baseline
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Different ordering strategies
        if trial % 4 == 0:
            order = all_points[:]
            random.shuffle(order)
        elif trial % 4 == 1:
            # Sort by distance from center
            cx, cy = (N-1)/2.0, (N-1)/2.0
            order = sorted(all_points, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2 + random.random()*0.1)
        elif trial % 4 == 2:
            # Random shuffle
            order = all_points[:]
            random.shuffle(order)
        else:
            # Sort by row, then random within row
            order = sorted(all_points, key=lambda p: (p[1], random.random()))
        
        result = _greedy(order)
        result = _try_remove_points(result, all_points)
        
        if best is None or len(result) < len(best):
            best = result
    
    # Try more aggressive removal on the best
    if time.time() - start_time < DEADLINE - 10:
        best = _try_remove_points(best, all_points)
    
    return best
