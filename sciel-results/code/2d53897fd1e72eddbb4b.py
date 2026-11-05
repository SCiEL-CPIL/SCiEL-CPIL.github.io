"""Rewritten program: randomized greedy + local swap optimization for
minimum maximal no-isosceles set on the 16x16 grid."""

import random
import time
import math

N = 16
DEADLINE = 320.0  # safety margin before 330s kill


def _sq(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _is_iso(a, b, c):
    d1 = _sq(a, b)
    d2 = _sq(a, c)
    d3 = _sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    """Check if p can be added to chosen without forming isosceles."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_ab = _sq(a, b)
            d_pb = _sq(p, b)
            if d_pa == d_ab or d_pa == d_pb or d_ab == d_pb:
                return False
    return True


def _is_maximal(chosen, all_points):
    """Check if no point in all_points can be added."""
    for p in all_points:
        if p not in chosen and _can_add(p, chosen):
            return False
    return True


def _greedy(chosen, order, rng):
    """Continue greedy: try to add points from order not in chosen."""
    chosen_set = set(chosen)
    for p in order:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def _local_swap(chosen, rng, max_swaps=200):
    """Try to remove a point and re-add others to reduce size.
    Actually, for maximal set, we can't just remove. Instead, try
    swap: remove one point, then greedily try to fill gaps, see if
    we end up with fewer points.
    
    Better approach: try replacing one point with another that allows
    the set to remain valid and potentially enables removing others.
    
    Simplest effective move: try removing a point p, then check if
    the remaining set is still maximal. If not, we need to add points.
    So this doesn't directly reduce size.
    
    Alternative: Since we want minimum maximal, we need a different
    strategy. Let's try: for each point in chosen, try removing it,
    then greedily add back minimum points to restore maximality,
    and see if total count decreased.
    """
    best = list(chosen)
    for _ in range(max_swaps):
        # Pick a random point to try removing
        idx = rng.randrange(len(best))
        p = best[idx]
        candidate = [q for i, q in enumerate(best) if i != idx]
        
        # Check if candidate is still valid (it should be, since subset)
        # Now check if it's maximal
        if _is_maximal(candidate, ALL_POINTS):
            # We removed a point and it's still maximal! Great.
            best = candidate
        else:
            # Need to add points to restore maximality
            # Greedily add points that are "forced" (blocked by fewest)
            # Actually, just greedily add any valid point
            added = _greedy(candidate, ALL_POINTS, rng)
            if len(added) < len(best):
                best = added
    return best


# Precompute all points
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_size = float('inf')
    best_set = None
    
    # Precompute some ordering strategies
    orders = []
    
    # Row-major
    orders.append(ALL_POINTS[:])
    # Random orders
    for _ in range(5):
        o = ALL_POINTS[:]
        rng.shuffle(o)
        orders.append(o)
    # Diagonal order
    diag = sorted(ALL_POINTS, key=lambda p: (p[0] + p[1], p[0]))
    orders.append(diag)
    # Reverse row-major
    orders.append(list(reversed(ALL_POINTS)))
    
    iteration = 0
    while time.time() - start_time < DEADLINE:
        # Try different strategies
        order = orders[iteration % len(orders)]
        
        # Greedy construction
        chosen = []
        for p in order:
            if _can_add(p, chosen):
                chosen.append(p)
        
        # Local optimization
        if time.time() - start_time < DEADLINE - 5:
            chosen = _local_swap(chosen, rng, max_swaps=100)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
        
        iteration += 1
        # Periodically add new random orders
        if iteration % 10 == 0:
            o = ALL_POINTS[:]
            rng.shuffle(o)
            orders.append(o)
    
    if best_set is None:
        # Fallback: simple greedy
        best_set = []
        for p in ALL_POINTS:
            if _can_add(p, best_set):
                best_set.append(p)
    
    return best_set
