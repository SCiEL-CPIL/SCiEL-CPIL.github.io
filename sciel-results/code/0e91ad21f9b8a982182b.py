"""Structural rewrite: use a bitset-based representation with numpy for fast
conflict checking, plus a local-search strategy that swaps points to reduce size."""

import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_POINTS = N * N  # 256

_DEADLINE = time.time() + 280


def _idx_to_coord(idx):
    return (idx % N, idx // N)


def _coord_to_idx(x, y):
    return y * N + x


def _precompute():
    """
    Precompute for each point p, the set of points q such that there exists
    a point r where (p, q, r) is isosceles.
    We store this as a numpy boolean array: conflict[p] is a boolean vector of length NUM_POINTS.
    conflict[p][q] = True means p and q CANNOT both be in the set (because some r completes an isosceles).
    
    Actually, the constraint is: no three points form an isosceles triangle.
    So for a candidate set S, we need: for all distinct a,b,c in S, (a,b,c) is NOT isosceles.
    
    A pair (a,b) is "forbidden" given a third point c if (a,b,c) is isosceles.
    So we can define: for each pair (a,b), the set of c such that (a,b,c) is isosceles.
    Alternatively: for each point c, the set of pairs (a,b) that conflict with c.
    
    For fast checking when adding point p to a set S:
    p can be added iff for all pairs (a,b) in S, (p,a,b) is not isosceles.
    
    We'll precompute: for each p, a bitset bad_pairs[p] which is a boolean matrix of shape (NUM_POINTS, NUM_POINTS)
    where bad_pairs[p][a][b] = True if (p,a,b) is isosceles.
    But that's 256 * 256 * 256 = 16.7M booleans, which is ~16MB. That's fine.
    
    Actually, let's think differently. For adding p to S:
    We need: no (a,b) in S such that (p,a,b) is isosceles.
    
    Precompute for each p: a boolean vector of length NUM_POINTS*NUM_POINTS? No, that's too big.
    
    Better: for each p, precompute a set of (a,b) pairs where (p,a,b) is isosceles.
    Store as a list of frozensets or a 2D boolean array per p.
    
    Let's use: conflicts[p] = a 2D boolean numpy array of shape (NUM_POINTS, NUM_POINTS)
    where conflicts[p][a][b] = True if triangle (p,a,b) is isosceles.
    Total memory: 256 * 256 * 256 bytes = 16 MB. Fine.
    """
    coords = [(_idx_to_coord(i)) for i in range(NUM_POINTS)]
    
    def d2(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    
    # conflicts[p] is a 2D boolean array
    conflicts = np.zeros((NUM_POINTS, NUM_POINTS, NUM_POINTS), dtype=bool)
    
    for p in range(NUM_POINTS):
        pc = coords[p]
        for a in range(NUM_POINTS):
            if a == p:
                continue
            ac = coords[a]
            for b in range(NUM_POINTS):
                if b == p or b == a:
                    continue
                bc = coords[b]
                d_pa = d2(pc, ac)
                d_pb = d2(pc, bc)
                d_ab = d2(ac, bc)
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    conflicts[p, a, b] = True
                    conflicts[p, b, a] = True
    
    return coords, conflicts


def _can_add(p, chosen_set, conflicts):
    """Check if point p can be added to chosen_set (a set of indices)."""
    # For all pairs (a,b) in chosen_set, check that (p,a,b) is not isosceles
    # Using the precomputed conflicts[p] which is a 2D array
    for a in chosen_set:
        for b in chosen_set:
            if a == b:
                continue
            if conflicts[p, a, b]:
                return False
    return True


def _greedy_max_coverage(seed, coords, conflicts):
    """Greedy: pick points in an order, adding if possible."""
    rng = random.Random(seed)
    order = list(range(NUM_POINTS))
    rng.shuffle(order)
    
    chosen = set()
    for p in order:
        if _can_add(p, chosen, conflicts):
            chosen.add(p)
    
    return chosen


def _is_maximal(chosen, coords, conflicts):
    """Check if no point can be added."""
    for p in range(NUM_POINTS):
        if p in chosen:
            continue
        if _can_add(p, chosen, conflicts):
            return False
    return True


def _local_search_reduce(chosen, coords, conflicts, max_iters=500):
    """
    Try to reduce the size of a maximal no-isosceles set.
    Strategy: try removing a point, then if not maximal, try adding a different point.
    Or: try swapping one point for another to reduce total size.
    """
    chosen = set(chosen)
    best = set(chosen)
    best_size = len(chosen)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        improved = False
        # Try removing each point
        for p in list(best):
            candidate = set(best)
            candidate.discard(p)
            
            # Check if still no-isosceles (removing points preserves this)
            # Check maximality
            if _is_maximal(candidate, coords, conflicts):
                best = candidate
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for p in list(best):
            without_p = set(best)
            without_p.discard(p)
            
            # Find a point q not in without_p that can be added
            for q in range(NUM_POINTS):
                if q in without_p:
                    continue
                if _can_add(q, without_p, conflicts):
                    candidate = without_p | {q}
                    if _is_maximal(candidate, coords, conflicts):
                        if len(candidate) < best_size:
                            best = candidate
                            best_size = len(best)
                            improved = True
                            break
            if improved:
                break
        
        if not improved:
            break
    
    return best


def _hill_climbing_swap(chosen, coords, conflicts, max_iters=1000):
    """
    More aggressive local search: try all single-point swaps.
    """
    chosen = set(chosen)
    best = set(chosen)
    best_size = len(chosen)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        improved = False
        for p in list(best):
            without_p = set(best)
            without_p.discard(p)
            
            # Try adding each q
            for q in range(NUM_POINTS):
                if q in without_p:
                    continue
                if _can_add(q, without_p, conflicts):
                    candidate = without_p | {q}
                    if _is_maximal(candidate, coords, conflicts):
                        if len(candidate) < best_size:
                            best = candidate
                            best_size = len(best)
                            improved = True
                            break
            if improved:
                break
        
        if not improved:
            break
    
    return best


def construct():
    coords, conflicts = _precompute()
    
    best = None
    best_size = float('inf')
    
    # Run multiple greedy starts with different seeds
    for seed in range(200):
        if time.time() > _DEADLINE - 20:
            break
        
        S = _greedy_max_coverage(seed, coords, conflicts)
        size = len(S)
        
        if size < best_size:
            best = S
            best_size = size
        
        # Local search to reduce
        if time.time() < _DEADLINE - 15:
            reduced = _hill_climbing_swap(S, coords, conflicts, max_iters=200)
            if len(reduced) < best_size:
                best = reduced
                best_size = len(best)
    
    if best is None:
        best = _greedy_max_coverage(0, coords, conflicts)
    
    # Final local search
    if time.time() < _DEADLINE - 5:
        final = _hill_climbing_swap(best, coords, conflicts, max_iters=500)
        if len(final) < len(best):
            best = final
    
    return [_idx_to_coord(i) for i in sorted(best)]
