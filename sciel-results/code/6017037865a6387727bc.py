import time
import math
import random

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Precompute coordinates
    xs = [i // N for i in range(N * N)]
    ys = [i % N for i in range(N * N)]
    
    all_points = list(range(N * N))
    
    def dist2(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx * dx + dy * dy
    
    selected = set()
    # dist_sets[q] = set of squared distances from q to other points in selected
    dist_sets = {}
    
    def can_add(p):
        if p in selected:
            return False
        seen_from_p = set()
        for q in selected:
            d = dist2(p, q)
            if d in seen_from_p:
                return False
            if d in dist_sets[q]:
                return False
            seen_from_p.add(d)
        return True

    def add(p):
        selected.add(p)
        dist_sets[p] = set()
        for q in selected:
            if q == p:
                continue
            d = dist2(p, q)
            dist_sets[p].add(d)
            dist_sets[q].add(d)

    def remove(p):
        selected.discard(p)
        if p not in dist_sets:
            return
        for q in selected:
            if q == p:
                continue
            d = dist2(p, q)
            if d in dist_sets[q]:
                dist_sets[q].discard(d)
        del dist_sets[p]
        
    def try_swap(p_out, p_in):
        """
        Attempts to swap p_out (in selected) with p_in (not in selected).
        Returns True if swap is valid and performed.
        """
        if p_in in selected or p_out not in selected:
            return False
            
        # Check validity of new set S' = S - {p_out} + {p_in}
        
        # 1. Check p_in's distances to all r in S' \ {p_in}
        seen_from_pin = set()
        for r in selected:
            if r == p_out:
                continue
            d = dist2(p_in, r)
            if d in seen_from_pin:
                return False
            seen_from_pin.add(d)
            
        # 2. Check for each r in S' \ {p_in}, if dist(p_in, r) conflicts
        for r in selected:
            if r == p_out:
                continue
            d_pr = dist2(p_in, r)
            d_pout_r = dist2(p_out, r)
            
            if d_pr in dist_sets[r]:
                # Conflict exists in current S.
                # It is resolved only if the conflicting point was p_out.
                if d_pout_r != d_pr:
                    return False
                # If d_pout_r == d_pr, the conflict was with p_out, which is removed.
                # So it's fine.
        
        # Swap is valid
        remove(p_out)
        add(p_in)
        return True

    def min_dist_to_selected(p):
        if not selected:
            return float('inf')
        return min(dist2(p, q) for q in selected)

    # Phase 1: Multi-ordering greedy initialization
    # Try several different orderings and keep the best result
    
    rng = random.Random(42)
    
    best_size = 0
    best_state = set()
    best_dist_sets = {}
    
    # Define multiple ordering strategies
    orderings = []
    
    # Strategy 1: Random shuffle
    shuffled = all_points.copy()
    rng.shuffle(shuffled)
    orderings.append(shuffled)
    
    # Strategy 2: Sort by (x+y, x-y) - diagonal ordering
    orderings.append(sorted(all_points, key=lambda p: (xs[p] + ys[p], xs[p] - ys[p])))
    
    # Strategy 3: Sort by distance from center (far first)
    cx, cy = N / 2.0, N / 2.0
    orderings.append(sorted(all_points, key=lambda p: -((xs[p] - cx)**2 + (ys[p] - cy)**2)))
    
    # Strategy 4: Hilbert-like curve (approximate with space-filling pattern)
    hilbert_order = []
    for y in range(N):
        for x in range(N):
            if y % 2 == 0:
                hilbert_order.append(y * N + x)
            else:
                hilbert_order.append(y * N + (N - 1 - x))
    orderings.append(hilbert_order)
    
    # Strategy 5: Random with different seed
    rng2 = random.Random(12345)
    shuffled2 = all_points.copy()
    rng2.shuffle(shuffled2)
    orderings.append(shuffled2)
    
    # Strategy 6: Sort by x, then y (row-major)
    orderings.append(sorted(all_points, key=lambda p: (ys[p], xs[p])))
    
    # Strategy 7: Sort by x*y (hyperbolic)
    orderings.append(sorted(all_points, key=lambda p: xs[p] * ys[p]))
    
    # Try each ordering
    for order_idx, ordering in enumerate(orderings):
        if time.time() > deadline - 120:
            break
        
        # Reset state
        selected.clear()
        dist_sets.clear()
        
        # Greedy add in this order
        for p in ordering:
            if time.time() > deadline - 120:
                break
            if can_add(p):
                add(p)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_state = selected.copy()
            best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
    
    # Restore best state
    selected.clear()
    dist_sets.clear()
    for p in best_state:
        selected.add(p)
    for k, v in best_dist_sets.items():
        dist_sets[k] = v.copy()
    
    unselected = [p for p in all_points if p not in selected]
    
    # Phase 2: Local search with swaps and adds
    rng_search = random.Random(99999)
    
    stuck_count = 0
    max_stuck = 200
    
    while time.time() < deadline:
        # Occasionally try to add a point
        if rng_search.random() < 0.3 and unselected:
            p = rng_search.choice(unselected)
            if can_add(p):
                add(p)
                unselected.remove(p)
                stuck_count = 0
                continue
        
        # Try swap
        if not selected or not unselected:
            continue
            
        p_out = rng_search.choice(list(selected))
        p_in = rng_search.choice(unselected)
        
        if try_swap(p_out, p_in):
            unselected.remove(p_in)
            unselected.append(p_out)
            stuck_count = 0
        else:
            stuck_count += 1
        
        # If stuck for too long, try a more aggressive restart-like move
        if stuck_count > max_stuck:
            # Remove a random point and try to add a different one
            if selected:
                to_remove = rng_search.choice(list(selected))
                remove(to_remove)
                if to_remove not in unselected:
                    unselected.append(to_remove)
            
            # Try adding several random points
            attempts = 0
            while attempts < 50 and unselected:
                candidate = rng_search.choice(unselected)
                if can_add(candidate):
                    add(candidate)
                    unselected.remove(candidate)
                    break
                attempts += 1
            
            stuck_count = 0
    
    result = [(i // N, i % N) for i in selected]
    return result

# EVOLVE-BLOCK-END
