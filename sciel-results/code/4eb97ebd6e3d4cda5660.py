import itertools
import time
import random
import sys
import numpy as np

N = 16
_DEADLINE = time.time() + 275

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute(all_points):
    n = len(all_points)
    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _can_add(x, chosen_set, dist):
    """
    Check if point x can be added to chosen_set without creating an isosceles triangle.
    """
    if not chosen_set:
        return True
    
    seen_distances = set()
    for p in chosen_set:
        d_xp = int(dist[x, p])
        if d_xp in seen_distances:
            return False
        seen_distances.add(d_xp)
        
    # Check condition 2: for each p, d(x,p) must not be in distances from p to other chosen points
    for p in chosen_set:
        d_xp = int(dist[x, p])
        for q in chosen_set:
            if q == p:
                continue
            if int(dist[p, q]) == d_xp:
                return False
    return True

def _is_maximal(chosen_set, all_indices, dist):
    """Check if no point can be added to chosen_set."""
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen_set, dist):
            return False
    return True

def _greedy_maximal(all_indices, dist, rng):
    """
    Construct a maximal set by greedily adding points.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen_set = set()
    for p in order:
        if _can_add(p, chosen_set, dist):
            chosen_set.add(p)
    # Ensure maximality (greedy usually produces maximal sets, but let's be sure)
    # Actually, if we iterate through ALL points and add if possible, the result is maximal.
    # Because if a point x was skipped, it was because it couldn't be added to the set *at that time*.
    # Since we only add points, the set only grows, so if x couldn't be added then, it can't be added now.
    # So the result is always maximal.
    return chosen_set

def _remove_and_remaximize(chosen_set, all_indices, dist, rng):
    """
    Remove a random point, then greedily add points to restore maximality.
    Return the new set if it is smaller or equal, else None.
    """
    if not chosen_set:
        return chosen_set
        
    s = set(chosen_set)
    # Pick a random point to remove
    p_to_remove = rng.choice(list(s))
    s.remove(p_to_remove)
    
    # Now s is not maximal. We need to add points to make it maximal.
    # We want to add as few points as possible.
    # Strategy: Find all addable points. Pick one that "blocks" the most other addable points?
    # Or just pick randomly?
    
    # Let's find all points that can be added to s
    addable = [x for x in all_indices if x not in s and _can_add(x, s, dist)]
    
    if not addable:
        # s is already maximal? Impossible, we just removed a point from a maximal set.
        # Unless the removed point was redundant for maximality.
        # If s is maximal, we have improved (size decreased).
        return s
        
    # Greedily add points from addable to s to make it maximal.
    # We want to minimize the number of additions.
    # Heuristic: Add the point that blocks the most other candidates.
    
    current_s = set(s)
    candidates = set(addable)
    
    while candidates:
        # Check if current_s is maximal
        if _is_maximal(current_s, all_indices, dist):
            break
            
        # Pick a candidate
        # Score each candidate by how many other candidates it would block
        best_cand = None
        best_score = -1
        # Sample if candidates is large
        sample = candidates if len(candidates) < 20 else set(rng.sample(list(candidates), 20))
        
        for c in sample:
            # If we add c, which other candidates become non-addable?
            # This is expensive to compute exactly.
            # Approximation: Just pick randomly?
            # Let's try a simple heuristic: pick the one with the most "conflicts" with other candidates.
            score = 0
            for other in candidates:
                if other == c:
                    continue
                # Check if c and other form an isosceles triangle with some point in current_s?
                # Or if adding c makes other un-addable?
                # If d(c, p) == d(other, p) for some p in current_s, they might conflict.
                # This is getting complicated. Let's just pick randomly from candidates.
                pass
            
            # Simple random pick
            # Actually, let's just pick the first one that allows us to make progress?
            # Let's just pick randomly.
            best_cand = c
            break # Just take the first one in sample? No, let's pick randomly.
        
        # Let's just pick a random candidate
        c = rng.choice(list(candidates))
        current_s.add(c)
        candidates.remove(c)
        
        # Remove from candidates any points that are no longer addable
        new_candidates = set()
        for x in candidates:
            if _can_add(x, current_s, dist):
                new_candidates.add(x)
        candidates = new_candidates
        
    return current_s

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist = _precompute(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Phase 1: Generate initial maximal sets
    while time.time() < _DEADLINE - 60:
        # Start with a small random set? No, greedy from empty gives a maximal set.
        # But greedy from empty with random order gives different sizes.
        chosen = _greedy_maximal(all_indices, dist, rng)
        
        # Try to reduce size using Remove-and-Re-maximize
        improved = True
        while improved and time.time() < _DEADLINE - 30:
            improved = False
            # Try multiple removals
            for _ in range(10):
                new_set = _remove_and_remaximize(chosen, all_indices, dist, rng)
                if new_set is not None and len(new_set) < len(chosen):
                    chosen = new_set
                    improved = True
                    break
                elif new_set is not None and len(new_set) == len(chosen) and rng.random() < 0.5:
                    # Accept same size to escape local optima?
                    # Might be risky. Let's only accept if strictly smaller for now.
                    pass
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen
            
    # Phase 2: Local Search on the best solution found
    if best is not None:
        s = set(best)
        while time.time() < _DEADLINE:
            # Try remove-and-re-maximize on the best solution
            improved = False
            for _ in range(5):
                new_set = _remove_and_remaximize(s, all_indices, dist, rng)
                if new_set is not None and len(new_set) < len(s):
                    s = new_set
                    improved = True
                    break
            if not improved:
                break
                
        if len(s) < best_size:
            best_size = len(s)
            best = s

    if best is None:
        best = _greedy_maximal(all_indices, dist, rng)
        
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
