"""Rewritten program: greedy hypergraph-based construction to minimize max no-three-collinear subset."""
import time
import random
from itertools import combinations

_DEADLINE = 115.0  # safety margin before 120s kill

def _cross(p1, p2, p3):
    """Cross product of (p2-p1) x (p3-p1). Zero iff collinear."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(j + 1, n):
                if _cross(points[i], points[j], points[k]) == 0:
                    count += 1
                    if count >= 4:
                        return True
    return False

def _build_triples(points):
    """Build list of all collinear triples as tuples of indices."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _cross(points[i], points[j], points[k]) == 0:
                    triples.append((i, j, k))
    return triples

def _count_three_collinear(points):
    """Count number of collinear triples."""
    triples = _build_triples(points)
    return len(triples)

def _max_independent_set_size(points, timeout=5.0):
    """
    Exact branch and bound for the largest subset with no three collinear.
    Returns the size of the largest such subset.
    """
    start = time.time()
    n = len(points)
    triples = _build_triples(points)
    
    # Precompute for each point, the set of triples it participates in
    point_triples = [[] for _ in range(n)]
    for t_idx, (a, b, c) in enumerate(triples):
        point_triples[a].append(t_idx)
        point_triples[b].append(t_idx)
        point_triples[c].append(t_idx)
    
    # We want to find the max subset S such that no triple is fully contained in S.
    # Equivalently, we want to remove the minimum number of points to hit all triples.
    # This is the hitting set / vertex cover problem in a 3-uniform hypergraph.
    # Score = n - (min hitting set size).
    
    # Branch and bound on the hitting set:
    # State: current set of points still "available" to be in the independent set,
    #        and the set of triples not yet hit.
    # We pick a point to either include or exclude.
    
    # Actually, let's do it the other way: we select points to REMOVE.
    # We want to minimize the number removed such that all triples are hit.
    # Equivalently, maximize the number kept such that no triple is fully kept.
    
    # Let's use a recursive approach:
    # f(kept_set, remaining_triples) -> max additional points we can keep
    # But this is expensive. Let's use a smarter approach.
    
    # Alternative: use the fact that n=20 is small. We can use a recursive branch and bound.
    # At each step, if there are no remaining triples, we can keep all remaining points.
    # Otherwise, pick a triple and branch on which of its 3 points to remove.
    
    # State: frozenset of points still available, frozenset of triple indices not yet hit
    # We want to minimize removals.
    
    # Let's represent state as (available_mask, remaining_triples_mask)
    # But with n=20, we can use integers as bitmasks.
    
    # Precompute triple masks
    triple_masks = []
    for (a, b, c) in triples:
        mask = (1 << a) | (1 << b) | (1 << c)
        triple_masks.append(mask)
    
    all_points_mask = (1 << n) - 1
    
    # We'll use a recursive function with memoization or just branch and bound.
    # Given the time constraint, let's do a simple branch and bound.
    
    best_found = 0
    
    def _search(available, remaining_triple_indices, removed_count, depth):
        nonlocal best_found
        if time.time() - start > timeout:
            return
        
        # Prune: if removed_count + (n - available.bit_count()) < n - best_found, 
        # i.e., even if we keep all remaining, we can't beat best_found
        # Actually, best_found is the max independent set size found so far.
        # Current kept = available.bit_count() - (points we'll remove from available)
        # We want to maximize kept, so minimize removals from available.
        # Lower bound on total removals = removed_count
        # Upper bound on kept = n - removed_count
        # If n - removed_count <= best_found, prune.
        if n - removed_count <= best_found:
            return
        
        if not remaining_triple_indices:
            # No more triples to hit, keep all available
            kept = available.bit_count()
            if kept > best_found:
                best_found = kept
            return
        
        # Pick a triple to branch on
        t_idx = remaining_triple_indices[0]
        t_mask = triple_masks[t_idx]
        
        # Find the points in this triple that are still available
        candidates = []
        for p in range(n):
            if t_mask & (1 << p) and available & (1 << p):
                candidates.append(p)
        
        if not candidates:
            # This triple is already hit (one of its points was removed)
            _search(available, remaining_triple_indices[1:], removed_count, depth + 1)
            return
        
        # Branch: remove one of the candidate points
        for p in candidates:
            new_available = available & ~(1 << p)
            # Recompute remaining triples: remove any triple that includes p
            new_remaining = []
            for idx in remaining_triple_indices:
                if not (triple_masks[idx] & (1 << p)):
                    new_remaining.append(idx)
            _search(new_available, new_remaining, removed_count + 1, depth + 1)
    
    _search(all_points_mask, list(range(len(triple_masks))), 0, 0)
    return best_found

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    return _max_independent_set_size(points, timeout=8.0)

def _construct_greedy():
    """
    Greedy construction: start with all points, iteratively remove the point
    that breaks the most collinear triples, while ensuring no four are collinear.
    """
    # Start with a reasonable set of candidate points
    # Use a grid or random points in [-100, 100]
    random.seed(42)
    
    # Generate many candidate points
    candidates = []
    for x in range(-50, 51):
        for y in range(-50, 51):
            candidates.append([x, y])
    
    # We need to select 20 points.
    # Strategy: start with a large set, then greedily remove points that are in many collinear triples.
    # But we need exactly 20 points with no four collinear.
    
    # Alternative: build up the set greedily.
    # Start with empty set, add points one by one, choosing the point that minimizes the increase in collinear triples.
    
    # Let's try a different approach: start with all 20 points from a known good configuration,
    # then use local search to improve.
    
    # Initial configuration: parabola y = x^2
    points = [[x, x * x] for x in range(-10, 10)]
    
    # Check validity
    if _has_four_collinear(points):
        # This shouldn't happen for the parabola, but just in case
        pass
    
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local search: try moving each point to a random nearby location
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < _DEADLINE - 20:  # leave 20s for final scoring
        iterations += 1
        # Pick a random point to move
        i = random.randint(0, 19)
        old_x, old_y = points[i]
        
        # Try a random new position
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        # Ensure distinctness
        if any(points[j][0] == new_x and points[j][1] == new_y for j in range(20) if j != i):
            continue
        
        points[i] = [new_x, new_y]
        
        if _has_four_collinear(points):
            points[i] = [old_x, old_y]
            continue
        
        new_score = _score(points)
        if new_score < best_score:
            best_score = new_score
            best_points = [p[:] for p in points]
        else:
            # Revert if not better (greedy hill climbing)
            points[i] = [old_x, old_y]
    
    return best_points

def construct():
    """Construct 20 distinct integer points with |x|, |y| <= 100 and no four collinear."""
    start = time.time()
    
    # Initial construction
    points = _construct_greedy()
    
    # Verify validity
    if len(set(tuple(p) for p in points)) != 20:
        # Fallback to parabola
        points = [[x, x * x] for x in range(-10, 10)]
    
    if _has_four_collinear(points):
        # Fallback to parabola
        points = [[x, x * x] for x in range(-10, 10)]
    
    for p in points:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            points = [[x, x * x] for x in range(-10, 10)]
            break
    
    return points
