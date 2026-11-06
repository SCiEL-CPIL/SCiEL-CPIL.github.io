"""Rewritten program: search for 20 integer points (|x|,|y|<=100) with no four collinear,
minimizing the largest subset with no three collinear (exact branch and bound)."""

import itertools
import time
import random
from math import gcd

N = 20
DEADLINE_MARGIN = 15  # seconds safety margin before the 150s kill


def _collinear3(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0


def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _collinear3(points[i], points[j], points[k]) and \
                       _collinear3(points[i], points[j], points[l]):
                        return True
    return False


def _max_no_three_collinear(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    # Precompute all 3-collinear triples
    triples = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(points[i], points[j], points[k]):
                    triples.add((i, j, k))
    
    # For each point, which triples does it participate in?
    point_in_triples = [[] for _ in range(n)]
    for (i, j, k) in triples:
        point_in_triples[i].append((j, k))
        point_in_triples[j].append((i, k))
        point_in_triples[k].append((i, j))
    
    # Branch and bound
    best = [0]
    
    def backtrack(remaining, chosen_count, chosen_set):
        if not remaining:
            best[0] = max(best[0], chosen_count)
            return
        # Upper bound: can't do better than chosen_count + len(remaining)
        if chosen_count + len(remaining) <= best[0]:
            return
        
        # Try adding the first point
        p = remaining[0]
        rest = remaining[1:]
        
        # Check if p can be added: no triple fully in chosen_set ∪ {p}
        can_add = True
        for (j, k) in point_in_triples[p]:
            if j in chosen_set and k in chosen_set:
                can_add = False
                break
        
        if can_add:
            chosen_set.add(p)
            # Filter remaining: remove any point that would form a triple with two in chosen
            new_remaining = []
            for q in rest:
                ok = True
                for (j, k) in point_in_triples[q]:
                    if j in chosen_set and k in chosen_set:
                        ok = False
                        break
                if ok:
                    new_remaining.append(q)
            backtrack(new_remaining, chosen_count + 1, chosen_set)
            chosen_set.remove(p)
        
        # Try not adding p
        backtrack(rest, chosen_count, chosen_set)
    
    backtrack(list(range(n)), 0, set())
    return best[0]


def _construct_seed():
    """Generate a valid 20-point configuration with no four collinear."""
    # Use a perturbed parabola or other construction
    # Points on y = x^2 for x in -10..9 have no three collinear (hence no four)
    # But score is 20 (all points no three collinear), which is maximal not minimal.
    # We want to MINIMIZE the score, so we want configurations where many triples are collinear
    # but no four are.
    
    # Strategy: use a grid-like arrangement with controlled collinearities.
    # For n=20, we want to maximize 3-collinear triples while keeping no 4 collinear.
    # This is the inverse of the no-three-collinear problem.
    
    # A simple approach: take points from a 4x5 grid, but perturb to avoid 4-collinear.
    # Actually, let's just use a known construction or random search.
    
    # Start with a base: points on a line would give high score (all collinear = 20 no-three? No, 3 on a line violates no-three).
    # We need no four collinear, but we WANT many three-collinear triples to reduce the max no-three subset.
    
    # Let's use a 4x5 grid: {0,1,2,3} x {0,1,2,3,4} = 20 points
    # In a 4x5 grid, there are many 3-collinear triples (rows, cols, diagonals)
    # But also 4-collinear (rows of 4, cols of 5) - violates constraint!
    
    # So we need to perturb. Let's use a "twisted" grid.
    
    # Alternative: use points (i, j) for i in 0..3, j in 0..4, but shift each row.
    # Row i: points (i + offset_i, j) for j in 0..4
    # Choose offsets to break 4-collinear while preserving many 3-collinear.
    
    # Actually, simplest valid seed: use the parabola (score=20, valid but not optimal).
    # Then we'll try to improve via local search.
    return [[x, x * x] for x in range(-10, 10)]


def _local_search(points, deadline):
    """Try to improve the configuration by moving individual points."""
    start = time.time()
    current_points = [p[:] for p in points]
    current_score = _max_no_three_collinear(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Precompute candidate positions
    candidates = [(x, y) for x in range(-100, 101) for y in range(-100, 101)]
    
    improved = True
    while improved and (time.time() - start) < deadline:
        improved = False
        for i in range(N):
            if (time.time() - start) > deadline:
                break
            orig = current_points[i]
            # Try moving point i to a few random candidate positions
            # Use a limited set for speed
            for _ in range(200):
                if (time.time() - start) > deadline:
                    break
                new_pos = random.choice(candidates)
                if new_pos == orig:
                    continue
                # Check validity: distinct and no four collinear
                test_points = [p[:] for p in current_points]
                test_points[i] = list(new_pos)
                
                # Quick check: distinct
                seen = set()
                valid_distinct = True
                for p in test_points:
                    if tuple(p) in seen:
                        valid_distinct = False
                        break
                    seen.add(tuple(p))
                if not valid_distinct:
                    continue
                
                # Check no four collinear (expensive, so do a quick version)
                # Only check lines involving the moved point
                n = len(test_points)
                has_4 = False
                for j in range(n):
                    if j == i:
                        continue
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        for l in range(n):
                            if l == i or l == j or l == k:
                                continue
                            if _collinear3(test_points[i], test_points[j], test_points[k]) and \
                               _collinear3(test_points[i], test_points[j], test_points[l]):
                                has_4 = True
                                break
                        if has_4:
                            break
                    if has_4:
                        break
                if has_4:
                    continue
                
                # Compute score
                new_score = _max_no_three_collinear(test_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in test_points]
                    current_points = [p[:] for p in test_points]
                    improved = True
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + (150 - DEADLINE_MARGIN)
    
    # Start with a seed
    points = _construct_seed()
    score = _max_no_three_collinear(points)
    
    best_points = [p[:] for p in points]
    best_score = score
    
    # Local search with time limit
    if (time.time() - start_time) < deadline:
        best_points, best_score = _local_search(points, deadline - (time.time() - start_time))
    
    # Return the best found
    return [p[:] for p in best_points]
