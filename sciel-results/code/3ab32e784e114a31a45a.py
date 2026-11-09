"""Integral point sets in general position - search for maximum size."""
import time
from fractions import Fraction
from math import gcd, isqrt
import random

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = pts[k]
                if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    x4, y4 = pts[l]
                    a1, b1, c1 = x1, y1, x1*x1 + y1*y1
                    a2, b2, c2 = x2, y2, x2*x2 + y2*y2
                    a3, b3, c3 = x3, y3, x3*x3 + y3*y3
                    a4, b4, c4 = x4, y4, x4*x4 + y4*y4
                    def det3(r1, r2, r3):
                        return (r1[0]*(r2[1]*r3[2]-r2[2]*r3[1])
                                - r1[1]*(r2[0]*r3[2]-r2[2]*r3[0])
                                + r1[2]*(r2[0]*r3[1]-r2[1]*r3[0]))
                    d = (det3((a2,b2,c2),(a3,b3,c3),(a4,b4,c4))
                         - det3((a1,b1,c1),(a3,b3,c3),(a4,b4,c4))
                         + det3((a1,b1,c1),(a2,b2,c2),(a4,b4,c4))
                         - det3((a1,b1,c1),(a2,b2,c2),(a3,b3,c3)))
                    if d == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx*dx + D*dy*dy
            if not _is_square(d2):
                return False
    return True

def _validate(pts, D):
    if len(pts) < 3:
        return False
    if len(set(pts)) != len(pts):
        return False
    if not _all_distances_rational(pts, D):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

def _precompute_sq_norms(pts):
    """Precompute x^2 + y^2 for each point."""
    return [x*x + y*y for x, y in pts]

def _try_add_point_fast(pts, D, candidate, sq_norms):
    """Check if adding candidate to pts keeps validity, using precomputed norms."""
    cx, cy = candidate
    n = len(pts)
    
    # Check distances from candidate to existing points
    for i in range(n):
        x1, y1 = pts[i]
        dx = cx - x1
        dy = cy - y1
        d2 = dx*dx + D*dy*dy
        if not _is_square(d2):
            return False
    
    # Check collinearity: candidate with any pair in pts
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx * (cy - y1) - dy * (cx - x1) == 0:
                return False
    
    # Check concyclicity: candidate with any triple in pts
    c4 = cx*cx + cy*cy
    for i in range(n):
        x1, y1 = pts[i]
        c1 = sq_norms[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            c2 = sq_norms[j]
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                c3 = sq_norms[k]
                def det3(r1, r2, r3):
                    return (r1[0]*(r2[1]*r3[2]-r2[2]*r3[1])
                            - r1[1]*(r2[0]*r3[2]-r2[2]*r3[0])
                            + r1[2]*(r2[0]*r3[1]-r2[1]*r3[0]))
                d = (det3((x1,y1,c1),(x2,y2,c2),(x3,y3,c3))
                     - det3((x1,y1,c1),(x2,y2,c2),(cx,cy,c4))
                     + det3((x1,y1,c1),(x3,y3,c3),(cx,cy,c4))
                     - det3((x2,y2,c2),(x3,y3,c3),(cx,cy,c4)))
                if d == 0:
                    return False
    return True

def _generate_candidates_with_distance_constraint(pts, D, max_coord):
    """Generate candidate points that are at rational distance from all existing points."""
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            # Quick filter: check distance to first point
            dx = x - pts[0][0]
            dy = y - pts[0][1]
            d2 = dx*dx + D*dy*dy
            if not _is_square(d2):
                continue
            # Check distance to all other points
            valid = True
            for p in pts[1:]:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + D*dy*dy
                if not _is_square(d2):
                    valid = False
                    break
            if valid:
                candidates.append((x, y))
    return candidates

def _search_greedy_fast(D, seed_pts, max_coord, time_limit):
    """Fast greedy search with pre-filtered candidates."""
    current = list(seed_pts)
    best = list(current)
    current_set = set(current)
    sq_norms = _precompute_sq_norms(current)
    
    # Pre-filter candidates that have rational distance to ALL seed points
    candidates = _generate_candidates_with_distance_constraint(current, D, max_coord)
    # Sort by distance from centroid for better locality
    cx = sum(p[0] for p in current) / len(current)
    cy = sum(p[1] for p in current) / len(current)
    candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    
    for candidate in candidates:
        if time.time() > time_limit:
            break
        if candidate in current_set:
            continue
        if _try_add_point_fast(current, D, candidate, sq_norms):
            current.append(candidate)
            current_set.add(candidate)
            sq_norms.append(candidate[0]**2 + candidate[1]**2)
            if len(current) > len(best):
                best = list(current)
                if len(best) >= 12:
                    break
    return best

def _local_search_swap(D, pts, max_coord, time_limit):
    """Try to improve by adding new points."""
    current = list(pts)
    best = list(current)
    current_set = set(current)
    sq_norms = _precompute_sq_norms(current)
    
    # Generate all candidates with rational distances
    all_candidates = _generate_candidates_with_distance_constraint(current, D, max_coord)
    
    # Sort by distance from centroid
    cx = sum(p[0] for p in current) / len(current)
    cy = sum(p[1] for p in current) / len(current)
    all_candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    
    # Try adding any candidate
    for candidate in all_candidates:
        if time.time() > time_limit:
            break
        if candidate in current_set:
            continue
        if _try_add_point_fast(current, D, candidate, sq_norms):
            current.append(candidate)
            current_set.add(candidate)
            sq_norms.append(candidate[0]**2 + candidate[1]**2)
            if len(current) > len(best):
                best = list(current)
    
    return best

def construct():
    D = 1
    
    # Known valid 5-point set
    seed5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate(seed5, D):
        # Fallback
        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in seed5]}
    
    best_pts = list(seed5)
    time_limit = _DEADLINE - 5.0
    
    # Try greedy search from seed5 with various max_coord values
    for mc in [60, 80, 100, 120, 150]:
        if time.time() > time_limit:
            break
        grown = _search_greedy_fast(D, seed5, mc, time_limit)
        if len(grown) > len(best_pts):
            best_pts = grown
    
    # Try multiple different seeds
    other_seeds = [
        [(0, 0), (3, 4), (0, 5)],
        [(0, 0), (5, 0), (0, 12)],
        [(0, 0), (9, 12), (0, 15)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (2, 0), (0, 3)],
        [(0, 0), (4, 3), (0, 5)],
        [(0, 0), (5, 0), (5, 12)],
        [(0, 0), (7, 0), (0, 24)],
        [(0, 0), (8, 0), (0, 15)],
        [(0, 0), (12, 0), (0, 5)],
    ]
    
    for seed in other_seeds:
        if time.time() > time_limit:
            break
        if _validate(seed, D):
            for mc in [50, 70, 90]:
                if time.time() > time_limit:
                    break
                grown = _search_greedy_fast(D, seed, mc, time_limit)
                if len(grown) > len(best_pts):
                    best_pts = grown
    
    # Try local search improvement on best_pts
    if time.time() < time_limit:
        improved = _local_search_swap(D, best_pts, 100, time_limit)
        if len(improved) > len(best_pts):
            best_pts = improved
    
    # Try random restarts with different initial configurations
    if time.time() < time_limit:
        rng = random.Random(42)
        for restart in range(5):
            if time.time() > time_limit:
                break
            # Random 3-point seed
            seed = [(0, 0)]
            for _ in range(2):
                x = rng.randint(-30, 30)
                y = rng.randint(-30, 30)
                if (x, y) not in seed and _is_square(x*x + y*y):
                    seed.append((y, x))  # swapped to get variety
            if len(seed) == 3 and _validate(seed, D):
                grown = _search_greedy_fast(D, seed, 80, time_limit)
                if len(grown) > len(best_pts):
                    best_pts = grown
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
