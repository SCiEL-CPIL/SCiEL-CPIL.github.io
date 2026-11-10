"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    b1 = a2 - a1
                    b2 = a3 - a1
                    b3 = a4 - a1
                    c1 = x2 - x1
                    c2 = x3 - x1
                    c3 = x4 - x1
                    d1 = y2 - y1
                    d2 = y3 - y1
                    d3 = y4 - y1
                    det = (b1 * (c2 * d3 - c3 * d2)
                           - c1 * (b2 * d3 - b3 * d2)
                           + d1 * (b2 * c3 - b3 * c2))
                    if det == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx * dx + D * dy * dy
            if not _is_square(d2):
                return False
    return True

def _dist_is_int(pt, existing, D):
    """Check if pt has integer distance to all points in existing."""
    for ep in existing:
        dx = pt[0] - ep[0]
        dy = pt[1] - ep[1]
        d2 = dx * dx + D * dy * dy
        if not _is_square(d2):
            return False
    return True

def _score_candidate(pt, existing, D):
    """Count how many existing points have integer distance to pt."""
    count = 0
    for ep in existing:
        dx = pt[0] - ep[0]
        dy = pt[1] - ep[1]
        d2 = dx * dx + D * dy * dy
        if _is_square(d2):
            count += 1
    return count

def _generate_candidates(max_coord=300, D=1):
    """Generate candidate points, prioritizing those on Pythagorean circles."""
    candidates = set()
    # Add origin and small lattice points
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.add((x, y))
    return list(candidates)

def _search(max_coord=200, max_points=8, deadline=145.0):
    """Search for integral point sets with no three collinear, no four concyclic."""
    start_time = time.time()
    
    # Known 5-point solution
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify known solution
    if not (_check_no_three_collinear(known_5) and 
            _check_no_four_concyclic(known_5) and 
            _all_distances_rational(known_5, D)):
        # Fallback to even smaller known sets
        known_5 = [(0, 0), (3, 4), (0, 5), (-3, 4)]
        if not (_check_no_three_collinear(known_5) and 
                _check_no_four_concyclic(known_5) and 
                _all_distances_rational(known_5, D)):
            known_5 = [(0, 0), (3, 4), (5, 0)]
    
    best_set = list(known_5)
    best_count = len(known_5)
    
    # Generate candidates
    candidates = _generate_candidates(max_coord, D)
    
    # Sort candidates by a heuristic: prefer points that are "interesting"
    # (have integer distances to many points from known configurations)
    def cand_priority(pt):
        # Prefer points with small coordinates first for efficiency
        return abs(pt[0]) + abs(pt[1])
    
    candidates.sort(key=cand_priority)
    
    # Try to extend the known set
    current = list(best_set)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        if len(current) >= max_points:
            break
        
        # Find best candidate to add
        best_cand = None
        best_score = -1
        
        for cand in candidates:
            if cand in current:
                continue
            s = _score_candidate(cand, current, D)
            if s > best_score:
                best_score = s
                best_cand = cand
        
        if best_cand is None or best_score < len(current):
            # Can't extend further with current approach
            # Try random restart with different starting points
            break
        
        # Check if adding best_cand maintains constraints
        new_set = current + [best_cand]
        if _check_no_three_collinear(new_set) and _check_no_four_concyclic(new_set):
            current = new_set
            if len(current) > best_count:
                best_count = len(current)
                best_set = list(current)
        else:
            # Try next best candidates
            found = False
            for cand in candidates:
                if cand in current or cand == best_cand:
                    continue
                s = _score_candidate(cand, current, D)
                if s < len(current):
                    break
                new_set = current + [cand]
                if _check_no_three_collinear(new_set) and _check_no_four_concyclic(new_set):
                    current = new_set
                    if len(current) > best_count:
                        best_count = len(current)
                        best_set = list(current)
                    found = True
                    break
            if not found:
                break
    
    return best_set, D

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_set, D = _search(max_coord=250, max_points=8, deadline=deadline)
    
    if time.time() > deadline - 3:
        # Fall back to known solution
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        D = 1
    
    # Final verification
    if not (_check_no_three_collinear(best_set) and 
            _check_no_four_concyclic(best_set) and 
            _all_distances_rational(best_set, D)):
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        D = 1
    
    # Ensure distinct points
    seen = set()
    unique_points = []
    for p in best_set:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)
    best_set = unique_points
    
    if len(best_set) < 3:
        best_set = [(0, 0), (3, 4), (5, 0)]
        D = 1
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
# EVOLVE-BLOCK-END
