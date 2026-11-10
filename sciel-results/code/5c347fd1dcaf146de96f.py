"""Rewritten program for integral point sets in general position."""
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

def _validate_set(pts, D):
    if len(pts) < 3:
        return False
    # Check distinctness
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if pts[i] == pts[j]:
                return False
    return (_all_distances_rational(pts, D) and
            _check_no_three_collinear(pts) and
            _check_no_four_concyclic(pts))

def _generate_pythagorean_candidates(center, max_coord):
    """Generate points at integer distance from center using Pythagorean triples."""
    candidates = set()
    cx, cy = center
    # Generate all Pythagorean triples with hypotenuse up to some limit
    max_hyp = int(math.sqrt(2) * max_coord) + 10
    for a in range(1, max_hyp):
        for b in range(a, max_hyp):
            c2 = a*a + b*b
            c = int(math.isqrt(c2))
            if c*c == c2:
                # (a, b, c) is a Pythagorean triple
                for dx, dy in [(a, b), (b, a), (-a, b), (a, -b),
                               (b, -a), (-b, a), (-a, -b), (-b, -a)]:
                    nx, ny = cx + dx, cy + dy
                    if -max_coord <= nx <= max_coord and -max_coord <= ny <= max_coord:
                        candidates.add((nx, ny))
                # Also handle the case where a or b is 0 (axis-aligned)
                for dx, dy in [(c, 0), (-c, 0), (0, c), (0, -c)]:
                    nx, ny = cx + dx, cy + dy
                    if -max_coord <= nx <= max_coord and -max_coord <= ny <= max_coord:
                        candidates.add((nx, ny))
    return candidates

def _score_candidate(pt, existing):
    """Score a candidate by how many existing points it has integer distance to."""
    count = 0
    for ep in existing:
        dx = pt[0] - ep[0]
        dy = pt[1] - ep[1]
        d2 = dx*dx + dy*dy
        if _is_square(d2):
            count += 1
    return count

def _greedy_expand(initial_set, candidates, max_points, deadline):
    """Greedy expansion of a set by adding candidates one at a time."""
    current = list(initial_set)
    if len(current) >= max_points:
        return current
    
    while len(current) < max_points:
        if time.time() > deadline:
            break
        
        # Find best candidate to add
        best_cand = None
        best_score = -1
        for cand in candidates:
            if cand in current:
                continue
            s = _score_candidate(cand, current)
            if s > best_score:
                best_score = s
                best_cand = cand
        
        if best_cand is None or best_score < len(current):
            break
        
        new_set = current + [best_cand]
        if _check_no_three_collinear(new_set) and _check_no_four_concyclic(new_set):
            current = new_set
        else:
            # Try other candidates with same score
            found = False
            for cand in sorted(candidates, key=lambda c: -_score_candidate(c, current)):
                if cand in current or cand == best_cand:
                    continue
                s = _score_candidate(cand, current)
                if s < len(current):
                    break
                new_set = current + [cand]
                if _check_no_three_collinear(new_set) and _check_no_four_concyclic(new_set):
                    current = new_set
                    found = True
                    break
            if not found:
                break
    
    return current

def _try_remove_and_replace(current, candidates, max_points, deadline):
    """Try removing one point and replacing with two new ones."""
    n = len(current)
    if n >= max_points:
        return current
    
    for idx in range(n):
        if time.time() > deadline:
            break
        remaining = current[:idx] + current[idx+1:]
        # Try to add two points to remaining
        expanded = _greedy_expand(remaining, candidates, max_points, deadline)
        if len(expanded) > n:
            return expanded
    return current

def _search_integral_points(max_coord=200, max_points=8, deadline=135.0):
    """Search for integer point sets with integer pairwise distances."""
    start_time = time.time()
    
    # Known 5-point solution
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = list(known_5)
    best_count = 5
    
    # Generate candidates around the known set
    candidates = set()
    for pt in known_5:
        candidates.update(_generate_pythagorean_candidates(pt, max_coord))
    
    # Also add some random-ish candidates
    import random
    random.seed(42)
    for _ in range(5000):
        x = random.randint(-max_coord, max_coord)
        y = random.randint(-max_coord, max_coord)
        candidates.add((x, y))
    
    candidates = list(candidates)
    
    # Greedy expansion from known set
    expanded = _greedy_expand(known_5, candidates, max_points, deadline)
    if len(expanded) > best_count:
        best_set = expanded
        best_count = len(expanded)
    
    # Try remove-and-replace
    improved = _try_remove_and_replace(best_set, candidates, max_points, deadline)
    if len(improved) > best_count:
        best_set = improved
        best_count = len(improved)
    
    return best_set if best_set else known_5

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Try to find a better configuration
    best_set = _search_integral_points(max_coord=200, max_points=8, deadline=deadline)
    
    # Verify the solution
    if not _validate_set(best_set, 1):
        # Fall back to known solution
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
# EVOLVE-BLOCK-END
