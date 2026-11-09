"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

def _is_square(n):
    """Check if n is a perfect square (n must be non-negative integer)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _is_square_frac(f):
    """Check if Fraction f is a perfect square of a rational number."""
    if f < 0:
        return False
    # f = p/q in lowest terms. Need p and q to be perfect squares.
    p, q = f.numerator, f.denominator
    return _is_square(p) and _is_square(q)

def _check_rational_dist(p1, p2, D):
    """Check if squared distance is a perfect square rational."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    num = dx * dx + D * dy * dy
    return _is_square_frac(num)

def _check_no_three_collinear(points, D):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if val == 0:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    
                    det = x1*(y2 - y3) - y1*(x2 - x3) + (x2*y3 - x3*y2)
                    if det == 0:
                        continue
                    
                    rhs1 = -(x1*x1 + D*y1*y1)
                    rhs2 = -(x2*x2 + D*y2*y2)
                    rhs3 = -(x3*x3 + D*y3*y3)
                    
                    det_A = rhs1*(y2 - y3) - y1*(rhs2 - rhs3) + (rhs2*y3 - rhs3*y2)
                    det_B = x1*(rhs2 - rhs3) - rhs1*(x2 - x3) + (x2*rhs3 - x3*rhs2)
                    det_C = x1*(y2*rhs3 - y3*rhs2) - y1*(x2*rhs3 - x3*rhs2) + rhs1*(x2*y3 - x3*y2)
                    
                    lhs = x4*det_A + y4*det_B + det_C
                    rhs_check = -det * (x4*x4 + D*y4*y4)
                    
                    if lhs == rhs_check:
                        return False
    return True

def _is_valid(points, D):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if not _check_rational_dist(points[i], points[j], D):
                return False
    if not _check_no_three_collinear(points, D):
        return False
    if n >= 4:
        if not _check_no_four_concyclic(points, D):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: Use D=3, construct hexagon-like set, then perturb to break concyclicity
    D = 3
    
    # Regular hexagon vertices in Q(sqrt(3)):
    # In the field Q(sqrt(3)), we can represent points as (x, y*sqrt(3))
    # where x, y are rational.
    # 
    # A regular hexagon centered at origin with "radius" r has vertices at:
    # (r, 0), (r/2, r*sqrt(3)/2), (-r/2, r*sqrt(3)/2), (-r, 0), (-r/2, -r*sqrt(3)/2), (r/2, -r*sqrt(3)/2)
    #
    # In our representation (x, y*sqrt(3)):
    # (r, 0), (r/2, r/2), (-r/2, r/2), (-r, 0), (-r/2, -r/2), (r/2, -r/2)
    #
    # Let's use r = 2 for simplicity:
    # (2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)
    #
    # All pairwise distances:
    # Adjacent vertices: dx^2 + D*dy^2 where (dx,dy) = (1,-1), (2,0), etc.
    # From (2,0) to (1,1): dx=-1, dy=1, dist^2 = 1 + 3*1 = 4, dist=2 ✓
    # From (1,1) to (-1,1): dx=-2, dy=0, dist^2 = 4, dist=2 ✓
    # From (-1,1) to (-2,0): dx=-1, dy=-1, dist^2 = 1+3 = 4, dist=2 ✓
    # From (-2,0) to (-1,-1): dx=1, dy=-1, dist^2 = 1+3 = 4, dist=2 ✓
    # From (-1,-1) to (1,-1): dx=2, dy=0, dist^2 = 4, dist=2 ✓
    # From (1,-1) to (2,0): dx=1, dy=1, dist^2 = 1+3 = 4, dist=2 ✓
    #
    # Opposite vertices: dist^2 = 4 + 3*0 = 4 or 0 + 3*4 = 12
    # From (2,0) to (-2,0): dist^2 = 16, dist=4 ✓
    # From (1,1) to (-1,-1): dist^2 = 4 + 3*4 = 16, dist=4 ✓
    #
    # Other pairs (skip one):
    # From (2,0) to (-1,1): dx=-3, dy=1, dist^2 = 9+3 = 12, dist=2*sqrt(3) ✗ NOT rational!
    #
    # So the regular hexagon does NOT have all rational distances. We need a different approach.
    #
    # Let me think differently. We need 6 points in Q(sqrt(D)) with all rational distances.
    # 
    # Actually, let's try a simpler construction. Start with a known 5-point set and search for a 6th.
    # The seed program already does this. Let's improve the search.
    #
    # Alternative: Use D=1 (ordinary plane) and find 6 integer points with all rational distances
    # and in general position.
    
    # Let's use the known 5-point set and do a more efficient search for the 6th point.
    D = 1
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = base_points
    
    # For a candidate (x, y) to be at integer distance from (0,0):
    # x^2 + y^2 must be a perfect square.
    # We can precompute all (x, y) with x^2 + y^2 = k^2 for k up to some limit.
    # These are Pythagorean triples.
    
    max_k = 2000
    candidates = set()
    for a in range(1, max_k + 1):
        for b in range(a, max_k + 1):
            c2 = a*a + b*b
            c = math.isqrt(c2)
            if c * c == c2 and c <= max_k:
                # (a, b, c) is a Pythagorean triple
                candidates.add((a, b))
                candidates.add((a, -b))
                candidates.add((-a, b))
                candidates.add((-a, -b))
                if a != b:
                    candidates.add((b, a))
                    candidates.add((b, -a))
                    candidates.add((-b, a))
                    candidates.add((-b, -a))
    
    # Also add (0, ±k) and (±k, 0)
    for k in range(1, max_k + 1):
        candidates.add((0, k))
        candidates.add((0, -k))
        candidates.add((k, 0))
        candidates.add((-k, 0))
    
    # Now filter: must be at integer distance from ALL base points
    found_6th = None
    for x, y in candidates:
        if time.time() > deadline - 30:
            break
        cand = (x, y)
        if cand in base_points:
            continue
        
        valid_dist = True
        for bx, by in base_points:
            if not _is_square((x - bx)**2 + (y - by)**2):
                valid_dist = False
                break
        
        if valid_dist:
            new_set = base_points + [cand]
            if _check_no_three_collinear(new_set, D) and _check_no_four_concyclic(new_set, D):
                found_6th = cand
                break
    
    if found_6th:
        best_points = base_points + [found_6th]
        
        # Try to find 7th point
        if time.time() < deadline - 20:
            found_7th = None
            for x, y in candidates:
                if time.time() > deadline - 20:
                    break
                cand = (x, y)
                if cand in best_points:
                    continue
                valid_dist = True
                for bx, by in best_points:
                    if not _is_square((x - bx)**2 + (y - by)**2):
                        valid_dist = False
                        break
                if valid_dist:
                    new_set = best_points + [cand]
                    if _check_no_three_collinear(new_set, D) and _check_no_four_concyclic(new_set, D):
                        found_7th = cand
                        break
            
            if found_7th:
                best_points = best_points + [found_7th]
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
