"""Program using MODEL-AND-OPTIMIZE: local search + geometric verification for integral point sets.

Model:
- Parameterize as (D, list of points with rational x and y*sqrt(D)).
- Objective: maximize number of distinct points n (3..12) such that:
  * all pairwise squared distances are rational (equivalently integral after scaling for a fixed D)
  * no three points are collinear (exact determinant 0 check in affine 2-space)
  * no four points are concyclic (determinant-based cyclic condition using squared distances)
  * |numerators|, |denominators| <= 1e40, D squarefree <= 1e12
- For D=1, reduce to integer lattice with integer squared distances.

Optimization:
- Start from a known 5-point integral set.
- Perform a time-budgeted local search over a bounded integer grid to add a 6th point
  that (i) has integer squared distances to the existing 5 and (ii) preserves all
  collinearity/concyclicity constraints.
- If time remains, do a broader search up to n=8 on a slightly larger grid but keep
  constraints tight to stay within the wall-clock limit.
- Always return a valid configuration of maximum found n (starting from n=5).

Deadline:
- Use time.perf_counter(); stop loops before ~110s wall-clock, then finalize.
"""
import time
from math import isqrt
from itertools import combinations

TIME_LIMIT = 150.0
SAFETY = 30.0
DEADLINE = TIME_LIMIT - SAFETY

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a

def _sqdist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _collinear3(A, B, C):
    # (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x)
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_coeff(A, B, C, D):
    # determinant
    # | x1 y1 x1^2+y1^2 1 |
    # | x2 y2 x2^2+y2^2 1 |
    # | x3 y3 x3^2+y3^2 1 |
    # | x4 y4 x4^2+y4^2 1 |
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    s4 = x4*x4 + y4*y4
    # Expand determinant with column 4 elimination (use row1 as pivot)
    # D = x1*y2*(s3 - s4) - x1*y3*(s2 - s4) + x1*y4*(s2 - s3)
    #   - y1*x2*(s3 - s4) + y1*x3*(s2 - s4) - y1*x4*(s2 - s3)
    #   + (s2*y3*x4 - s2*y4*x3) - (s3*y2*x4 - s3*y4*x2) + (s4*y2*x3 - s4*y3*x2)
    d = (
        x1 * y2 * (s3 - s4) - x1 * y3 * (s2 - s4) + x1 * y4 * (s2 - s3)
        - y1 * x2 * (s3 - s4) + y1 * x3 * (s2 - s4) - y1 * x4 * (s2 - s3)
        + s2 * (y3 * x4 - y4 * x3)
        - s3 * (y2 * x4 - y4 * x2)
        + s4 * (y2 * x3 - y3 * x2)
    )
    return d

def _valid_set(pts):
    # Check distinct
    seen = set(pts)
    if len(seen) != len(pts):
        return False
    # Check all squared distances are perfect squares (D=1)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _is_square(_sqdist(pts[i], pts[j])):
                return False
    # No 3 collinear
    for a, b, c in combinations(pts, 3):
        if _collinear3(a, b, c) == 0:
            return False
    # No 4 concyclic
    for a, b, c, d in combinations(pts, 4):
        if _conic_coeff(a, b, c, d) == 0:
            return False
    return True

def _format_points(pts):
    # D=1 so y is rational y with den=1
    return {"D": 1, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _try_extend(current, deadline):
    """Try to extend an integral point set by one point using efficient search.
    
    Uses the fact that for D=1, we need integer points with all pairwise
    squared distances being perfect squares.
    """
    n = len(current)
    if n < 2:
        return current
    
    P0 = current[0]
    P1 = current[1]
    x0, y0 = P0
    x1, y1 = P1
    
    # Bound the search region
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    xmin, xmax = min(xs) - 300, max(xs) + 300
    ymin, ymax = min(ys) - 300, max(ys) + 300
    
    existing = set(current)
    
    # Precompute squared distances from candidate positions to existing points
    # For efficiency, we'll use the linear constraint from P0 and P1
    
    dx01 = x1 - x0
    dy01 = y1 - y0
    
    t0 = time.perf_counter()
    found = None
    
    # Strategy: iterate over x, then for each x, find y such that
    # (x-x0)^2 + (y-y0)^2 is a perfect square
    # This is equivalent to finding Pythagorean triples with one leg = |x-x0|
    
    for x in range(xmin, xmax + 1):
        if time.perf_counter() > deadline:
            break
        
        dx0 = x - x0
        dx0_sq = dx0 * dx0
        
        # Find all dy0 such that dx0_sq + dy0^2 is a perfect square
        # and y = y0 + dy0 is in [ymin, ymax]
        
        # Bound: dy0^2 <= (ymax - ymin)^2 + (xmax - xmin)^2
        max_range = max(ymax - ymin, xmax - xmin)
        max_dy0 = max_range + 100
        
        for dy0 in range(-max_dy0, max_dy0 + 1):
            y = y0 + dy0
            if y < ymin or y > ymax:
                continue
            if (x, y) in existing:
                continue
            
            # Check distance to P0 is perfect square
            r0_sq = dx0_sq + dy0 * dy0
            if not _is_square(r0_sq):
                continue
            
            # Check distance to P1
            dx1 = x - x1
            dy1 = y - y1
            r1_sq = dx1 * dx1 + dy1 * dy1
            if not _is_square(r1_sq):
                continue
            
            # Check distances to remaining points
            ok = True
            for i in range(2, n):
                dx = x - current[i][0]
                dy = y - current[i][1]
                if not _is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            
            # Quick collinearity test
            bad = False
            for a in range(n):
                for b in range(a + 1, n):
                    if _collinear3(current[a], current[b], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            
            # Quick concyclicity check with triples from existing points
            for a, b, c in combinations(range(n), 3):
                if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            # Candidate found
            found = (x, y)
            break
        
        if found is not None:
            break
    
    if found is None:
        return current
    
    new_set = current + [found]
    if _valid_set(new_set):
        return new_set
    
    # If full validation fails, continue searching
    checked = {found}
    
    for x in range(xmin, xmax + 1):
        if time.perf_counter() > deadline:
            break
        
        dx0 = x - x0
        dx0_sq = dx0 * dx0
        
        max_range = max(ymax - ymin, xmax - xmin)
        max_dy0 = max_range + 100
        
        for dy0 in range(-max_dy0, max_dy0 + 1):
            y = y0 + dy0
            if y < ymin or y > ymax:
                continue
            if (x, y) in existing or (x, y) in checked:
                continue
            checked.add((x, y))
            
            r0_sq = dx0_sq + dy0 * dy0
            if not _is_square(r0_sq):
                continue
            
            dx1 = x - x1
            dy1 = y - y1
            r1_sq = dx1 * dx1 + dy1 * dy1
            if not _is_square(r1_sq):
                continue
            
            ok = True
            for i in range(2, n):
                dx = x - current[i][0]
                dy = y - current[i][1]
                if not _is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            
            bad = False
            for a in range(n):
                for b in range(a + 1, n):
                    if _collinear3(current[a], current[b], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            
            for a, b, c in combinations(range(n), 3):
                if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            new_set2 = current + [(x, y)]
            if _valid_set(new_set2):
                return new_set2
            break
        else:
            continue
        break
    
    return current

def construct():
    # Seed 5-point set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Validate base (should be valid)
    if not _valid_set(base):
        # Fallback to smaller valid known set: use the first 3 points
        base = base[:3]

    t0 = time.perf_counter()
    deadline = t0 + DEADLINE

    best = base[:]

    # Try iterative extension with time guard
    while len(best) < 12 and (time.perf_counter() < deadline):
        extended = _try_extend(best, deadline)
        if len(extended) <= len(best):
            break
        best = extended

    # Final safeguard
    if not _valid_set(best):
        # Ensure we return at least a minimal valid set
        best = base[:3]

    return _format_points(best)
