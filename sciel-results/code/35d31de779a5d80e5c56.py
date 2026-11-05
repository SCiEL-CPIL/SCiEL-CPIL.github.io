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

def _generate_offsets(max_radius):
    """Precompute all integer offsets (dx, dy) with dx^2 + dy^2 a perfect square."""
    offsets = []
    for dx in range(-max_radius, max_radius + 1):
        for dy in range(-max_radius, max_radius + 1):
            s = dx * dx + dy * dy
            if s > 0 and _is_square(s):
                offsets.append((dx, dy))
    return offsets

def _try_extend(current, deadline, offset_cache=None):
    # current: list of (x, y) integer points
    # try adding a point from a bounded grid; if found, verify full set.
    n = len(current)
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    xmin, xmax = min(xs) - 200, max(xs) + 200
    ymin, ymax = min(ys) - 200, max(ys) + 200

    existing = set(current)
    x0, y0 = current[0]
    x1, y1 = current[1]

    # Precompute squared distances between existing points
    existing_dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            existing_dists[(i, j)] = _sqdist(current[i], current[j])

    # Use two reference points to constrain candidates
    # For candidate (x, y), we need:
    #   (x-x0)^2 + (y-y0)^2 = a^2 for some integer a
    #   (x-x1)^2 + (y-y1)^2 = b^2 for some integer b
    # This means the candidate lies at intersection of two circles.
    
    max_dist = 300
    cand = None

    # Enumerate possible distances from point 0 and point 1
    # For each pair (a, b), solve for intersection points
    for a in range(1, max_dist + 1):
        if time.perf_counter() > deadline:
            break
        a2 = a * a
        for b in range(1, max_dist + 1):
            # Check triangle inequality with existing distance d01
            d01 = isqrt(existing_dists[(0, 1)])
            if a + b < d01 or abs(a - b) > d01:
                continue
            
            # Find intersection of circles:
            # (x-x0)^2 + (y-y0)^2 = a^2
            # (x-x1)^2 + (y-y1)^2 = b^2
            # Subtract: (x-x1)^2 - (x-x0)^2 + (y-y1)^2 - (y-y0)^2 = b^2 - a^2
            # 2x(x0-x1) + (x0^2 - x1^2) + 2y(y0-y1) + (y0^2 - y1^2) = b^2 - a^2
            # 2x(x0-x1) + 2y(y0-y1) = b^2 - a^2 - (x0^2 - x1^2) - (y0^2 - y1^2)
            
            dx01 = x1 - x0
            dy01 = y1 - y0
            rhs = a2 - b2 + x1*x1 - x0*x0 + y1*y1 - y0*y0
            
            # Line: 2*dx01*x + 2*dy01*y = rhs
            # We need integer solutions
            # Try to find integer (x, y) on this line within bounds
            # and check if they're on the first circle
            
            # Parametrize: if dx01 != 0, x = (rhs - 2*dy01*y) / (2*dx01)
            # For integer x, need (rhs - 2*dy01*y) divisible by 2*dx01
            
            # Simpler: just scan y in range and check if x is integer
            # But this is O(max_dist) per (a,b) pair, which is too slow.
            
            # Alternative: for each y in [ymin, ymax], compute x and check
            # Only do this if the line actually has solutions in range
            if dx01 == 0 and dy01 == 0:
                continue
            
            # Check if line has any integer points in our search region
            # The line equation: 2*dx01*x + 2*dy01*y = rhs
            # For integer solutions, rhs must be even (since LHS is always even)
            if rhs % 2 != 0:
                continue
            
            half_rhs = rhs // 2
            # dx01*x + dy01*y = half_rhs
            
            # Find gcd and check divisibility
            g = _gcd(abs(dx01), abs(dy01))
            if g == 0:
                continue
            if half_rhs % g != 0:
                continue
            
            # Now find one particular solution
            # Extended Euclidean or just scan a small range
            # For efficiency, only scan if the line passes through our bounding box
            
            # Estimate y range for which x would be in [xmin, xmax]
            if dx01 != 0:
                # x = (half_rhs - dy01*y) / dx01
                # xmin <= (half_rhs - dy01*y) / dx01 <= xmax
                # Need to handle sign of dx01
                if dx01 > 0:
                    y_low = (half_rhs - dx01 * xmax) // dy01 if dy01 != 0 else -10**9
                    y_high = (half_rhs - dx01 * xmin) // dy01 if dy01 != 0 else 10**9
                else:
                    y_low = (half_rhs - dx01 * xmin) // dy01 if dy01 != 0 else -10**9
                    y_high = (half_rhs - dx01 * xmax) // dy01 if dy01 != 0 else 10**9
                
                # Clamp to our search range
                y_start = max(ymin, y_low - 1)
                y_end = min(ymax, y_high + 1)
                
                if y_start > y_end:
                    continue
                
                # Scan y values
                for y in range(y_start, y_end + 1):
                    num = half_rhs - dy01 * y
                    if num % dx01 != 0:
                        continue
                    x = num // dx01
                    if x < xmin or x > xmax or y < ymin or y > ymax:
                        continue
                    if (x, y) in existing:
                        continue
                    
                    # Verify it's on the first circle
                    dx = x - x0
                    dy = y - y0
                    if dx*dx + dy*dy != a2:
                        continue
                    
                    # Check distances to all other existing points
                    ok = True
                    for i in range(2, n):
                        dxi = x - current[i][0]
                        dyi = y - current[i][1]
                        if not _is_square(dxi * dxi + dyi * dyi):
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Quick collinearity test
                    bad = False
                    for ai in range(n):
                        for bi in range(ai + 1, n):
                            if _collinear3(current[ai], current[bi], (x, y)) == 0:
                                bad = True
                                break
                        if bad:
                            break
                    if bad:
                        continue
                    
                    # Quick concyclicity with 3 existing points
                    for ai, bi, ci in combinations(range(n), 3):
                        if _conic_coeff(current[ai], current[bi], current[ci], (x, y)) == 0:
                            bad = True
                            break
                    if bad:
                        continue
                    
                    cand = (x, y)
                    break
                
                if cand is not None:
                    break
            
            if cand is not None:
                break
        
        if cand is not None:
            break

    if cand is None:
        # Fallback: simpler grid search with precomputed offsets
        if offset_cache is None:
            offset_cache = _generate_offsets(150)
        
        for dx, dy in offset_cache:
            if time.perf_counter() > deadline:
                break
            x = x0 + dx
            y = y0 + dy
            if (x, y) in existing:
                continue
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            
            ok = True
            for i in range(1, n):
                dxi = x - current[i][0]
                dyi = y - current[i][1]
                if not _is_square(dxi * dxi + dyi * dyi):
                    ok = False
                    break
            if not ok:
                continue
            
            bad = False
            for ai in range(n):
                for bi in range(ai + 1, n):
                    if _collinear3(current[ai], current[bi], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            
            for ai, bi, ci in combinations(range(n), 3):
                if _conic_coeff(current[ai], current[bi], current[ci], (x, y)) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            cand = (x, y)
            break

    if cand is None:
        return current

    new_set = current + [cand]
    if _valid_set(new_set):
        return new_set
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
    offset_cache = None

    # Try iterative extension with time guard
    while len(best) < 12 and (time.perf_counter() < deadline):
        extended = _try_extend(best, deadline, offset_cache)
        if len(extended) <= len(best):
            break
        best = extended

    # Final safeguard
    if not _valid_set(best):
        # Ensure we return at least a minimal valid set
        best = base[:3]

    return _format_points(best)
