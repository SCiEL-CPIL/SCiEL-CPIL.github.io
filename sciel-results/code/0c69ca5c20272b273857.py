import time
import math
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = 140.0  # safety margin vs 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _norm_sq(p1, p2, D):
    """Return (num, den) for squared distance between p1 and p2 in Q(sqrt(D))."""
    x1n, x1d, y1n, y1d = p1
    x2n, x2d, y2n, y2d = p2
    dxn = x1n * x2d - x2n * x1d
    dxd = x1d * x2d
    dyn = y1n * y2d - y2n * y1d
    dyd = y1d * y2d
    num = dxn * dxn * dyd * dyd + D * dyn * dyn * dxd * dxd
    den = dxd * dxd * dyd * dyd
    return num, den

def _is_rational_dist(p1, p2, D):
    num, den = _norm_sq(p1, p2, D)
    return _is_square(num)

def _collinear(p1, p2, p3):
    x1, y1 = Fraction(p1[0], p1[1]), Fraction(p1[2], p1[3])
    x2, y2 = Fraction(p2[0], p2[1]), Fraction(p2[2], p2[3])
    x3, y3 = Fraction(p3[0], p3[1]), Fraction(p3[2], p3[3])
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _concyclic(p1, p2, p3, p4):
    pts = []
    for p in (p1, p2, p3, p4):
        x = Fraction(p[0], p[1])
        y = Fraction(p[2], p[3])
        pts.append((x, y))
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    x4, y4 = pts[3]
    r1 = (x1*x1 + y1*y1 - (x4*x4 + y4*y4), x1 - x4, y1 - y4)
    r2 = (x2*x2 + y2*y2 - (x4*x4 + y4*y4), x2 - x4, y2 - y4)
    r3 = (x3*x3 + y3*y3 - (x4*x4 + y4*y4), x3 - x4, y3 - y4)
    det = (r1[0] * (r2[1]*r3[2] - r2[2]*r3[1]) -
           r1[1] * (r2[0]*r3[2] - r2[2]*r3[0]) +
           r1[2] * (r2[0]*r3[1] - r2[1]*r3[0]))
    return det == 0

def _valid_add(pts, cand, D):
    for p in pts:
        if not _is_rational_dist(cand, p, D):
            return False
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            if _collinear(cand, pts[i], pts[j]):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _concyclic(cand, pts[i], pts[j], pts[k]):
                    return False
    return True

def _try_add(pts, cand, D):
    if not _valid_add(pts, cand, D):
        return None
    return pts + [cand]

def construct():
    start = time.time()
    
    # Known 5-point integral point set in Z^2 (D=1)
    base = [
        [0, 1, 0, 1],
        [-56, 1, 33, 1],
        [-16, 1, 63, 1],
        [0, 1, 33, 1],
        [16, 1, 63, 1]
    ]
    D = 1
    best = base[:]
    
    # Generate candidates: grid + rational points
    candidates = []
    for x in range(-80, 81):
        for y in range(-80, 81):
            candidates.append([x, 1, y, 1])
    
    # Add some rational candidates with small denominators
    for xd in range(1, 8):
        for yd in range(1, 8):
            for x in range(-20, 21):
                for y in range(-20, 21):
                    g = _gcd(x, xd)
                    if g > 1:
                        x //= g
                        xd //= g
                    g = _gcd(y, yd)
                    if g > 1:
                        y //= g
                        yd //= g
                    candidates.append([x, xd, y, yd])
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Remove base points
    base_set = set(tuple(p) for p in base)
    unique_candidates = [c for c in unique_candidates if tuple(c) not in base_set]
    
    # Greedy search: repeatedly add the best candidate
    current = base[:]
    current_set = set(tuple(p) for p in current)
    
    while _time_left() > 10:
        # Find all valid candidates
        valid = []
        for c in unique_candidates:
            if tuple(c) in current_set:
                continue
            if _valid_add(current, c, D):
                valid.append(c)
        
        if not valid:
            # No more integer/rational points can be added; try random search
            found = False
            for _ in range(5000):
                if _time_left() < 5:
                    break
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                xd = random.choice([1, 1, 1, 2, 3, 4, 5])
                yd = random.choice([1, 1, 1, 2, 3, 4, 5])
                g = _gcd(x, xd)
                x //= g; xd //= g
                g = _gcd(y, yd)
                y //= g; yd //= g
                cand = [x, xd, y, yd]
                if tuple(cand) in current_set:
                    continue
                if _valid_add(current, cand, D):
                    current.append(cand)
                    current_set.add(tuple(cand))
                    found = True
                    break
            if not found:
                break
        
        # Sort valid candidates by some heuristic (e.g., distance to centroid)
        cx = sum(Fraction(p[0], p[1]) for p in current) / len(current)
        cy = sum(Fraction(p[2], p[3]) for p in current) / len(current)
        
        def dist_to_centroid(c):
            cx_c = Fraction(c[0], c[1])
            cy_c = Fraction(c[2], c[3])
            return abs(cx_c - cx) + abs(cy_c - cy)
        
        valid.sort(key=dist_to_centroid)
        
        # Add the first valid candidate
        new_pt = valid[0]
        current.append(new_pt)
        current_set.add(tuple(new_pt))
        best = current[:]
    
    # Convert to required format
    final_points = []
    for p in best:
        x_num, x_den, y_num, y_den = p
        final_points.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': D, 'points': final_points}
