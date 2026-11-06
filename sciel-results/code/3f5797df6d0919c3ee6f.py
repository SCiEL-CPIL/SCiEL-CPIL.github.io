"""
Search for integral point sets in general position (no 3 collinear, no 4 concyclic)
in the plane with points (x, y*sqrt(D)) for rational x, y.
"""
import time
import random
import math
from math import gcd

DEADLINE = 140.0  # safety margin before 150s kill

def _is_square(n):
    """Check if n is a perfect square using isqrt."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist_int(p, q, D):
    """Squared distance between integer points. Returns int."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _dist_rational_int(p, q, D):
    """Check if squared distance is a perfect square for integer coords."""
    d2 = _sq_dist_int(p, q, D)
    return _is_square(d2)

def _no_three_collinear_int(points):
    """Check no three points are collinear. Points are (x,y) int tuples."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic_int(points, D):
    """Check no four points are concyclic for integer points in (x, y*sqrt(D)) space."""
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
                    # Circle through p1, p2, p3: equation x^2 + y^2 + Ax + By + C = 0
                    # For point (x, y*sqrt(D)): x^2 + D*y^2 + Ax + B*y*sqrt(D) + C = 0
                    # This must hold for all 4 points. Using the form:
                    # x^2 + D*y^2 + Ax + B*y + C = 0 (treating y as the coordinate)
                    # Actually in (x, y*sqrt(D)) space, the "circle" equation is:
                    # x^2 + D*y^2 + Ax + By + C = 0
                    # where y is the integer coordinate.
                    # Check if p4 lies on the circle through p1, p2, p3.
                    # System: x_i^2 + D*y_i^2 + A*x_i + B*y_i + C = 0 for i=1,2,3
                    # Then check if x4^2 + D*y4^2 + A*x4 + B*y4 + C == 0
                    a1 = x1 * x1 + D * y1 * y1
                    a2 = x2 * x2 + D * y2 * y2
                    a3 = x3 * x3 + D * y3 * y3
                    # Solve for A, B, C from:
                    # A*x1 + B*y1 + C = -a1
                    # A*x2 + B*y2 + C = -a2
                    # A*x3 + B*y3 + C = -a3
                    # Subtract eq1 from eq2 and eq3:
                    # A*(x2-x1) + B*(y2-y1) = -(a2-a1)
                    # A*(x3-x1) + B*(y3-y1) = -(a3-a1)
                    dx2 = x2 - x1
                    dy2 = y2 - y1
                    dx3 = x3 - x1
                    dy3 = y3 - y1
                    rhs2 = -(a2 - a1)
                    rhs3 = -(a3 - a1)
                    det = dx2 * dy3 - dx3 * dy2
                    if det == 0:
                        # p1, p2, p3 collinear - can't form circle, skip
                        continue
                    # Cramer's rule
                    A_num = rhs2 * dy3 - rhs3 * dy2
                    B_num = dx2 * rhs3 - dx3 * rhs2
                    # C = -a1 - A*x1 - B*y1
                    # Check: A*x4 + B*y4 + C = -a4
                    # A*x4 + B*y4 - a1 - A*x1 - B*y1 = -a4
                    # A*(x4-x1) + B*(y4-y1) = a1 - a4
                    # (A_num/det)*(x4-x1) + (B_num/det)*(y4-y1) = a1 - a4
                    # A_num*(x4-x1) + B_num*(y4-y1) = det*(a1 - a4)
                    lhs = A_num * (x4 - x1) + B_num * (y4 - y1)
                    rhs = det * (a1 - (x4 * x4 + D * y4 * y4))
                    if lhs == rhs:
                        return False
    return True

def _all_dist_rational_int(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_rational_int(points[i], points[j], D):
                return False
    return True

def _is_valid_int(points, D):
    """Validate integer-coordinate point set."""
    if len(points) < 3:
        return False
    if not _all_dist_rational_int(points, D):
        return False
    if not _no_three_collinear_int(points):
        return False
    if not _no_four_concyclic_int(points, D):
        return False
    return True

def _find_candidates_around(existing, D, max_coord=500):
    """Find integer points at integer distance from all existing points."""
    n = len(existing)
    if n == 0:
        return []
    candidates = None
    for (ex, ey) in existing:
        valid_for_this = set()
        for dx in range(-max_coord, max_coord + 1):
            for dy in range(-max_coord, max_coord + 1):
                if dx == 0 and dy == 0:
                    continue
                d2 = dx * dx + D * dy * dy
                if _is_square(d2):
                    px = ex + dx
                    py = ey + dy
                    valid_for_this.add((px, py))
        if candidates is None:
            candidates = valid_for_this
        else:
            candidates = candidates & valid_for_this
        if candidates is not None and len(candidates) == 0:
            return []
    if candidates is None:
        return []
    existing_set = set(existing)
    candidates = candidates - existing_set
    return list(candidates)

def _try_extend(points, D, max_coord=500, deadline=None):
    """Try to extend a valid point set by adding one more point."""
    if deadline is not None and time.time() > deadline:
        return points
    candidates = _find_candidates_around(points, D, max_coord)
    random.shuffle(candidates)
    for cand in candidates:
        if deadline is not None and time.time() > deadline:
            break
        new_set = points + [cand]
        if _is_valid_int(new_set, D):
            return new_set
    return points

def _try_extend_multi(points, D, max_coord, deadline, max_attempts=5):
    """Try multiple extensions in sequence."""
    current = points[:]
    for _ in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        new_set = _try_extend(current, D, max_coord, deadline)
        if len(new_set) > len(current):
            current = new_set
        else:
            break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known 6-point integral set in Z^2 (D=1)
    # The classic 6-point set: (0,0), (56,0), (33,33), (0,33), (33,0), (56,33)
    # Actually let me use a known valid 6-point set
    # Known: (0,0), (56,0), (0,33), (40,45), (48,55), (33,33) - need to verify
    # Let me use the well-known 5-point set and try to extend
    known5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify known 5-point set
    if not _is_valid_int(known5, 1):
        # Try another known set
        known5 = [(0, 0), (56, 0), (0, 33), (40, 45), (48, 55)]
        if not _is_valid_int(known5, 1):
            known5 = [(0, 0), (3, 4), (0, 5)]
    
    best_points = known5[:]
    best_D = 1
    best_len = len(best_points)
    
    # Phase 1: Try to extend known set with increasing search radius
    current = best_points[:]
    for radius in [100, 200, 400, 600, 800, 1000]:
        if time.time() > deadline - 30:
            break
        new_set = _try_extend(current, 1, max_coord=radius, deadline=deadline - 30)
        if len(new_set) > len(current):
            current = new_set
            best_points = current[:]
            best_len = len(current)
            if best_len >= 8:
                break
        # Also try from the original known5 with this radius
        if len(current) == len(known5):
            alt = _try_extend(known5, 1, max_coord=radius, deadline=deadline - 30)
            if len(alt) > len(best_points):
                best_points = alt[:]
                best_len = len(best_points)
                current = best_points[:]
    
    # Phase 2: Try multiple known 5-point configurations
    known_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 0), (0, 33), (40, 45), (48, 55)],
        [(0, 0), (48, 0), (0, 33), (33, 44), (40, 45)],
        [(0, 0), (63, 0), (0, 33), (56, 33), (33, 63)],
        [(0, 0), (28, 0), (0, 21), (21, 28), (28, 21)],
    ]
    
    for ks in known_sets:
        if time.time() > deadline - 20:
            break
        if _is_valid_int(ks, 1):
            if len(ks) > best_len:
                best_points = ks[:]
                best_len = len(ks)
            # Try to extend
            extended = _try_extend(ks, 1, max_coord=300, deadline=deadline - 20)
            if len(extended) > best_len:
                best_points = extended[:]
                best_len = len(extended)
                if best_len >= 8:
                    break
    
    # Phase 3: Random search for larger sets
    if time.time() < deadline - 15:
        for attempt in range(30000):
            if time.time() > deadline - 15:
                break
            x = random.randint(-200, 200)
            y = random.randint(-200, 200)
            cand = best_points + [(x, y)]
            if _is_valid_int(cand, 1):
                best_points = cand[:]
                best_len = len(cand)
                if best_len >= 8:
                    break
    
    # Phase 4: Try D=2, D=3 with systematic search
    if time.time() < deadline - 10:
        for D_val in [2, 3, 5, 6]:
            if time.time() > deadline - 5:
                break
            # Find a small valid set for this D
            for attempt in range(5000):
                if time.time() > deadline - 5:
                    break
                n = random.randint(3, 5)
                pts = []
                for _ in range(n):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    pts.append((x, y))
                if _is_valid_int(pts, D_val):
                    if len(pts) > best_len:
                        best_points = pts[:]
                        best_D = D_val
                        best_len = len(pts)
                    # Try to extend
                    extended = _try_extend(pts, D_val, max_coord=200, deadline=deadline - 5)
                    if len(extended) > best_len:
                        best_points = extended[:]
                        best_D = D_val
                        best_len = len(extended)
                    if best_len >= 8:
                        break
            if best_len >= 8:
                break
    
    # Phase 5: Final aggressive extension attempt
    if time.time() < deadline - 3 and best_len < 8:
        for radius in [1000, 1500, 2000]:
            if time.time() > deadline - 3:
                break
            new_set = _try_extend(best_points, best_D, max_coord=radius, deadline=deadline - 3)
            if len(new_set) > best_len:
                best_points = new_set[:]
                best_len = len(best_points)
                if best_len >= 8:
                    break
    
    # Format result
    result_points = []
    for (x, y) in best_points:
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': result_points}
