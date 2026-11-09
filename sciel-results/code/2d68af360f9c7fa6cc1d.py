"""Search for integral point sets: points with rational coordinates and rational pairwise distances,
no three collinear, no four concyclic."""
import math
import time
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    """Check if n is a perfect square (n must be non-negative integer)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _check_pairwise_distances(points):
    """Check all pairwise distances are rational (squared distance is perfect square for D=1).
    Returns True if all distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    return True

def _check_no_three_collinear(points):
    """Check no three points are collinear using exact integer arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    """Check no four points are concyclic using exact integer arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    dx, dy = points[l]
                    a1 = ax * ax + ay * ay
                    b1 = bx * bx + by * by
                    c1 = cx * cx + cy * cy
                    d1 = dx * dx + dy * dy
                    
                    def det3(p1, p2, p3, q1, q2, q3, r1, r2, r3):
                        return (p1*(q2*r3 - q3*r2) - q1*(p2*r3 - p3*r2) + r1*(p2*q3 - p3*q2))
                    
                    M14 = det3(bx, by, b1, cx, cy, c1, dx, dy, d1)
                    M24 = det3(ax, ay, a1, cx, cy, c1, dx, dy, d1)
                    M34 = det3(ax, ay, a1, bx, by, b1, dx, dy, d1)
                    M44 = det3(ax, ay, a1, bx, by, b1, cx, cy, c1)
                    
                    det_val = M14 - M24 + M34 - M44
                    if det_val == 0:
                        return False
    return True

def _check_valid(points):
    """Full validity check: pairwise rational distances, no 3 collinear, no 4 concyclic."""
    if not _check_pairwise_distances(points):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points):
        return False
    return True

def _generate_heronian_points(max_radius=200):
    """Generate points with integer distance to origin using Pythagorean triples."""
    points = set()
    points.add((0, 0))
    # Generate all primitive and non-primitive Pythagorean triples
    for a in range(1, max_radius + 1):
        for b in range(1, max_radius + 1):
            c2 = a * a + b * b
            c = math.isqrt(c2)
            if c * c == c2:
                # (a, b) has integer distance c to origin
                for sign_a in [1, -1]:
                    for sign_b in [1, -1]:
                        points.add((sign_a * a, sign_b * b))
                        points.add((sign_a * b, sign_b * a))
    return list(points)

def _find_candidates_fast(existing_points, candidate_pool, deadline):
    """Find candidates from pool that have integer distance to all existing points
    and don't create collinearity or concyclicity."""
    n = len(existing_points)
    valid_candidates = []
    
    for (cx, cy) in candidate_pool:
        if time.time() >= deadline:
            break
        
        # Check integer distance to all existing
        ok = True
        for (ex, ey) in existing_points:
            dx = cx - ex
            dy = cy - ey
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                ok = False
                break
        if not ok:
            continue
        
        # Check no three collinear
        collinear = False
        for i in range(n):
            for j in range(i + 1, n):
                ax, ay = existing_points[i]
                bx, by = existing_points[j]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if cross == 0:
                    collinear = True
                    break
            if collinear:
                break
        if collinear:
            continue
        
        # Check no four concyclic
        concyclic = False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    ax, ay = existing_points[i]
                    bx, by = existing_points[j]
                    ex, ey = existing_points[k]
                    fx, fy = cx, cy
                    a1 = ax * ax + ay * ay
                    b1 = bx * bx + by * by
                    e1 = ex * ex + ey * ey
                    f1 = fx * fx + fy * fy
                    
                    def det3(p1, p2, p3, q1, q2, q3, r1, r2, r3):
                        return (p1*(q2*r3 - q3*r2) - q1*(p2*r3 - p3*r2) + r1*(p2*q3 - p3*q2))
                    
                    M14 = det3(bx, by, b1, ex, ey, e1, fx, fy, f1)
                    M24 = det3(ax, ay, a1, ex, ey, e1, fx, fy, f1)
                    M34 = det3(ax, ay, a1, bx, by, b1, fx, fy, f1)
                    M44 = det3(ax, ay, a1, bx, by, b1, ex, ey, e1)
                    
                    det_val = M14 - M24 + M34 - M44
                    if det_val == 0:
                        concyclic = True
                        break
                if concyclic:
                    break
            if concyclic:
                break
        if concyclic:
            continue
        
        valid_candidates.append((cx, cy))
    
    return valid_candidates

def _search_backtrack(start_points, candidate_pool, deadline, max_points=8):
    """Backtracking search to find largest valid set."""
    best = list(start_points)
    best_size = len(best)
    
    def backtrack(current, idx):
        nonlocal best, best_size
        if time.time() >= deadline:
            return
        
        if len(current) > best_size:
            best = list(current)
            best_size = len(best)
        
        if len(current) >= max_points:
            return
        
        # Find valid candidates from pool starting at idx
        for i in range(idx, len(candidate_pool)):
            if time.time() >= deadline:
                return
            
            cand = candidate_pool[i]
            
            # Quick check: integer distance to all current points
            ok = True
            for (ex, ey) in current:
                dx = cand[0] - ex
                dy = cand[1] - ey
                d2 = dx * dx + dy * dy
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            
            # Check collinearity
            collinear = False
            for j in range(len(current)):
                for k in range(j + 1, len(current)):
                    ax, ay = current[j]
                    bx, by = current[k]
                    cross = (bx - ax) * (cand[1] - ay) - (by - ay) * (cand[0] - ax)
                    if cross == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check concyclicity
            concyclic = False
            for j in range(len(current)):
                for k in range(j + 1, len(current)):
                    for l in range(k + 1, len(current)):
                        ax, ay = current[j]
                        bx, by = current[k]
                        ex, ey = current[l]
                        fx, fy = cand
                        a1 = ax * ax + ay * ay
                        b1 = bx * bx + by * by
                        e1 = ex * ex + ey * ey
                        f1 = fx * fx + fy * fy
                        
                        def det3(p1, p2, p3, q1, q2, q3, r1, r2, r3):
                            return (p1*(q2*r3 - q3*r2) - q1*(p2*r3 - p3*r2) + r1*(p2*q3 - p3*q2))
                        
                        M14 = det3(bx, by, b1, ex, ey, e1, fx, fy, f1)
                        M24 = det3(ax, ay, a1, ex, ey, e1, fx, fy, f1)
                        M34 = det3(ax, ay, a1, bx, by, b1, fx, fy, f1)
                        M44 = det3(ax, ay, a1, bx, by, b1, ex, ey, e1)
                        
                        det_val = M14 - M24 + M34 - M44
                        if det_val == 0:
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            if concyclic:
                continue
            
            # Valid candidate - recurse
            current.append(cand)
            backtrack(current, i + 1)
            current.pop()
    
    backtrack(list(start_points), 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Generate candidate pool of Heronian points
    candidate_pool = _generate_heronian_points(300)
    
    # Remove origin from pool if it's in our start set
    # Start with known valid sets
    start_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (5, 0), (0, 12), (9, 12)],
        [(0, 0), (7, 24), (15, 8), (0, 24), (8, 24)],
        [(0, 0), (5, 12), (12, 5), (0, 13), (13, 0)],
    ]
    
    best = []
    best_size = 0
    
    for start in start_sets:
        if time.time() >= deadline - 10:
            break
        
        # Verify start is valid
        if not _check_valid(start):
            continue
        
        if len(start) > best_size:
            best = list(start)
            best_size = len(start)
        
        # Try to extend via backtracking
        # Filter pool to points not in start
        pool_filtered = [p for p in candidate_pool if p not in start]
        
        result = _search_backtrack(start, pool_filtered, deadline, max_points=8)
        if len(result) > best_size:
            best = result
            best_size = len(result)
        
        if best_size >= 8:
            break
    
    # If we haven't found 8 points, try a more targeted search
    if best_size < 7 and time.time() < deadline - 20:
        # Try incremental extension with different radii
        for max_r in [100, 150, 200]:
            if time.time() >= deadline - 5:
                break
            pool = _generate_heronian_points(max_r)
            for start in start_sets[:2]:
                if not _check_valid(start):
                    continue
                pool_filtered = [p for p in pool if p not in start]
                result = _search_backtrack(start, pool_filtered, deadline, max_points=8)
                if len(result) > best_size:
                    best = result
                    best_size = len(result)
                if best_size >= 8:
                    break
            if best_size >= 8:
                break
    
    # Convert to required format
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': 1, 'points': points_out}

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result['points'])} points")
    for p in result['points']:
        print(f"  ({p[0][0]}/{p[0][1]}, {p[1][0]}/{p[1][1]})")
