"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _no_four_concyclic_fast(points, D):
    """Check no four points are concyclic using determinant test.
    Points are (x, y_coeff) where actual y = y_coeff * sqrt(D).
    Uses the circle equation: x^2 + y^2 + Ax + By + C = 0
    For point (x_i, y_i*sqrt(D)): x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
    This requires B=0 for rationality unless we work in the field extension.
    Better approach: use the determinant condition directly.
    """
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1c = points[i]
                    x2, y2c = points[j]
                    x3, y3c = points[k]
                    x4, y4c = points[l]
                    
                    # Circle through 3 points: x^2 + D*y^2 + A*x + B*y + C = 0
                    # For point (x_i, y_i): x_i^2 + D*y_i^2 + A*x_i + B*y_i + C = 0
                    # Solve for A, B, C from first 3 points, check 4th
                    # Using Cramer's rule / determinant
                    # |x1  y1  x1^2+D*y1^2  1|
                    # |x2  y2  x2^2+D*y2^2  1| = 0
                    # |x3  y3  x3^2+D*y3^2  1|
                    # |x4  y4  x4^2+D*y4^2  1|
                    # Wait, the equation is x^2 + D*y^2 + A*x + B*y + C = 0
                    # So: A*x_i + B*y_i + C = -(x_i^2 + D*y_i^2)
                    # Matrix: [x_i, y_i, 1] * [A, B, C]^T = -(x_i^2 + D*y_i^2)
                    
                    def det3(m):
                        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                              - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                              + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
                    
                    # Check if 4 points are concyclic
                    # The determinant of the 4x4 matrix should be 0
                    # |x1  y1  x1^2+D*y1^2  1|
                    # |x2  y2  x2^2+D*y2^2  1|
                    # |x3  y3  x3^2+D*y3^2  1|
                    # |x4  y4  x4^2+D*y4^2  1|
                    
                    # Expand along last column (all 1s):
                    # = M11 - M21 + M31 - M41 where M_i1 is the 3x3 minor removing row i, col 4
                    # Actually, let's use the standard formula:
                    # det = sum_{i=1}^{4} (-1)^{i+4} * 1 * det(minor_{i,4})
                    
                    def minor_4x4_remove_row(rows, r):
                        return [list(row) for idx, row in enumerate(rows) if idx != r]
                    
                    rows = [
                        [x1, y1c, x1*x1 + D*y1c*y1c, 1],
                        [x2, y2c, x2*x2 + D*y2c*y2c, 1],
                        [x3, y3c, x3*x3 + D*y3c*y3c, 1],
                        [x4, y4c, x4*x4 + D*y4c*y4c, 1]
                    ]
                    
                    total = 0
                    for r in range(4):
                        minor = minor_4x4_remove_row(rows, r)
                        # Remove last column
                        minor3 = [[row[0], row[1], row[2]] for row in minor]
                        d = det3(minor3)
                        total += (-1)**(r+3) * d  # (-1)^(r+4) * 1
                    
                    if total == 0:
                        return False
    return True

def _precompute_int_dist_offsets(max_coord):
    """Precompute all (dx, dy) with dx^2 + dy^2 a perfect square, |dx|,|dy| <= max_coord."""
    offsets = {}
    for dx in range(-max_coord, max_coord + 1):
        for dy in range(-max_coord, max_coord + 1):
            d2 = dx*dx + dy*dy
            if d2 > 0 and _is_square(d2):
                offsets[(dx, dy)] = d2
    return offsets

def _check_collinear(points, x, y):
    """Check if point (x, y) is collinear with any pair of points."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # (x2-x1)*(y-y1) - (y2-y1)*(x-x1) == 0
            cp = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if cp == 0:
                return True
    return False

def _try_expand(points, D, int_offsets, deadline, max_coord):
    """Try to add one point to the given set. Returns (new_points, success)."""
    n = len(points)
    if n < 2:
        return points, False
    
    # Try different anchor pairs
    for anchor_i in range(min(4, n)):
        for anchor_j in range(anchor_i+1, min(5, n)):
            if time.time() > deadline:
                return points, False
            
            a0, b0 = points[anchor_i]
            a1, b1 = points[anchor_j]
            da = a0 - a1
            db = b0 - b1
            
            for dx0, dy0 in int_offsets:
                if time.time() > deadline:
                    return points, False
                
                dx1 = dx0 + da
                dy1 = dy0 + db
                
                if (dx1, dy1) not in int_offsets:
                    continue
                
                x = a0 + dx0
                y = b0 + dy0
                
                # Check against all other points
                all_ok = True
                for i in range(n):
                    if i == anchor_i or i == anchor_j:
                        continue
                    ai, bi = points[i]
                    dxi = x - ai
                    dyi = y - bi
                    if (dxi, dyi) not in int_offsets:
                        all_ok = False
                        break
                if not all_ok:
                    continue
                
                # Check not collinear with any pair
                if _check_collinear(points, x, y):
                    continue
                
                # Check no four concyclic
                candidate = points + [(Fraction(x), Fraction(y))]
                if not _no_four_concyclic_fast(candidate, D):
                    continue
                
                return candidate, True
    return points, False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Known 5-point integral point set (Erdos-Moser type)
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    base_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    
    best = base_frac
    best_count = 5
    
    # Precompute integer-distance offsets for a reasonable range
    max_coord = 5000
    int_offsets = _precompute_int_dist_offsets(max_coord)
    
    # Try to expand the base set
    for _ in range(10):  # Try up to 10 expansions
        if time.time() > deadline:
            break
        new_points, success = _try_expand(best, D, int_offsets, deadline, max_coord)
        if not success:
            break
        best = new_points
        best_count += 1
    
    # If we couldn't expand with D=1, try other known configurations
    if best_count < 6:
        # Try a different base configuration
        alt_bases = [
            # Another known 5-point set
            [(0, 0), (3, 4), (0, 5), (-3, 4), (0, 3)],
            # Symmetric configuration
            [(0, 0), (5, 0), (0, 5), (-5, 0), (0, -5)],
            # Different arrangement
            [(0, 0), (12, 0), (0, 9), (-12, 0), (0, 7)],
        ]
        
        for alt_pts in alt_bases:
            if time.time() > deadline:
                break
            
            # Verify this is a valid base
            valid = True
            for i in range(len(alt_pts)):
                for j in range(i+1, len(alt_pts)):
                    dx = alt_pts[i][0] - alt_pts[j][0]
                    dy = alt_pts[i][1] - alt_pts[j][1]
                    d2 = dx*dx + dy*dy
                    if not _is_square(d2):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            alt_frac = [(Fraction(x), Fraction(y)) for x, y in alt_pts]
            
            # Check no three collinear
            collinear = False
            for i in range(len(alt_frac)):
                for j in range(i+1, len(alt_frac)):
                    for k in range(j+1, len(alt_frac)):
                        x1, y1 = alt_frac[i]
                        x2, y2 = alt_frac[j]
                        x3, y3 = alt_frac[k]
                        if (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    break
            
            if collinear:
                continue
            
            # Check no four concyclic
            if not _no_four_concyclic_fast(alt_frac, 1):
                continue
            
            # Try to expand this alternative
            cur = alt_frac
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_pts, success = _try_expand(cur, 1, int_offsets, deadline, max_coord)
                if not success:
                    break
                cur = new_pts
            
            if len(cur) > best_count:
                best = cur
                best_count = len(cur)
    
    # Try with D=2 or other small squarefree values
    if best_count < 6 and time.time() < deadline - 30:
        for D_try in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
            if time.time() > deadline - 10:
                break
            
            # For D>1, we need rational distances in the field Q(sqrt(D))
            # Distance squared = (x1-x2)^2 + D*(y1-y2)^2 must be a square in Q
            # This is harder to search directly, so use a simpler approach:
            # Try small integer coordinates and check
            
            # For now, skip D>1 as it's much harder to search efficiently
            pass
    
    # Format output
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END
