"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_squarefree(n):
    if n == 1:
        return True
    r = isqrt(n)
    for i in range(2, r + 1):
        if n % (i * i) == 0:
            return False
    return True

def _check_valid(points, D, max_numer=10**40):
    n = len(points)
    if n < 3 or n > 12:
        return False
    if D < 1 or D > 10**12:
        return False
    if not _is_squarefree(D):
        return False
    
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False

    # Check pairwise distances rational
    for i in range(n):
        for j in range(i+1, n):
            dx_num = points[i][0][0]*points[j][0][1] - points[j][0][0]*points[i][0][1]
            dx_den = points[i][0][1]*points[j][0][1]
            dy_num = points[i][1][0]*points[j][1][1] - points[j][1][0]*points[i][1][1]
            dy_den = points[i][1][1]*points[j][1][1]
            
            # dist^2 = (dx)^2 + D*(dy)^2
            num = dx_num*dx_num * dy_den*dy_den + D * dy_num*dy_num * dx_den*dx_den
            den = dx_den*dx_den * dy_den*dy_den
            if den == 0:
                return False
            if not _is_square(num * den):
                return False

    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1n, x1d = points[i][0]
                y1n, y1d = points[i][1]
                x2n, x2d = points[j][0]
                y2n, y2d = points[j][1]
                x3n, x3d = points[k][0]
                y3n, y3d = points[k][1]
                
                def sub_frac(n1, d1, n2, d2):
                    return n1*d2 - n2*d1, d1*d2
                
                dx12_n, dx12_d = sub_frac(x2n, x2d, x1n, x1d)
                dy12_n, dy12_d = sub_frac(y2n, y2d, y1n, y1d)
                dx13_n, dx13_d = sub_frac(x3n, x3d, x1n, x1d)
                dy13_n, dy13_d = sub_frac(y3n, y3d, y1n, y1d)
                
                num = dx12_n*dy13_n*dy12_d*dx13_d - dy12_n*dx13_n*dx12_d*dy13_d
                if num != 0:
                    continue
                return False

    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    def get_row(pt):
                        x = Fraction(pt[0][0], pt[0][1])
                        y = Fraction(pt[1][0], pt[1][1])
                        return [x*x + D*y*y, x, y, 1]
                    
                    M = [
                        get_row(points[i]),
                        get_row(points[j]),
                        get_row(points[k]),
                        get_row(points[l])
                    ]
                    
                    def det3(m):
                        return (m[0][0]*(m[1][1]*m[2][2]-m[2][1]*m[1][2]) -
                                m[0][1]*(m[1][0]*m[2][2]-m[2][0]*m[1][2]) +
                                m[0][2]*(m[1][0]*m[2][1]-m[2][0]*m[1][1]))
                    
                    det = 0
                    for c in range(4):
                        minor = [row[:c] + row[c+1:] for row in M[1:]]
                        det += ((-1)**c) * M[0][c] * det3(minor)
                    
                    if det == 0:
                        return False

    return True

def _precompute_base_info(base_points):
    """Precompute lines and circles for a set of base points."""
    n = len(base_points)
    bxs = [p[0] for p in base_points]
    bys = [p[1] for p in base_points]
    base_set = set(base_points)
    
    # Precompute lines: dy*x - dx*y + c = 0
    lines = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = base_points[i]
            x2, y2 = base_points[j]
            dx = x2 - x1
            dy = y2 - y1
            c = dx*y1 - dy*x1
            lines.append((dy, -dx, c))
    
    # Precompute circles: x^2 + y^2 + A*x + B*y + C = 0
    circles = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = base_points[i]
                x2, y2 = base_points[j]
                x3, y3 = base_points[k]
                
                dx21 = x2 - x1
                dy21 = y2 - y1
                dx31 = x3 - x1
                dy31 = y3 - y1
                
                rhs2 = -((x2*x2 - x1*x1) + (y2*y2 - y1*y1))
                rhs3 = -((x3*x3 - x1*x1) + (y3*y3 - y1*y1))
                
                det = dx21*dy31 - dx31*dy21
                if det == 0:
                    continue
                
                A = Fraction(rhs2*dy31 - rhs3*dy21, det)
                B = Fraction(dx21*rhs3 - dx31*rhs2, det)
                C = -Fraction(x1*x1 + y1*y1) - A*x1 - B*y1
                
                circles.append((A, B, C))
    
    return bxs, bys, base_set, lines, circles

def _search_for_point(base_points, search_range, deadline):
    """Search for a new point that extends base_points to a valid set."""
    bxs, bys, base_set, lines, circles = _precompute_base_info(base_points)
    n_base = len(base_points)
    
    count = 0
    for x in range(-search_range, search_range + 1):
        x2 = x*x
        for y in range(-search_range, search_range + 1):
            count += 1
            if count % 500 == 0:
                if time.time() > deadline:
                    return None
            
            if (x, y) in base_set:
                continue
            
            y2 = y*y
            
            # Distance check
            valid_dist = True
            for i in range(n_base):
                dx = x - bxs[i]
                dy = y - bys[i]
                d2 = dx*dx + dy*dy
                if not _is_square(d2):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Collinearity check
            valid_coll = True
            for dy, dx, c in lines:
                if dy*x - dx*y - c == 0:
                    valid_coll = False
                    break
            
            if not valid_coll:
                continue
            
            # Concyclicity check
            valid_conc = True
            for A, B, C in circles:
                val = x2 + y2 + A*x + B*y + C
                if val == 0:
                    valid_conc = False
                    break
            
            if not valid_conc:
                continue
            
            # Candidate found! Verify with full check
            new_point = [[x, 1], [y, 1]]
            test_points = [ [[p[0], 1], [p[1], 1]] for p in base_points ] + [new_point]
            if _check_valid(test_points, 1):
                return (x, y)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with the known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = base_points
    D = 1
    
    # Try to find a 6th point with expanding search ranges
    search_ranges = [50, 100, 150, 200, 300, 400]
    found_6th = None
    
    for sr in search_ranges:
        if time.time() > deadline - 10:
            break
        result = _search_for_point(base_points, sr, deadline - 5)
        if result is not None:
            found_6th = result
            break
    
    if found_6th:
        best_points = base_points + [found_6th]
        
        # Try to find a 7th point
        if time.time() < deadline - 20:
            search_ranges_7 = [30, 50, 80, 120]
            found_7th = None
            for sr in search_ranges_7:
                if time.time() > deadline - 10:
                    break
                result = _search_for_point(best_points, sr, deadline - 5)
                if result is not None:
                    found_7th = result
                    break
            
            if found_7th:
                best_points = best_points + [found_7th]
                
                # Try to find an 8th point if we have time
                if time.time() < deadline - 15:
                    search_ranges_8 = [20, 40, 60]
                    found_8th = None
                    for sr in search_ranges_8:
                        if time.time() > deadline - 5:
                            break
                        result = _search_for_point(best_points, sr, deadline - 3)
                        if result is not None:
                            found_8th = result
                            break
                    
                    if found_8th:
                        best_points = best_points + [found_8th]
    
    # Convert best_points to the required format
    formatted_points = []
    for p in best_points:
        x, y = p
        formatted_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': formatted_points}

# EVOLVE-BLOCK-END
