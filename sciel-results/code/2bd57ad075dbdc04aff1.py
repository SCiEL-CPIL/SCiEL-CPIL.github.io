import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def time_left():
        return deadline - (time.time() - t0)

    # Known valid 5-point integral set (D=1)
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    def is_square(n):
        if n < 0:
            return False, 0
        r = math.isqrt(n)
        if r * r == n:
            return True, r
        return False, 0

    def check_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                        return True
        return False

    def det4(m):
        return (m[0][0] * (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]))
               - m[0][1] * (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]))
               + m[0][2] * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
               - m[0][3] * (m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]) - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]) + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])))

    def check_concyclic(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        a, b = pts[i]
                        c, d = pts[j]
                        e, f = pts[k]
                        g, h = pts[l]
                        m = [
                            [a * a + b * b, a, b, 1],
                            [c * c + d * d, c, d, 1],
                            [e * e + f * f, e, f, 1],
                            [g * g + h * h, g, h, 1]
                        ]
                        if det4(m) == 0:
                            return True
        return False

    def validate_set(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d2 = dx * dx + dy * dy
                ok, _ = is_square(d2)
                if not ok:
                    return False
        if check_collinear(pts):
            return False
        if check_concyclic(pts):
            return False
        return True

    # Verify base set
    if not validate_set(base_points):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    best = list(base_points)

    # Strategy 1: Brute force search for 6th point
    found_6 = False
    search_range = 200
    for x in range(-search_range, search_range + 1):
        if time_left() < 15:
            break
        for y in range(-search_range, search_range + 1):
            new_pt = (x, y)
            if new_pt in base_points:
                continue
            valid = True
            for p in base_points:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx * dx + dy * dy
                ok, _ = is_square(d2)
                if not ok:
                    valid = False
                    break
            if not valid:
                continue
            candidate = base_points + [new_pt]
            if validate_set(candidate):
                best = candidate
                found_6 = True
                break
        if found_6 or time_left() < 15:
            break

    if len(best) == 6 and time_left() > 20:
        # Try to find a 7th point
        found_7 = False
        search_range2 = 250
        for x in range(-search_range2, search_range2 + 1):
            if time_left() < 10:
                break
            for y in range(-search_range2, search_range2 + 1):
                new_pt = (x, y)
                if new_pt in best:
                    continue
                valid = True
                for p in best:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx * dx + dy * dy
                    ok, _ = is_square(d2)
                    if not ok:
                        valid = False
                        break
                if not valid:
                    continue
                candidate = best + [new_pt]
                if validate_set(candidate):
                    best = candidate
                    found_7 = True
                    break
            if found_7 or time_left() < 10:
                break

    # Strategy 2: Search with D > 1 for potentially larger sets
    # For D=2, points are (x, y*sqrt(2)) with rational x, y
    # Distance squared = (x1-x2)^2 + D*(y1-y2)^2
    # For this to be rational, we need specific conditions
    
    # Let's try a systematic search for small D values
    if time_left() > 30:
        for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
            if time_left() < 20:
                break
            # For D=2, try to find 6+ points
            # Points: (a, b*sqrt(D)) where a, b are rational
            # Distance^2 between (a1, b1*sqrt(D)) and (a2, b2*sqrt(D)) = (a1-a2)^2 + D*(b1-b2)^2
            # For this to be rational, (a1-a2)^2 + D*(b1-b2)^2 must be rational
            # If a, b are integers, this is always rational
            
            # Try integer points for small D
            # Actually, let's try a different approach: search for rational points
            # that give rational distances
            
            # For now, let's just extend the brute force with a slightly larger range
            # and check if we can find more points
            
            pass

    # Strategy 3: Try to find 6th point with a smarter search
    # Focus on points that are at integer distance from all 5 base points
    if len(best) == 5 and time_left() > 25:
        # The 5 base points are:
        # P0=(0,0), P1=(-56,33), P2=(-16,63), P3=(0,33), P4=(16,63)
        # A 6th point (x,y) must satisfy:
        # x^2 + y^2 = d0^2
        # (x+56)^2 + (y-33)^2 = d1^2
        # (x+16)^2 + (y-63)^2 = d2^2
        # x^2 + (y-33)^2 = d3^2
        # (x-16)^2 + (y-63)^2 = d4^2
        # for some integers d0, d1, d2, d3, d4
        
        # From P0 and P3: x^2 + y^2 - (x^2 + (y-33)^2) = d0^2 - d3^2
        # y^2 - (y-33)^2 = d0^2 - d3^2
        # 66y - 1089 = d0^2 - d3^2
        # y = (d0^2 - d3^2 + 1089) / 66
        
        # From P2 and P4: (x+16)^2 + (y-63)^2 - ((x-16)^2 + (y-63)^2) = d2^2 - d4^2
        # (x+16)^2 - (x-16)^2 = d2^2 - d4^2
        # 64x = d2^2 - d4^2
        # x = (d2^2 - d4^2) / 64
        
        # So x and y are determined by (d2^2 - d4^2) and (d0^2 - d3^2)
        # We can enumerate small d2, d4, d0, d3 and check
        
        found_6b = False
        max_d = 300
        for d2 in range(1, max_d + 1):
            if time_left() < 10:
                break
            for d4 in range(1, max_d + 1):
                x_num = d2 * d2 - d4 * d4
                if x_num % 64 != 0:
                    continue
                x = x_num // 64
                for d0 in range(1, max_d + 1):
                    if time_left() < 5:
                        break
                    for d3 in range(1, max_d + 1):
                        y_num = d0 * d0 - d3 * d3 + 1089
                        if y_num % 66 != 0:
                            continue
                        y = y_num // 66
                        new_pt = (x, y)
                        if new_pt in best:
                            continue
                        # Check distances to all 5 points
                        valid = True
                        for p in best:
                            dx = x - p[0]
                            dy = y - p[1]
                            d2_val = dx * dx + dy * dy
                            ok, _ = is_square(d2_val)
                            if not ok:
                                valid = False
                                break
                        if not valid:
                            continue
                        candidate = best + [new_pt]
                        if validate_set(candidate):
                            best = candidate
                            found_6b = True
                            break
                    if found_6b:
                        break
                if found_6b:
                    break
            if found_6b:
                break

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
