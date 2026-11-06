import math
import time
import random
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    c1 = x1*x1 + y1*y1
    c2 = x2*x2 + y2*y2
    c3 = x3*x3 + y3*y3
    c4 = x4*x4 + y4*y4
    
    def det3(a,b,c,d,e,f,g,h,i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    
    D1 = det3(x2, y2, 1, x3, y3, 1, x4, y4, 1)
    D2 = det3(c2, y2, 1, c3, y3, 1, c4, y4, 1)
    D3 = det3(c2, x2, 1, c3, x3, 1, c4, x4, 1)
    D4 = det3(c2, x2, y2, c3, x3, y3, c4, x4, y4)
    
    return c1*D1 - x1*D2 + y1*D3 - 1*D4 == 0

def _valid_set(points):
    n = len(points)
    pts = set(points)
    if len(pts) != n:
        return False
        
    for i in range(n):
        for j in range(i+1, n):
            d2 = _dist_sq(points[i], points[j])
            if d2 == 0 or not _is_square(d2):
                return False
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _is_valid_extension(existing, new_p):
    """Check if new_p can be added to existing set."""
    if new_p in existing:
        return False
    
    # Check distances to all existing points
    for p in existing:
        d2 = _dist_sq(new_p, p)
        if d2 == 0 or not _is_square(d2):
            return False
    
    # Check collinearity with all pairs in existing
    n = len(existing)
    for i in range(n):
        for j in range(i+1, n):
            if _are_collinear(existing[i], existing[j], new_p):
                return False
    
    # Check concyclicity with all triples in existing
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_concyclic(existing[i], existing[j], existing[k], new_p):
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _valid_set(base_points):
        # Fallback to simpler set
        base_points = [(0, 0), (3, 4), (0, 5), (4, 3)]
    
    best_points = list(base_points)
    
    # Strategy: Use rational parameterization of circles through triples
    # For any 3 points, generate rational points on their circumcircle
    # and check if they're at integer distance from all existing points
    
    def _rational_circle_points(p1, p2, p3, max_den=200):
        """Generate rational points on circle through p1, p2, p3 using parameterization."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Find circle equation: x^2 + y^2 + Dx + Ey + F = 0
        # Using the 3x3 system
        # x1^2+y1^2 + D*x1 + E*y1 + F = 0
        # x2^2+y2^2 + D*x2 + E*y2 + F = 0
        # x3^2+y3^2 + D*x3 + E*y3 + F = 0
        
        c1 = x1*x1 + y1*y1
        c2 = x2*x2 + y2*y2
        c3 = x3*x3 + y3*y3
        
        # Solve for D, E, F
        # Subtract first from second: (x2-x1)D + (y2-y1)E = c1 - c2
        # Subtract first from third: (x3-x1)D + (y3-y1)E = c1 - c3
        
        a1 = x2 - x1
        b1 = y2 - y1
        c1_sub = c1 - c2
        
        a2 = x3 - x1
        b2 = y3 - y1
        c2_sub = c1 - c3
        
        det = a1 * b2 - a2 * b1
        if det == 0:
            return []  # Collinear
        
        D = (c1_sub * b2 - c2_sub * b1) / det
        E = (a1 * c2_sub - a2 * c1_sub) / det
        F = -(c1 + D * x1 + E * y1)
        
        # Center: (-D/2, -E/2), Radius^2 = (D^2 + E^2)/4 - F
        cx = -D / 2
        cy = -E / 2
        r2 = (D*D + E*E) / 4 - F
        
        if r2 <= 0:
            return []
        
        # Parameterize circle: (x - cx) = r*cos(t), (y - cy) = r*sin(t)
        # Using rational parameterization: cos(t) = (1-t^2)/(1+t^2), sin(t) = 2t/(1+t^2)
        # So x = cx + r*(1-t^2)/(1+t^2), y = cy + r*2t/(1+t^2)
        
        # But we need integer/rational points. Let's use a different approach:
        # If we have one rational point on the circle, we can generate more.
        # p1 is on the circle. Use chord parameterization from p1.
        
        # Line through p1 with slope m: y - y1 = m(x - x1)
        # Intersect with circle.
        
        candidates = []
        for m_num in range(-max_den, max_den + 1):
            for m_den in range(1, max_den + 1):
                if m_num == 0 and m_den == 1:
                    continue  # Skip horizontal, handled separately
                
                # m = m_num / m_den
                # y = y1 + (m_num/m_den)(x - x1)
                # Substitute into circle equation...
                
                # This gets complex. Let's just do a simple grid search instead.
                pass
        
        return candidates
    
    # Simpler approach: systematic grid search with early termination
    # Search for 6th point
    found_6 = False
    search_range = 150
    
    for y in range(-search_range, search_range + 1):
        if time.time() > deadline:
            break
        for x in range(-search_range, search_range + 1):
            if time.time() > deadline:
                break
            
            new_p = (x, y)
            if _is_valid_extension(base_points, new_p):
                best_points = base_points + [new_p]
                found_6 = True
                break
        if found_6:
            break
    
    # Try to find 7th point
    if found_6:
        found_7 = False
        search_range2 = 120
        
        for y in range(-search_range2, search_range2 + 1):
            if time.time() > deadline:
                break
            for x in range(-search_range2, search_range2 + 1):
                if time.time() > deadline:
                    break
                
                new_p = (x, y)
                if _is_valid_extension(best_points, new_p):
                    best_points = best_points + [new_p]
                    found_7 = True
                    break
            if found_7:
                break
        
        # Random search for 7th point if not found
        if not found_7:
            for _ in range(50000):
                if time.time() > deadline:
                    break
                x = random.randint(-200, 200)
                y = random.randint(-200, 200)
                new_p = (x, y)
                if _is_valid_extension(best_points, new_p):
                    best_points = best_points + [new_p]
                    found_7 = True
                    break
    
    # Try to find 8th point
    if found_6:
        found_8 = False
        search_range3 = 100
        
        for y in range(-search_range3, search_range3 + 1):
            if time.time() > deadline:
                break
            for x in range(-search_range3, search_range3 + 1):
                if time.time() > deadline:
                    break
                
                new_p = (x, y)
                if _is_valid_extension(best_points, new_p):
                    best_points = best_points + [new_p]
                    found_8 = True
                    break
            if found_8:
                break
        
        if not found_8:
            for _ in range(30000):
                if time.time() > deadline:
                    break
                x = random.randint(-250, 250)
                y = random.randint(-250, 250)
                new_p = (x, y)
                if _is_valid_extension(best_points, new_p):
                    best_points = best_points + [new_p]
                    found_8 = True
                    break
    
    # Final verification
    if not _valid_set(best_points):
        best_points = base_points
        if not _valid_set(best_points):
            best_points = base_points[:4]
    
    # Format output
    out_points = []
    for x, y in best_points:
        out_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': out_points}

if __name__ == '__main__':
    res = construct()
    print(res['D'], len(res['points']))
