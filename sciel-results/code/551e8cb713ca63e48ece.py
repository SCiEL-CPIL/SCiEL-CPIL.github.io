import time
import math
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Known 5-point integral set (D=1)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_int_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        if d2 < 0:
            return False
        r = math.isqrt(d2)
        return r*r == d2
    
    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])
    
    def det4(m):
        """Compute 4x4 determinant exactly using integers."""
        a,b,c,d = m[0]
        e,f,g,h = m[1]
        i,j,k,l = m[2]
        m2,n,o,p = m[3]
        return (
            a*(f*(k*p-l*o) - g*(j*p-l*n) + h*(j*o-k*n)) -
            b*(e*(k*p-l*o) - g*(i*p-l*m2) + h*(i*o-k*m2)) +
            c*(e*(j*p-l*n) - f*(i*p-l*m2) + h*(i*n-j*m2)) -
            d*(e*(j*o-k*n) - f*(i*o-k*m2) + g*(i*n-j*m2))
        )
    
    def is_concyclic(p1, p2, p3, p4):
        """Check if 4 points are concyclic using exact integer arithmetic."""
        m = [
            [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
            [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
        ]
        return det4(m) == 0
    
    def can_add_point(points, p_new):
        """Check if p_new can be added to points maintaining all constraints."""
        # Check distinct
        for p in points:
            if p == p_new:
                return False
        # Check integer distances
        for p in points:
            if not is_int_dist(p, p_new):
                return False
        # Check no three collinear
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if is_collinear(points[i], points[j], p_new):
                    return False
        # Check no four concyclic
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if is_concyclic(points[i], points[j], points[k], p_new):
                        return False
        return True
    
    def search_for_point(points, x_range, y_range, time_limit):
        """Search for a point that can be added to the set."""
        for x in range(x_range[0], x_range[1] + 1):
            if time.time() > time_limit:
                return None
            for y in range(y_range[0], y_range[1] + 1):
                p_new = (x, y)
                if can_add_point(points, p_new):
                    return p_new
        return None
    
    best_points = seed_points[:]
    best_size = 5
    
    # Try to find 6th point with expanding search radius
    search_radii = [100, 150, 200, 300, 400, 500]
    found_6 = False
    
    for r in search_radii:
        if time.time() > deadline - 30:
            break
        p6 = search_for_point(best_points, (-r, r), (-r, r), deadline - 30)
        if p6 is not None:
            best_points.append(p6)
            best_size = 6
            found_6 = True
            break
    
    # If we have 6 points, try to find a 7th
    if found_6:
        for r in [100, 200, 300, 500]:
            if time.time() > deadline - 10:
                break
            p7 = search_for_point(best_points, (-r, r), (-r, r), deadline - 10)
            if p7 is not None:
                best_points.append(p7)
                best_size = 7
                break
    
    # Format output
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
