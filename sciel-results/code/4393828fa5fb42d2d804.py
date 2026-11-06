import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    # The seed solution: 5 integer points with integer pairwise distances
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Let's verify and try to extend with more points.
    
    # First, let's establish the base 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point that maintains all constraints
    # Strategy: search for integer points (x, y) such that:
    # 1. Distance to each existing point is rational (integer for integer points)
    # 2. No three collinear
    # 3. No four concyclic
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    def check_constraints(points):
        """Check no three collinear and no four concyclic"""
        n = len(points)
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0:
                        return False
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        # Four points are concyclic iff the determinant is zero
                        # Using the circle equation: x^2+y^2+Dx+Ey+F=0
                        # For 4 points to be concyclic, the 4x4 system must be dependent
                        # Use the condition: (x1^2+y1^2)(x2-x3)(x2-x4) + ... = 0
                        # Simpler: compute circumcircle of first 3, check if 4th is on it
                        # Circumcenter of (x1,y1),(x2,y2),(x3,y3):
                        d = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                        if d == 0:
                            continue  # collinear, already checked
                        ux = ((x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2)) // d
                        uy = ((x1*x1+y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3+y3*y3)*(x2-x1)) // d
                        # Check if (x4,y4) is on this circle
                        r2 = (x1-ux)*(x1-ux) + (y1-uy)*(y1-uy)
                        if (x4-ux)*(x4-ux) + (y4-uy)*(y4-uy) == r2:
                            return False
        return True
    
    def all_distances_rational(points):
        """Check all pairwise distances are rational"""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if not is_perfect_square(dx*dx + dy*dy):
                    return False
        return True
    
    # Verify base set
    assert all_distances_rational(base_points), "Base set has non-rational distances"
    assert check_constraints(base_points), "Base set violates constraints"
    
    best_points = base_points[:]
    
    # Try to find a 6th point by searching over a grid
    # We'll search in a reasonable range
    max_coord = 200
    found_6th = False
    
    # For efficiency, precompute which grid points have rational distance to all base points
    candidates = []
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            if (x, y) in base_points:
                continue
            # Check distance to each base point
            ok = True
            for bx, by in base_points:
                dx = x - bx
                dy = y - by
                if not is_perfect_square(dx*dx + dy*dy):
                    ok = False
                    break
            if ok:
                # Check no three collinear with any pair from base
                collinear = False
                for i in range(len(base_points)):
                    for j in range(i+1, len(base_points)):
                        x1, y1 = base_points[i]
                        x2, y2 = base_points[j]
                        if (x2-x1)*(y-y1) - (y2-y1)*(x-x1) == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if not collinear:
                    candidates.append((x, y))
    
    # Now try adding each candidate and check 4-concyclic
    for cand in candidates:
        new_points = base_points + [cand]
        if check_constraints(new_points):
            best_points = new_points
            found_6th = True
            break
    
    # If we found a 6th point, try to find a 7th
    if found_6th and time.time() < deadline:
        max_coord2 = 300
        candidates7 = []
        for x in range(-max_coord2, max_coord2+1):
            for y in range(-max_coord2, max_coord2+1):
                if (x, y) in best_points:
                    continue
                ok = True
                for bx, by in best_points:
                    dx = x - bx
                    dy = y - by
                    if not is_perfect_square(dx*dx + dy*dy):
                        ok = False
                        break
                if ok:
                    # Check no three collinear
                    collinear = False
                    for i in range(len(best_points)):
                        for j in range(i+1, len(best_points)):
                            x1, y1 = best_points[i]
                            x2, y2 = best_points[j]
                            if (x2-x1)*(y-y1) - (y2-y1)*(x-x1) == 0:
                                collinear = True
                                break
                        if collinear:
                            break
                    if not collinear:
                        candidates7.append((x, y))
        
        for cand in candidates7:
            new_points = best_points + [cand]
            if check_constraints(new_points):
                best_points = new_points
                break
    
    # If still time, try to find an 8th
    if len(best_points) >= 7 and time.time() < deadline:
        max_coord3 = 400
        candidates8 = []
        for x in range(-max_coord3, max_coord3+1):
            for y in range(-max_coord3, max_coord3+1):
                if (x, y) in best_points:
                    continue
                ok = True
                for bx, by in best_points:
                    dx = x - bx
                    dy = y - by
                    if not is_perfect_square(dx*dx + dy*dy):
                        ok = False
                        break
                if ok:
                    collinear = False
                    for i in range(len(best_points)):
                        for j in range(i+1, len(best_points)):
                            x1, y1 = best_points[i]
                            x2, y2 = best_points[j]
                            if (x2-x1)*(y-y1) - (y2-y1)*(x-x1) == 0:
                                collinear = True
                                break
                        if collinear:
                            break
                    if not collinear:
                        candidates8.append((x, y))
        
        for cand in candidates8:
            new_points = best_points + [cand]
            if check_constraints(new_points):
                best_points = new_points
                break
    
    # Format the result
    result_points = [[[x, 1], [y, 1]] for x, y in best_points]
    return {'D': 1, 'points': result_points}
