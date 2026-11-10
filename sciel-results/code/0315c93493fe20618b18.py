"""Search for integral point sets in general position (D=1, integer points)."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    # Precompute some useful integer points
    # We'll search in a bounded box, trying to maximize count
    # Key: all pairwise distances must be integers
    # No three collinear, no four concyclic

    def is_int_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        # Check if d2 is a perfect square
        r = int(d2**0.5)
        return r*r == d2

    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1,y1 = points[i]
                    x2,y2 = points[j]
                    x3,y3 = points[k]
                    # Cross product
                    if (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0:
                        return False
        return True

    def no_four_concyclic(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        x1,y1 = points[i]
                        x2,y2 = points[j]
                        x3,y3 = points[k]
                        x4,y4 = points[l]
                        # Four points concyclic iff the determinant is 0
                        # |x^2+y^2  x  y  1|
                        # |x^2+y^2  x  y  1|
                        # |x^2+y^2  x  y  1|
                        # |x^2+y^2  x  y  1|
                        a1 = x1*x1 + y1*y1
                        a2 = x2*x2 + y2*y2
                        a3 = x3*x3 + y3*y3
                        a4 = x4*x4 + y4*y4
                        # Compute determinant of 4x4 matrix
                        # Using expansion
                        # det = a1*(x2*(y3-y4) - y2*(x3-x4) + (x3*y4 - y3*x4))
                        #     - x1*(a2*(y3-y4) - a3*(y2-y4) + a3*y2 - a2*y3 ... )
                        # Let me use a cleaner formula
                        # det of [[a1,x1,y1,1],[a2,x2,y2,1],[a3,x3,y3,1],[a4,x4,y4,1]]
                        # = a1*(x2*(y3-y4) - y2*(x3-x4) + (x3*y4-y3*x4))
                        #   - x1*(a2*(y3-y4) - y2*(a3-a4) + a3*y4 - a2*y3 ... )
                        # This is getting complex. Let me use a different approach.
                        # Four points are concyclic iff:
                        # (x1-x2)^2+(y1-y2)^2 times something... 
                        # Actually, use the determinant directly with cofactor expansion
                        # Row1: [a1, x1, y1, 1]
                        # det = a1 * det[[x2,y2,1],[x3,y3,1],[x4,y4,1]]
                        #     - x1 * det[[a2,y2,1],[a3,y3,1],[a4,y4,1]]
                        #     + y1 * det[[a2,x2,1],[a3,x3,1],[a4,x4,1]]
                        #     - 1 * det[[a2,x2,y2],[a3,x3,y3],[a4,x4,y4]]
                        
                        def det3(r1, r2, r3):
                            return (r1[0]*(r2[1]*r3[2]-r2[2]*r3[1])
                                    - r1[1]*(r2[0]*r3[2]-r2[2]*r3[0])
                                    + r1[2]*(r2[0]*r3[1]-r2[1]*r3[0]))
                        
                        d = (a1 * det3([x2,y2,1],[x3,y3,1],[x4,y4,1])
                             - x1 * det3([a2,y2,1],[a3,y3,1],[a4,y4,1])
                             + y1 * det3([a2,x2,1],[a3,x3,1],[a4,x4,1])
                             - det3([a2,x2,y2],[a3,x3,y3],[a4,x4,y4]))
                        if d == 0:
                            return False
        return True

    def all_int_distances(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if not is_int_dist(points[i], points[j]):
                    return False
        return True

    def valid_set(points):
        if not all_int_distances(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True

    # Strategy: Start with known good small sets, try to extend
    # Known 5-point set from seed
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find 6-point sets by adding points
    # Search space: integer points in a reasonable range
    # For efficiency, precompute candidate points that have integer distances
    # to all points in the current set
    
    best_count = 5
    best_points = base5
    
    # Generate candidate points in a box
    # Points should be at integer distances from each other
    # Let's try a broader search
    
    # First, let's try to extend the base5 set
    # A point (x,y) must have integer distance to all 5 points
    # dist to (0,0): x^2+y^2 = a^2
    # dist to (-56,33): (x+56)^2+(y-33)^2 = b^2
    # etc.
    
    # This is a system of Diophantine equations. Let's try a search approach.
    
    # Generate all integer points in [-200, 200]^2 that form integer distances
    # among themselves (at least for small subsets)
    
    # For efficiency, let's use a different approach:
    # Known integral point sets can be found in literature
    # Let me try a local search / hill climbing approach
    
    # Start with the 5-point set, try to add a 6th point
    # The 6th point must be at integer distance from all 5 existing points
    
    # Let's try a randomized local search
    import random
    random.seed(42)
    
    # Try to find a 6th point
    # Point must satisfy: for each existing point (xi, yi), (x-xi)^2 + (y-yi)^2 is a perfect square
    # This is very restrictive. Let's try a grid search in a reasonable range.
    
    # Actually, let me try known constructions from literature
    # There are known 6-point integral sets. Let me try to find one.
    
    # Alternative: use a more systematic search
    # Try all integer points in [-100, 100] x [-100, 100]
    # and check if they can be added to the base set
    
    # For speed, let's limit the search
    candidates = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            if (x, y) in base5:
                continue
            # Check integer distance to all base5 points
            ok = True
            for (xi, yi) in base5:
                if not is_int_dist((x,y), (xi,yi)):
                    ok = False
                    break
            if ok:
                candidates.append((x, y))
    
    # Try adding each candidate
    for cand in candidates:
        new_set = base5 + [cand]
        if valid_set(new_set):
            best_count = 6
            best_points = new_set
            break
    
    if elapsed() > DEADLINE - 20:
        # Return what we have
        pts = best_points
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # Try to find 7-point sets
    # This is much harder. Let's try a different approach:
    # Start from scratch with a local search
    
    # Generate a pool of "good" points: pairs with integer distances
    # Then search for large cliques in the "integer distance" graph
    # with the additional constraints
    
    # For a 7-point set, we need 7 points all pairwise at integer distances,
    # no three collinear, no four concyclic.
    
    # Let me try a more targeted search
    # Known: there exist integral point sets with 6 points.
    # Let me try to find one by searching more broadly.
    
    # Try a different base set
    # The seed set has a specific structure. Let me try other known sets.
    
    # Actually, let me try a random restart local search
    # Start with 3 random integer points, greedily add more
    
    best_overall = best_points
    
    for restart in range(50):
        if elapsed() > DEADLINE - 10:
            break
            
        # Start with 3 points
        # Choose 3 points that form an integer triangle
        # and are not collinear
        pts = []
        # Try to build up from 3 points
        # Pick first two points at integer distance
        # Then find third
        
        # Use a small set of known good starting triangles
        # (0,0), (3,4), (5,0) - right triangle
        # (0,0), (5,0), (3,4)
        
        # Let me try a systematic approach
        # Fix (0,0) and try to build from there
        
        # Actually, let me just try to extend the 6-point set if we found one
        if best_count >= 6:
            # Try to add a 7th point to best_points
            for x in range(-150, 151):
                for y in range(-150, 151):
                    if (x, y) in best_points:
                        continue
                    ok = True
                    for (xi, yi) in best_points:
                        if not is_int_dist((x,y), (xi,yi)):
                            ok = False
                            break
                    if ok:
                        new_set = best_points + [(x,y)]
                        if valid_set(new_set):
                            best_count = 7
                            best_points = new_set
                            best_overall = new_set
                            break
                if best_count >= 7:
                    break
        else:
            # Try random construction
            # Pick 3 random integer points forming integer triangle
            for attempt in range(200):
                if elapsed() > DEADLINE - 10:
                    break
                p1 = (random.randint(-50, 50), random.randint(-50, 50))
                p2 = (random.randint(-50, 50), random.randint(-50, 50))
                if p1 == p2 or not is_int_dist(p1, p2):
                    continue
                # Find p3
                found_p3 = False
                for p3 in [(random.randint(-50,50), random.randint(-50,50)) for _ in range(100)]:
                    if p3 == p1 or p3 == p2:
                        continue
                    if is_int_dist(p1, p3) and is_int_dist(p2, p3):
                        # Check not collinear
                        if (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) != 0:
                            pts = [p1, p2, p3]
                            found_p3 = True
                            break
                if not found_p3:
                    continue
                
                # Try to add more points
                for _ in range(100):
                    if elapsed() > DEADLINE - 10:
                        break
                    added = False
                    for x in range(-80, 81):
                        for y in range(-80, 81):
                            p = (x, y)
                            if p in pts:
                                continue
                            ok = all(is_int_dist(p, q) for q in pts)
                            if ok:
                                new_pts = pts + [p]
                                if no_three_collinear(new_pts) and no_four_concyclic(new_pts):
                                    pts = new_pts
                                    added = True
                                    break
                        if added:
                            break
                    if not added:
                        break
                
                if len(pts) > len(best_overall):
                    best_overall = pts
                    best_count = len(pts)
                    best_points = pts
    
    pts = best_overall
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
