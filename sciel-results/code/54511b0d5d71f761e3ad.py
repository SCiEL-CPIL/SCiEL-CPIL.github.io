import math
import random
import time

def construct():
    """
    Finds a set of points with rational distances, no three collinear, and no four concyclic.
    Returns a dictionary with 'D' and 'points'.
    """
    # We aim to find 6 points.
    # Using integer points (D=1) simplifies rational distance check to perfect square check.
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def sq_dist(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def is_concyclic(p1, p2, p3, p4):
        # Check if 4 points lie on a circle.
        # Determinant of the matrix formed by coordinates and squared norms should be 0.
        # | x  y  x^2+y^2  1 |
        # | x  y  x^2+y^2  1 |
        # | x  y  x^2+y^2  1 |
        # | x  y  x^2+y^2  1 |
        # For integer coordinates, we can compute the determinant exactly.
        
        def det4(m):
            # m is 4x4
            # Expansion along first row
            return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                 - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                 + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                 - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))

        rows = []
        for p in [p1, p2, p3, p4]:
            rows.append([p[0], p[1], p[0]**2 + p[1]**2, 1])
        
        return det4(rows) == 0

    def is_valid_addition(candidate, current_set):
        # Check if candidate can be added to current_set
        for p in current_set:
            d2 = sq_dist(candidate, p)
            if not is_square(d2):
                return False
        
        # Check collinearity with all pairs
        n = len(current_set)
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(current_set[i], current_set[j], candidate):
                    return False
        
        # Check concyclicity with all triples
        if n >= 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if is_concyclic(current_set[i], current_set[j], current_set[k], candidate):
                            return False
        return True

    # Search space: integer points in a box
    # We can bias the search towards points that form nice triangles.
    # Start with a known good set of 3-4 points to speed up search.
    # Or just random search.
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Generate a list of candidate points
    # We'll try a grid search with some randomization
    # Let's try to find 6 points.
    
    # Initial seed points to anchor the search (optional, but helps)
    # A right triangle (0,0), (3,0), (0,4) is a good start.
    # Distances: 3, 4, 5.
    # Let's try to build from here.
    
    initial_points = [(0,0), (3,0), (0,4)]
    
    # Verify initial points are valid
    if not is_valid_addition((3,0), [(0,0)]): return {'D': 1, 'points': [[[0,1],[0,1]], [[3,1],[0,1]]]} # Fallback
    # Actually, let's just do a general search.
    
    # We will use a backtracking search.
    # To make it efficient, we can pre-filter candidates against the first few points.
    
    # Let's try a randomized search for 6 points.
    # We'll pick points from a grid [-50, 50] x [-50, 50].
    
    best_set = [(0,0), (3,0), (0,4), (9,4)] # (9,4) is dist 5 from (0,4)? No.
    # Let's just use a robust search.
    
    # We'll try to extend a set of 3 points.
    # Points: (0,0), (3,0), (0,4)
    # Candidate pool: all integer points in [-100, 100]x[-100, 100]
    # Sort candidates by some heuristic? E.g., distance to origin?
    
    # To ensure we find a solution if one exists in the box, we iterate.
    # But 200x200 = 40000 points.
    # Backtracking depth 3 (to get to 6 from 3).
    # Branching factor is high.
    
    # Alternative: Use a known construction.
    # It is known that the following 6 points work:
    # (0,0), (3,0), (0,4), (5,0), (0,12), (5,12) ... let's check.
    # Actually, let's just run the search.
    
    # We'll generate a list of all integer points in a box.
    box = 60
    points_pool = [(x, y) for x in range(-box, box+1) for y in range(-box, box+1)]
    
    # Remove duplicates and ensure distinct
    points_pool = list(set(points_pool))
    
    # We'll try to find a clique of size 6 in the compatibility graph.
    # But we also have collinear/concyclic constraints which depend on the set, not just pairs.
    # So we must do incremental checking.
    
    # We'll use a greedy heuristic with backtracking.
    # Start with (0,0).
    # Add points that are compatible.
    
    def search(current_set, pool):
        if time.time() > deadline:
            return current_set
        if len(current_set) == 6:
            return current_set
        
        # Try to add a point from the pool
        # To improve performance, we can shuffle the pool or sort by some metric.
        # Let's just iterate.
        
        for i, p in enumerate(pool):
            if p in current_set:
                continue
            if is_valid_addition(p, current_set):
                new_set = current_set + [p]
                new_pool = pool[i+1:] # Avoid re-trying previous points? 
                # Actually, since order doesn't matter for the final set, 
                # and we want to find ANY valid set, we can just recurse.
                # But we must be careful not to miss combinations.
                # Standard backtracking:
                res = search(new_set, pool[i+1:])
                if res is not None and len(res) == 6:
                    return res
        return None
    
    # To make the search faster, we can start with a few fixed points.
    # (0,0) is a good anchor.
    # Let's fix (0,0) in the set to reduce symmetry.
    # And maybe (3,0) and (0,4) to reduce further.
    
    # Let's try searching with (0,0) fixed.
    # Pool should exclude (0,0).
    pool_no_origin = [p for p in points_pool if p != (0,0)]
    
    # We'll try to find 5 more points.
    # This is still a large search space.
    # Let's restrict the pool to points with integer distances to (0,0).
    # i.e., x^2 + y^2 is a square.
    
    pythagorean_points = [p for p in pool_no_origin if is_square(p[0]**2 + p[1]**2)]
    
    # This reduces the pool significantly.
    # Now we need to find 5 points from this pool that are mutually compatible (distances, collinear, concyclic).
    
    # Let's run the search with this reduced pool.
    # We'll fix (0,0) and try to add 5 points.
    
    result = search([(0,0)], pythagorean_points)
    
    if result is not None and len(result) == 6:
        # Format the result
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in result]}
    else:
        # Fallback to 5 points if 6 not found
        # The seed had 5 points. Let's use a known valid 5-point set.
        # (0,0), (-56, 33), (-16, 63), (0, 33), (16, 63)
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
