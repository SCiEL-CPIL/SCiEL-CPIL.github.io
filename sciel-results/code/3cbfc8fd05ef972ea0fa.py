import time
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to check if a point is valid with a set of points
    # Points are (x, y) integers. D=1.
    # Returns True if valid, False otherwise.
    
    def check_validity(points):
        n = len(points)
        if n < 3:
            return True
        
        # Check pairwise distances are rational (integers for D=1)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist_sq = dx*dx + dy*dy
                # Check if dist_sq is a perfect square
                root = int(dist_sq**0.5)
                if root*root != dist_sq:
                    return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Cross product
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
                        # Circle through 3 points, check 4th
                        # General equation: x^2 + y^2 + Dx + Ey + F = 0
                        # D = -2a, E = -2b, F = a^2+b^2-r^2
                        # Determinant of:
                        # | x1^2+y1^2, x1, y1, 1 |
                        # | x2^2+y2^2, x2, y2, 1 | = 0
                        # | x3^2+y3^2, x3, y3, 1 |
                        # | x4^2+y4^2, x4, y4, 1 |
                        
                        # To avoid floating point, use integer arithmetic
                        # Or simpler: 
                        # (x1^2+y1^2)(x2y3-x3y2+x3y1-x1y3) + ... is messy.
                        # Let's use the determinant of 4x4 matrix.
                        # Since coords are small, this is fast.
                        
                        m = [
                            [x1*x1 + y1*y1, x1, y1, 1],
                            [x2*x2 + y2*y2, x2, y2, 1],
                            [x3*x3 + y3*y3, x3, y3, 1],
                            [x4*x4 + y4*y4, x4, y4, 1]
                        ]
                        
                        # Calculate determinant
                        # Expansion along first row
                        def det4(m):
                            # m is 4x4
                            # Cramer's rule or direct expansion
                            # Direct expansion for 4x4
                            return (
                                m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1])) -
                                m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0])) +
                                m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])) -
                                m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                            )
                            
                        if det4(m) == 0:
                            return False
        return True

    # Initial known 5-point set
    # (0,0), (-56, 33), (-16, 63), (0, 33), (16, 63)
    initial_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = initial_points
    
    # Try to expand
    # Candidate pool: lattice points in a box
    # We want to find points that are "far" enough to avoid concyclicity/collinearity
    # but close enough to have integer distances? No, integer distance is the constraint.
    # So we need points (x,y) such that distance to all existing points is integer.
    
    # This is a hard constraint. 
    # Strategy: 
    # 1. Start with 5 points.
    # 2. Generate candidates for the 6th point.
    #    A point (x,y) is a candidate if dist(P_new, P_i) is integer for all P_i in current set.
    #    This can be found by intersecting hyperbolas/circles.
    #    Simpler: Random search in a box.
    
    def find_candidate(existing, box_size=200):
        # Try to find a point that is integer-distance from all existing points
        # and not collinear/concyclic
        # Random search is inefficient. 
        # Better: Iterate over x, y in range and check.
        # But 200x200 = 40000 checks. Each check involves sqrt.
        # 40000 * 5 points * sqrt is fast.
        
        candidates = []
        for x in range(-box_size, box_size+1):
            for y in range(-box_size, box_size+1):
                # Skip if already in set
                if (x,y) in existing:
                    continue
                valid = True
                for px, py in existing:
                    dx = x - px
                    dy = y - py
                    d2 = dx*dx + dy*dy
                    r = int(d2**0.5)
                    if r*r != d2:
                        valid = False
                        break
                if valid:
                    # Quick check: not collinear with any pair
                    collinear = False
                    for i in range(len(existing)):
                        for j in range(i+1, len(existing)):
                            x1, y1 = existing[i]
                            x2, y2 = existing[j]
                            if (x2-x1)*(y-y1) - (y2-y1)*(x-x1) == 0:
                                collinear = True
                                break
                        if collinear:
                            break
                    if not collinear:
                        candidates.append((x,y))
        return candidates

    # Search loop
    # We will try to grow the set.
    # If we get stuck at 5, we try random restarts with different initial sets?
    # Or just try to add to the initial set.
    
    current_set = list(initial_points)
    
    # Try to add points one by one
    for target_size in range(6, 9):
        if time.time() > deadline:
            break
            
        added = False
        # Generate candidates for adding one point
        # To speed up, we can filter candidates by checking concyclicity with existing sets of 3
        candidates = find_candidate(current_set, box_size=150)
        
        # Shuffle candidates to try different ones
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
                
            new_set = current_set + [cand]
            if check_validity(new_set):
                current_set = new_set
                added = True
                best_set = current_set[:]
                break
        
        if not added:
            # If we couldn't add a point, we might try to replace one to open up space?
            # For now, just break and return what we have.
            # A more advanced search would do local search (swap).
            # Given time, simple greedy is safer.
            pass

    # Format output
    D = 1
    points_out = []
    for x, y in best_set:
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': D, 'points': points_out}
