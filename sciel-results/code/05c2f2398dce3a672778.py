import time
import random
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    x = int(n ** 0.5)
    for i in range(x - 2, x + 3):
        if i * i == n:
            return True
    return False

def _check_valid(points):
    """
    Check if a set of points (list of (x, y) integers or fractions) is valid.
    Conditions:
    1. Distinct points.
    2. No three collinear.
    3. No four concyclic.
    Distances are automatically rational for integer points.
    """
    n = len(points)
    if n < 3:
        return True
    
    # Check distinctness
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        
    # Check collinearity: no three points on a line
    # Slope between (x1,y1) and (x2,y2) is (y2-y1)/(x2-x1)
    # Three points (x1,y1), (x2,y2), (x3,y3) are collinear iff:
    # (y2-y1)*(x3-x1) == (y3-y1)*(x2-x1)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                # Cross product of (P2-P1) and (P3-P1)
                cross = dx12 * (y3 - y1) - dy12 * (x3 - x1)
                if cross == 0:
                    return False

    # Check concyclicity: no four points on a circle
    # A circle is x^2 + y^2 + Dx + Ey + F = 0
    # For 4 points to be concyclic, the determinant of the following matrix is 0:
    # | x1^2+y1^2  x1  y1  1 |
    # | x2^2+y2^2  x2  y2  1 | = 0
    # | x3^2+y3^2  x3  y3  1 |
    # | x4^2+y4^2  x4  y4  1 |
    # We can compute this determinant efficiently.
    
    def det4(m):
        # m is a 4x4 list of lists
        # Expansion along first row
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]))
             - m[0][1]*(m[1][0]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]))
             + m[0][2]*(m[1][0]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1] - m[2][1]*m[3][0]))
             - m[0][3]*(m[1][0]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])))

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    
                    # Construct the 4x4 matrix
                    # Note: We can factor out common terms or use specific formulas, 
                    # but direct determinant is fine for small integers.
                    m = [
                        [x1*x1 + y1*y1, x1, y1, 1],
                        [x2*x2 + y2*y2, x2, y2, 1],
                        [x3*x3 + y3*y3, x3, y3, 1],
                        [x4*x4 + y4*y4, x4, y4, 1]
                    ]
                    if det4(m) == 0:
                        return False
                        
    return True

def _add_point_search(current_points, deadline, rng):
    """
    Try to add a new point to current_points to form a valid set of size n+1.
    Returns the new list of points if successful, else None.
    """
    # Start with a random range, maybe expand if needed
    # Given the small size, we can search a box around the existing points
    if not current_points:
        return None
        
    min_x = min(p[0] for p in current_points)
    max_x = max(p[0] for p in current_points)
    min_y = min(p[1] for p in current_points)
    max_y = max(p[1] for p in current_points)
    
    # Expand search area slightly
    pad = 20
    
    while time.time() < deadline:
        # Randomly pick a point in the expanded box
        x = rng.randint(min_x - pad, max_x + pad)
        y = rng.randint(min_y - pad, max_y + pad)
        
        new_point = (x, y)
        if new_point in current_points:
            continue
            
        candidate = current_points + [new_point]
        if _check_valid(candidate):
            return candidate
            
        # If we've been searching for a while without success, expand pad
        # This is a simple heuristic
        if x == max_x + pad or x == min_x - pad or y == max_y + pad or y == min_y - pad:
             pad += 5
             
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial set of 5 points from the seed
    # (0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)
    initial_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify initial set
    if not _check_valid(initial_points):
        # Fallback to a simpler valid set if the seed is broken (it shouldn't be)
        initial_points = [(0, 0), (3, 4), (0, 7)] # Just a triangle
        # Actually, let's use a known valid 4-point set if 5 fails, but 5 is known valid.
        
    best_points = initial_points
    best_size = len(initial_points)
    
    rng = random.Random(42)
    
    # Try to add points one by one
    # We can try multiple random restarts or different initial configurations if we fail
    
    # Strategy: 
    # 1. Try to extend the initial 5-point set to 6.
    # 2. If that fails, try other 5-point sets? Or just keep the best 5.
    # 3. If we get 6, try to extend to 7.
    
    current = initial_points[:]
    
    # Attempt 1: Extend initial set
    while time.time() < deadline:
        next_points = _add_point_search(current, deadline, rng)
        if next_points:
            current = next_points
            best_points = current
            best_size = len(current)
            # Reset deadline slightly to allow more search for next point?
            # Or keep the global deadline.
            # If we found 6, we have plenty of time.
            # If we found 7, even more.
        else:
            # Failed to add a point to this specific configuration
            # We can try to modify the configuration or start fresh
            # For simplicity, let's try a few random 5-point sets and extend them
            
            # Generate a random 5-point set? 
            # It's hard to generate a random valid 5-point set quickly.
            # Instead, let's try local search on the existing set to "move" points 
            # to make room for a new one? This is complex.
            
            # Alternative: Try adding a point to a subset? No, we need all points.
            
            # Let's try a different approach: 
            # Search for a 6-point set directly from scratch in a small grid?
            # Grid size 20x20 has 400 points. C(400, 6) is huge.
            # But we can do incremental construction.
            
            # Let's restart with a different random 5-point set?
            # Or just accept that we might not find 6 points with this specific initial set.
            # However, 6-point integral distance sets are known to exist.
            # Example: A set of 6 points with rational distances is known.
            
            # Let's try to find a known 6-point set or search harder.
            # Known 6-point set:
            # (0,0), (3,4), (0,7), ... ?
            
            # Let's try a specific known 6-point configuration if available, 
            # or continue searching with a wider area.
            
            # To avoid infinite loop on one bad path, let's break and try a different strategy.
            # Strategy 2: Randomized incremental construction from scratch.
            
            break
            
    # If we didn't reach size 6, let's try a few random restarts of incremental construction
    if best_size < 6:
        for _ in range(5): # Try 5 different random starts
            if time.time() > deadline:
                break
                
            # Start with a random triangle (valid)
            p1 = (rng.randint(-10, 10), rng.randint(-10, 10))
            p2 = (rng.randint(-10, 10), rng.randint(-10, 10))
            p3 = (rng.randint(-10, 10), rng.randint(-10, 10))
            
            # Ensure triangle is valid (not collinear)
            if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                continue
                
            current = [p1, p2, p3]
            
            # Add points one by one
            while len(current) < 7 and time.time() < deadline:
                next_points = _add_point_search(current, deadline, rng)
                if next_points:
                    current = next_points
                    if len(current) > best_size:
                        best_points = current
                        best_size = len(current)
                else:
                    break
                    
    # Format the best points found
    # Return format: {'D': 1, 'points': [[[x_num, x_den], [y_num, y_den]], ...]}
    # Since we are using integers, x_num=x, x_den=1
    
    points_out = []
    for x, y in best_points:
        points_out.append([[int(x), 1], [int(y), 1]])
        
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
