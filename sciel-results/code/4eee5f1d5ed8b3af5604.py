import math
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    # Known 5-point configuration with integer distances
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_perfect_square(n):
        if n < 0:
            return False
        root = math.isqrt(n)
        return root * root == n

    def is_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (x3-x1)(y2-y1) == 0
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]) == 0

    def is_concyclic(p1, p2, p3, p4):
        # Check if p4 lies on the circle defined by p1, p2, p3
        # Determinant of the 4x4 matrix:
        # | x^2+y^2  x  y  1 |
        # ...
        # If det == 0, they are concyclic.
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Circle equation: x^2 + y^2 + Ax + By + C = 0
        # A, B, C derived from p1, p2, p3.
        # To avoid division, we use the determinant form directly or 
        # check if the linear system for A,B,C satisfied by p1,p2,p3 
        # is also satisfied by p4.
        
        # System for A, B:
        # (x2-x1)A + (y2-y1)B = -(x2^2+y2^2 - x1^2-y1^2)
        # (x3-x1)A + (y3-y1)B = -(x3^2+y3^2 - x1^2-y1^2)
        
        A = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        if A == 0:
            # Collinear, which is already forbidden, but treat as concyclic for safety
            return True
        
        B1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
        B2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
        
        # Cramer's rule for A and B coefficients (let's call them coeff_A, coeff_B to avoid confusion with point A)
        # coeff_A = (B1*(y3-y1) - B2*(y2-y1)) / A
        # coeff_B = (B2*(x2-x1) - B1*(x3-x1)) / A
        
        # The equation for p4 is:
        # x4^2 + y4^2 + coeff_A*x4 + coeff_B*y4 + C = 0
        # C = -(x1^2+y1^2 + coeff_A*x1 + coeff_B*y1)
        
        # Substitute C:
        # x4^2 + y4^2 + coeff_A*x4 + coeff_B*y4 + x1^2+y1^2 + coeff_A*x1 + coeff_B*y1 = 0
        # Multiply by A:
        # A*(x4^2+y4^2 + x1^2+y1^2) + (B1*(y3-y1) - B2*(y2-y1))*(x4+x1) + (B2*(x2-x1) - B1*(x3-x1))*(y4+y1) = 0
        
        num_A = B1*(y3-y1) - B2*(y2-y1)
        num_B = B2*(x2-x1) - B1*(x3-x1)
        
        val = A*(x4*x4 + y4*y4 + x1*x1 + y1*y1) + num_A*(x4+x1) + num_B*(y4+y1)
        
        return val == 0

    def check_candidate(candidate, existing_points):
        # Check if candidate can be added to existing_points
        # 1. Distances to all existing points must be perfect squares
        for p in existing_points:
            dx = candidate[0] - p[0]
            dy = candidate[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_perfect_square(d2):
                return False
        
        # 2. No 3 collinear
        n = len(existing_points)
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(candidate, existing_points[i], existing_points[j]):
                    return False
        
        # 3. No 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_concyclic(candidate, existing_points[i], existing_points[j], existing_points[k]):
                        return False
                        
        return True

    # Search for a 6th point
    # Generate candidates with integer distance to (0,0)
    # These are points (x,y) such that x^2 + y^2 = k^2
    # We search in a box of size [-R, R]
    
    best_set = seed_points
    
    # We can try to extend the set to 7 points if we find a 6th, but let's focus on 6 first.
    # The seed is 5 points.
    
    # Generate Pythagorean triples up to a certain limit
    # x = k * (u^2 - v^2), y = k * 2uv, hyp = k * (u^2 + v^2)
    # We want |x|, |y| <= R
    
    R = 500  # Search radius
    
    candidates = set()
    # Add (0,0) is already in seed, so we don't need to add it as a candidate for the 6th point
    # We generate points (x,y) != (0,0)
    
    # Simple generation by iterating k, u, v
    # To be efficient, we can just iterate x, y and check if x^2+y^2 is a square.
    # But generating from triples is faster.
    
    # Let's iterate x and y in a smaller range first, then expand if needed?
    # Or just generate triples.
    
    max_u = 50
    for u in range(1, max_u + 1):
        for v in range(1, u):
            if (u - v) % 2 == 0: # Ensure primitive triple
                continue
            if math.gcd(u, v) != 1:
                continue
                
            x_base = u*u - v*v
            y_base = 2*u*v
            k_base = u*u + v*v
            
            # Scale by k
            # Max k such that k*x_base <= R
            max_k = R // max(x_base, y_base) if max(x_base, y_base) > 0 else 0
            
            for k in range(1, max_k + 1):
                x = k * x_base
                y = k * y_base
                # Add all sign combinations
                candidates.add((x, y))
                candidates.add((x, -y))
                candidates.add((-x, y))
                candidates.add((-x, -y))
                # Swap x,y
                candidates.add((y, x))
                candidates.add((y, -x))
                candidates.add((-y, x))
                candidates.add((-y, -x))
                
    # Also add points on axes
    for i in range(1, R+1):
        candidates.add((i, 0))
        candidates.add((-i, 0))
        candidates.add((0, i))
        candidates.add((0, -i))
        
    # Remove seed points from candidates
    for p in seed_points:
        candidates.discard(p)
        
    # Sort candidates by distance to origin to find small ones first?
    # Or just iterate.
    
    found_6 = None
    for cand in candidates:
        if time.time() - start_time > deadline:
            break
            
        if check_candidate(cand, seed_points):
            found_6 = cand
            break
            
    if found_6:
        points_6 = seed_points + [found_6]
        # Optional: try to find a 7th point
        # This might be too slow or hard, but let's try a quick search
        # If we find 7, score is even better.
        
        candidates_7 = set(candidates)
        candidates_7.discard(found_6)
        
        found_7 = None
        # Limit search for 7th point to save time
        time_limit_7 = time.time() + 20.0
        for cand in candidates_7:
            if time.time() > time_limit_7:
                break
            if check_candidate(cand, points_6):
                found_7 = cand
                break
                
        if found_7:
            points_6 = points_6 + [found_7]
            
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points_6]}
        
    # If no 6th point found, return the 5-point set
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
