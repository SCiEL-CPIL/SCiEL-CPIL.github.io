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
            # dx^2 = dx_num^2 / dx_den^2
            # D*dy^2 = D * dy_num^2 / dy_den^2
            # sum = (dx_num^2 * dy_den^2 + D * dy_num^2 * dx_den^2) / (dx_den^2 * dy_den^2)
            num = dx_num*dx_num * dy_den*dy_den + D * dy_num*dy_num * dx_den*dx_den
            den = dx_den*dx_den * dy_den*dy_den
            if den == 0:
                return False
            if not _is_square(num) or not _is_square(den):
                return False
            # Actually, we need the ratio to be a perfect square of a rational.
            # i.e. num/den must be a perfect square.
            # This is equivalent to num * den being a perfect square? No.
            # Let num/den = k^2. Then num = k^2 den.
            # If num and den are integers, num/den is a square iff num*den is a square?
            # Let num = 2, den = 8. num/den = 1/4 = (1/2)^2. num*den = 16 = 4^2.
            # Let num = 1, den = 2. num/den = 0.5. Not square. num*den = 2. Not square.
            # Let num = 4, den = 9. num/den = 4/9. Square. num*den = 36. Square.
            # Yes, num/den is a rational square iff num*den is an integer square.
            if not _is_square(num * den):
                return False

    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i][0][0]/points[i][0][1], points[i][1][0]/points[i][1][1]
                x2, y2 = points[j][0][0]/points[j][0][1], points[j][1][0]/points[j][1][1]
                x3, y3 = points[k][0][0]/points[k][0][1], points[k][1][0]/points[k][1][1]
                # Cross product (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                # Use exact arithmetic
                # (x2-x1) = (x2_n*x1_d - x1_n*x2_d) / (x1_d*x2_d)
                # Let's just use Fraction for simplicity in checking, as n<=12
                # Or stick to ints.
                dx12_n = x2*points[j][0][1] - x1*points[i][0][1] # This is wrong, x2 is float
                # Let's do it properly with ints
                x1n, x1d = points[i][0]
                y1n, y1d = points[i][1]
                x2n, x2d = points[j][0]
                y2n, y2d = points[j][1]
                x3n, x3d = points[k][0]
                y3n, y3d = points[k][1]
                
                # Vector 12: (x2n/x2d - x1n/x1d, y2n/y2d - y1n/y1d)
                # Vector 13: (x3n/x3d - x1n/x1d, y3n/y3d - y1n/y1d)
                # Cross product: (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                
                def sub_frac(n1, d1, n2, d2):
                    return n1*d2 - n2*d1, d1*d2
                
                dx12_n, dx12_d = sub_frac(x2n, x2d, x1n, x1d)
                dy12_n, dy12_d = sub_frac(y2n, y2d, y1n, y1d)
                dx13_n, dx13_d = sub_frac(x3n, x3d, x1n, x1d)
                dy13_n, dy13_d = sub_frac(y3n, y3d, y1n, y1d)
                
                # Cross = (dx12_n/dx12_d) * (dy13_n/dy13_d) - (dy12_n/dy12_d) * (dx13_n/dx13_d)
                # = (dx12_n*dy13_n*dy12_d*dx13_d - dy12_n*dx13_n*dx12_d*dy13_d) / (dx12_d*dy13_d*dy12_d*dx13_d)
                num = dx12_n*dy13_n*dy12_d*dx13_d - dy12_n*dx13_n*dx12_d*dy13_d
                if num != 0:
                    continue
                # If num is 0, they are collinear
                return False

    # Check no four concyclic
    # For D=1, points (x,y). Circle: x^2 + y^2 + Ax + By + C = 0
    # Determinant of:
    # [x^2+y^2, x, y, 1]
    # Must be 0 for 4 points.
    # For general D, "circle" is x^2 + D y^2 + Ax + By + C = 0
    # Matrix rows: [x^2 + D y^2, x, y, 1]
    # We can check if any 4 points are linearly dependent in this way.
    # Or just check determinant of 4x4 matrix for every combination of 4 points.
    # n<=12, C(12,4)=495. 4x4 det is fast.
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    # Build 4x4 matrix
                    # Row for point p: [x^2 + D*y^2, x, y, 1]
                    # Use exact rationals? Or just check if det is 0.
                    # Since coordinates are rationals, we can clear denominators.
                    # Or just use Fraction for the determinant calculation.
                    # Given n is small, Fraction is fine.
                    
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
                    
                    # Calculate determinant
                    # Simple expansion for 4x4
                    det = 0
                    # Using Sarrus-like or cofactor expansion
                    # Let's just do cofactor expansion on first row
                    def det3(m):
                        return (m[0][0]*(m[1][1]*m[2][2]-m[2][1]*m[1][2]) -
                                m[0][1]*(m[1][0]*m[2][2]-m[2][0]*m[1][2]) +
                                m[0][2]*(m[1][0]*m[2][1]-m[2][0]*m[1][1]))
                    
                    for c in range(4):
                        minor = [row[:c] + row[c+1:] for row in M[1:]]
                        det += ((-1)**c) * M[0][c] * det3(minor)
                    
                    if det == 0:
                        return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with the known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = base_points
    D = 1
    
    # Try to find a 6th point
    # Search space: integers x, y in some range
    # Since D=1, distance squared is (x1-x2)^2 + (y1-y2)^2.
    # We need this to be a perfect square for all existing points.
    
    # Heuristic: Try points near the existing ones or on a grid.
    # The existing points are in a 112x63 box.
    # Let's search in a box around them, e.g., [-200, 200] x [-200, 200]
    
    # To speed up, precompute for each candidate (x,y) the squared distances to base points.
    # If any is not a square, reject.
    
    # Also need to check collinearity and concyclicity with new point.
    # Collinearity: Check against all pairs of base points.
    # Concyclicity: Check against all triples of base points.
    
    # Let's iterate x, y
    # Range estimation: The points are spread out. A new point should likely be nearby or symmetric.
    # Let's try a grid search.
    
    # Optimization: 
    # 1. Check distance squares first (fastest reject).
    # 2. Check collinearity.
    # 3. Check concyclicity.
    
    # We can precompute the circle equations for all triples of base points?
    # There are C(5,3)=10 triples.
    # Circle: x^2 + y^2 + Ax + By + C = 0
    # For a new point (x,y), it lies on the circle if x^2 + y^2 + Ax + By + C = 0.
    
    circles = []
    for i in range(5):
        for j in range(i+1, 5):
            for k in range(j+1, 5):
                p1, p2, p3 = base_points[i], base_points[j], base_points[k]
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                
                # Solve for A, B, C
                # x1^2+y1^2 + A x1 + B y1 + C = 0
                # x2^2+y2^2 + A x2 + B y2 + C = 0
                # x3^2+y3^2 + A x3 + B y3 + C = 0
                
                # Subtract 1 from 2:
                # (x2^2-x1^2) + (y2^2-y1^2) + A(x2-x1) + B(y2-y1) = 0
                # Subtract 1 from 3:
                # (x3^2-x1^2) + (y3^2-y1^2) + A(x3-x1) + B(y3-y1) = 0
                
                # Let dx21 = x2-x1, dy21 = y2-y1, etc.
                # A dx21 + B dy21 = -((x2^2-x1^2) + (y2^2-y1^2))
                # A dx31 + B dy31 = -((x3^2-x1^2) + (y3^2-y1^2))
                
                dx21 = x2 - x1
                dy21 = y2 - y1
                dx31 = x3 - x1
                dy31 = y3 - y1
                
                rhs2 = -((x2*x2 - x1*x1) + (y2*y2 - y1*y1))
                rhs3 = -((x3*x3 - x1*x1) + (y3*y3 - y1*y1))
                
                det = dx21*dy31 - dx31*dy21
                if det == 0:
                    continue # Collinear, no circle (or line)
                
                A_num = (rhs2*dy31 - rhs3*dy21) / det
                B_num = (dx21*rhs3 - dx31*rhs2) / det
                
                # C = - (x1^2 + y1^2 + A x1 + B y1)
                C_num = - (x1*x1 + y1*y1 + A_num*x1 + B_num*y1)
                
                circles.append((A_num, B_num, C_num))
    
    # Search for 6th point
    # We will search integers x, y
    # To avoid re-checking, we can just iterate.
    # Range: Let's try [-200, 200]
    # If we find one, we can stop or keep looking for better (though score is just count).
    
    found_6th = None
    
    # Sort search order? Maybe distance from origin or center?
    # Center of mass:
    cx = sum(p[0] for p in base_points) / 5
    cy = sum(p[1] for p in base_points) / 5
    
    # Generate candidates
    # To make it faster, we can check distance squares first.
    
    # Let's try a spiral or box search.
    # Box: x in [-150, 150], y in [-150, 150]
    
    # Precompute base points squared distances? No, depends on candidate.
    
    # Let's just loop.
    # 300x300 = 90,000 candidates.
    # For each:
    # 5 distance checks (sqrt? no, is_square).
    # 5 collinearity checks (cross product).
    # 10 concyclicity checks (evaluate polynomial).
    
    # This should be fast enough in Python if optimized.
    
    # Optimization: 
    # Distance check: (x-xi)^2 + (y-yi)^2 must be square.
    # We can precompute xi, yi.
    
    bxs = [p[0] for p in base_points]
    bys = [p[1] for p in base_points]
    
    # Collinearity check: 
    # For each pair (i,j) in base, check if (x,y) is on line ij.
    # Line: (y-yi)(xj-xi) - (x-xi)(yj-yi) = 0
    # (y-yi)(xj-xi) = (x-xi)(yj-yi)
    
    lines = []
    for i in range(5):
        for j in range(i+1, 5):
            x1, y1 = base_points[i]
            x2, y2 = base_points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Equation: dy*x - dx*y + (dx*y1 - dy*x1) = 0
            # dy*x - dx*y = dx*y1 - dy*x1
            lines.append((dy, -dx, dx*y1 - dy*x1))
    
    # Concyclicity check:
    # For each circle (A, B, C), check if x^2 + y^2 + A*x + B*y + C == 0
    
    # Search
    # We'll search x, y in a range.
    # Let's start with a smaller range to find something quickly, then expand if time permits.
    
    search_range = 100
    
    # To prioritize, maybe search near existing points?
    # Or just grid.
    
    # Let's use a list of candidates and sort by some metric? 
    # Or just iterate.
    
    # To speed up, we can check if x,y is already in base_points.
    base_set = set(base_points)
    
    # We'll iterate x from -search_range to search_range
    # and y from -search_range to search_range
    
    # We can break early if we find a valid point.
    
    # Let's add a time check inside the loop.
    
    # To make it even faster, we can precompute x^2 and y^2?
    # x^2 is independent of y.
    
    # Let's structure the loop:
    # for x in range(...):
    #   x2 = x*x
    #   for y in range(...):
    #     y2 = y*y
    #     ...
    
    # Distance check:
    # for i in 5:
    #   d2 = (x-bx[i])^2 + (y-by[i])^2
    #   if not is_square(d2): break
    
    # Collinearity:
    # for (dy, dx, c) in lines:
    #   if dy*x + dx*y + c != 0: continue # Wait, sign
    #   # dy*x - dx*y = c  => dy*x - dx*y - c = 0
    #   # So if dy*x - dx*y - c == 0, it's collinear.
    
    # Concyclicity:
    # for (A, B, C) in circles:
    #   if x2 + y2 + A*x + B*y + C == 0: fail
    
    # Note: A, B, C are Fractions.
    
    # Let's implement this.
    
    # To avoid iterating the full range if we find a point, we can return immediately.
    # But maybe there are multiple? Doesn't matter, we just need 6.
    
    # What if we don't find one in [-100, 100]?
    # We can expand.
    
    # Let's try.
    
    # To speed up the inner loop, we can skip if (x,y) is in base_set.
    
    # One more optimization: 
    # If we find a point, we should verify it with the full _check_valid function 
    # to be absolutely sure (especially for concyclicity with D=1).
    
    # Let's run the search.
    
    # We'll use a nested loop.
    # To handle the deadline, we check time every 1000 iterations.
    
    count = 0
    for x in range(-search_range, search_range + 1):
        x2 = x*x
        for y in range(-search_range, search_range + 1):
            count += 1
            if count % 1000 == 0:
                if time.time() > deadline:
                    break
            
            if (x, y) in base_set:
                continue
            
            y2 = y*y
            
            # Distance check
            valid_dist = True
            for i in range(5):
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
                # dy*x - dx*y - c == 0
                if dy*x - dx*y - c == 0:
                    valid_coll = False
                    break
            
            if not valid_coll:
                continue
            
            # Concyclicity check
            valid_conc = True
            for A, B, C in circles:
                # x2 + y2 + A*x + B*y + C == 0
                # A, B, C are Fractions
                val = x2 + y2 + A*x + B*y + C
                if val == 0:
                    valid_conc = False
                    break
            
            if not valid_conc:
                continue
            
            # Candidate found!
            # Verify with full check
            new_point = [[x, 1], [y, 1]]
            test_points = base_points + [new_point]
            if _check_valid(test_points, 1):
                best_points = test_points
                found_6th = (x, y)
                break
        
        if found_6th:
            break
    
    # If we found a 6th point, great.
    # Can we find a 7th?
    # Probably harder.
    # Given time budget, maybe just return 6 if found.
    # If not found, return 5.
    
    # If we have time, we could try to find a 7th point using the 6-point set as base.
    # But the search space is larger and checks are more expensive.
    # C(6,3)=20 circles, C(6,2)=15 lines, 6 distances.
    # Search range might need to be smaller or smarter.
    
    # Let's just return what we have.
    
    # Convert best_points from list of tuples (x, y) to the required format:
    # [[[x_num, x_den], [y_num, y_den]], ...]
    # D=1, so points are (x, y*1). x, y are integers in our search.
    formatted_points = []
    for p in best_points:
        x, y = p
        # x is int, y is int. Represent as [x, 1] and [y, 1]
        formatted_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': formatted_points}

# EVOLVE-BLOCK-END
