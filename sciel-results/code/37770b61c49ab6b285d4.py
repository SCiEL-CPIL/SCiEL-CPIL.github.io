import random
import math
import time
from fractions import Fraction

def construct():
        start_time = time.time()
        time_limit = 115.0 # Leave buffer for overhead
        
        def is_square(n):
            if n < 0: return False
            r = math.isqrt(n)
            return r * r == n

        def are_collinear(p1, p2, p3):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

        def are_concyclic(p1, p2, p3, p4):
            # Exact rational check
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            # Circle equation: A(x^2+y^2) + Bx + Cy + D = 0
            # We solve for B, C, D relative to A=1? No, A can be 0 (line) but no 3 collinear.
            # So A != 0.
            # x1^2+y1^2 + (B/A)x1 + (C/A)y1 + (D/A) = 0
            # Let b=B/A, c=C/A, d=D/A.
            # x1^2+y1^2 + b x1 + c y1 + d = 0
            # x2^2+y2^2 + b x2 + c y2 + d = 0
            # x3^2+y3^2 + b x3 + c y3 + d = 0
            
            # Subtract 1 from 2:
            # (x2^2+y2^2 - x1^2-y1^2) + b(x2-x1) + c(y2-y1) = 0
            # (x3^2+y3^2 - x1^2-y1^2) + b(x3-x1) + c(y3-y1) = 0
            
            a11 = x2 - x1
            a12 = y2 - y1
            b1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
            
            a21 = x3 - x1
            a22 = y3 - y1
            b2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
            
            det = a11 * a22 - a12 * a21
            if det == 0:
                return False # Collinear, handled separately
            
            # b = (b1*a22 - a12*b2) / det
            # c = (a11*b2 - b1*a21) / det
            # d = -(x1^2+y1^2 + b*x1 + c*y1)
            
            # Check p4:
            # x4^2+y4^2 + b*x4 + c*y4 + d = 0
            # Multiply by det to avoid fractions:
            # det*(x4^2+y4^2) + (b1*a22 - a12*b2)*x4 + (a11*b2 - b1*a21)*y4 + det*d = 0
            # det*d = -det*(x1^2+y1^2) - (b1*a22 - a12*b2)*x1 - (a11*b2 - b1*a21)*y1
            
            b_num = b1 * a22 - a12 * b2
            c_num = a11 * b2 - b1 * a21
            d_num = -det * (x1*x1 + y1*y1) - b_num * x1 - c_num * y1
            
            val = det * (x4*x4 + y4*y4) + b_num * x4 + c_num * y4 + d_num
            return val == 0

        def check_validity(points):
            n = len(points)
            # Check distances
            for i in range(n):
                for j in range(i+1, n):
                    dx = points[i][0] - points[j][0]
                    dy = points[i][1] - points[j][1]
                    if not is_square(dx*dx + dy*dy):
                        return False
            
            # Check collinear
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_collinear(points[i], points[j], points[k]):
                            return False
            
            # Check concyclic
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            if are_concyclic(points[i], points[j], points[k], points[l]):
                                return False
            return True

        # Seeds
        seeds = [
            [(0, 0), (3, 4), (0, 5)], # 3 points
            [(0, 0), (-56, 33), (-16, 63)], # 3 points
            [(0, 0), (5, 0), (0, 12)], # 3 points
            [(0, 0), (-56, 33), (-16, 63), (0, 33)] # 4 points
        ]
        
        # Generate candidates
        # We use a grid. To be efficient, we pre-compute distances? No, just filter on the fly.
        # Grid size 500x500 is 250k points.
        candidates = []
        for x in range(-500, 501):
            for y in range(-500, 501):
                candidates.append((x, y))
        
        best_set = []
        
        for seed in seeds:
            # Verify seed is valid
            if not check_validity(seed):
                continue
            
            current_set = seed[:]
            # Filter candidates to those with integer distance to all points in seed
            # This is a strong filter.
            valid_candidates = []
            for p in candidates:
                if p in current_set: continue
                ok = True
                for q in current_set:
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    if not is_square(dx*dx + dy*dy):
                        ok = False
                        break
                if ok:
                    valid_candidates.append(p)
            
            # Shuffle
            random.seed(12345)
            random.shuffle(valid_candidates)
            
            # Greedy Add with Backtracking (limited depth)
            # Since backtracking is expensive, we do a few random restarts of the greedy part
            for restart in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                random.shuffle(valid_candidates)
                cur = seed[:]
                
                for p in valid_candidates:
                    if time.time() - start_time > time_limit:
                        break
                    
                    # Check if p can be added
                    # 1. Distance to all in cur is already guaranteed by valid_candidates
                    # 2. Check general position with cur + p
                    new_set = cur + [p]
                    if check_validity(new_set):
                        cur = new_set
                
                if len(cur) > len(best_set):
                    best_set = cur[:]
        
        if not best_set:
            # Fallback to the original 5-point set if something went wrong
            best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        
        # Format output
        points_out = []
        for x, y in best_set:
            points_out.append([[x, 1], [y, 1]])
            
        return {'D': 1, 'points': points_out}
