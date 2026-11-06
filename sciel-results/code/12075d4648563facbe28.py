import math
import time
import random

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
    # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    # Determinant of 4x4 matrix
    # | x1^2+y1^2, x1, y1, 1 |
    # | x2^2+y2^2, x2, y2, 1 |
    # | x3^2+y3^2, x3, y3, 1 |
    # | x4^2+y4^2, x4, y4, 1 |
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    c1 = x1*x1 + y1*y1
    c2 = x2*x2 + y2*y2
    c3 = x3*x3 + y3*y3
    c4 = x4*x4 + y4*y4
    
    # Calculate determinant using Sarrus-like expansion or sub-determinants
    # det = c1*(x2*(y3*1 - y4*1) - y2*(x3*1 - x4*1) + 1*(x3*y4 - x4*y3))
    #     - x1*(c2*(y3*1 - y4*1) - y2*(c3*1 - c4*1) + 1*(c3*y4 - c4*y3))
    #     + y1*(c2*(x3*1 - x4*1) - x2*(c3*1 - c4*1) + 1*(c3*x4 - c4*x3))
    #     - 1*(c2*(x3*y4 - x4*y3) - x2*(c3*y4 - c4*y3) + y2*(c3*x4 - c4*x3))
    
    # Let's use a more compact formula for determinant of 4x4
    # Rows: [c, x, y, 1]
    
    # M11 = c2*(y3-y4) - y2*(x3-x4) + (x3*y4 - x4*y3)
    # M12 = c2*(y3-y4) - y2*(x3-x4) + (x3*y4 - x4*y3) ... wait, signs.
    
    # Expansion along first row:
    # + c1 * det([[x2,y2,1],[x3,y3,1],[x4,y4,1]])
    # - x1 * det([[c2,y2,1],[c3,y3,1],[c4,y4,1]])
    # + y1 * det([[c2,x2,1],[c3,x3,1],[c4,x4,1]])
    # - 1  * det([[c2,x2,y2],[c3,x3,y3],[c4,x4,y4]])
    
    # 3x3 det [[a,b,c],[d,e,f],[g,h,i]] = a(ei-fh) - b(di-fg) + c(dh-eg)
    
    def det3(a,b,c,d,e,f,g,h,i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    
    D1 = det3(x2, y2, 1, x3, y3, 1, x4, y4, 1)
    D2 = det3(c2, y2, 1, c3, y3, 1, c4, y4, 1)
    D3 = det3(c2, x2, 1, c3, x3, 1, c4, x4, 1)
    D4 = det3(c2, x2, y2, c3, x3, y3, c4, x4, y4)
    
    return c1*D1 - x1*D2 + y1*D3 - 1*D4 == 0

def _valid_set(points):
    n = len(points)
    # Check distinctness
    pts = set(points)
    if len(pts) != n:
        return False
        
    # Check rational distances (integer squares)
    for i in range(n):
        for j in range(i+1, n):
            d2 = _dist_sq(points[i], points[j])
            if not _is_square(d2):
                return False
    
    # Check collinearity (no 3)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    
    # Check concyclicity (no 4)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Initial 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify base set is valid
    if not _valid_set(base_points):
        # Fallback to a simpler valid set if this one fails (shouldn't happen)
        # Try to find a valid 4-point set and expand?
        # For now, assume it's valid.
        pass
        
    best_points = base_points
    
    # Search for 6th point
    # We can restrict the search space.
    # The points are roughly in x: [-56, 16], y: [0, 63].
    # Let's search in a larger box, e.g., [-100, 100] x [-100, 100]
    # To be faster, we can check distances first.
    
    found_6 = False
    # Iterate through possible y, then x
    # Or random search? Systematic is better for completeness within budget.
    # But box might be large.
    # Let's try a systematic search in a moderate box first.
    
    # Pre-compute squared distances from new point (x,y) to base points
    # d_i^2 = (x-x_i)^2 + (y-y_i)^2 = S_i
    # S_i must be square.
    
    # We can also use the property that if we fix y, x must satisfy quadratic equations.
    # (x-x_i)^2 + (y-y_i)^2 = k_i^2
    # x^2 - 2x_i x + x_i^2 + y^2 - 2y_i y + y_i^2 = k_i^2
    # This is a system of equations.
    
    # For now, simple grid search with pruning.
    range_limit = 100
    # To speed up, we can sample y values and solve for x?
    # No, let's just iterate x, y in a smaller box first, then expand if time permits.
    
    # Actually, random search might find it faster if the box is large.
    # But systematic ensures we don't miss small integer solutions.
    
    # Let's try systematic search in [-150, 150]
    # That's 300x300 = 90,000 points.
    # For each point, check 5 distances. 450k ops. Very fast.
    # Then check collinearity (C(5,2)=10 pairs) and concyclicity (C(5,3)=10 triples).
    # 10 collinearity checks + 10 concyclicity checks per candidate.
    # 90k * 20 = 1.8M ops. Python can do this in < 1s.
    
    for y in range(-150, 151):
        if time.time() > deadline:
            break
        for x in range(-150, 151):
            if time.time() > deadline:
                break
            
            new_p = (x, y)
            
            # Quick check: distinct from existing
            if new_p in base_points:
                continue
            
            # Check distances
            valid_dist = True
            for p in base_points:
                d2 = _dist_sq(new_p, p)
                if not _is_square(d2):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Check collinearity
            valid_coll = True
            for i in range(len(base_points)):
                for j in range(i+1, len(base_points)):
                    if _are_collinear(base_points[i], base_points[j], new_p):
                        valid_coll = False
                        break
                if not valid_coll:
                    break
            
            if not valid_coll:
                continue
            
            # Check concyclicity
            valid_conc = True
            for i in range(len(base_points)):
                for j in range(i+1, len(base_points)):
                    for k in range(j+1, len(base_points)):
                        if _are_concyclic(base_points[i], base_points[j], base_points[k], new_p):
                            valid_conc = False
                            break
                    if not valid_conc:
                        break
                if not valid_conc:
                    break
            
            if valid_conc:
                # Found a 6-point set
                best_points = base_points + [new_p]
                found_6 = True
                break
        
        if found_6:
            break
    
    # If we found 6 points, try to find a 7th point
    if found_6:
        # Search for 7th point
        # This will be slower as we check 6 distances, C(6,2)=15 collinearity, C(6,3)=20 concyclicity
        # But we can restrict the search area or use random sampling.
        # Let's try a smaller systematic box first, then random.
        
        found_7 = False
        # Try systematic in smaller box first
        for y in range(-100, 101):
            if time.time() > deadline:
                break
            for x in range(-100, 101):
                if time.time() > deadline:
                    break
                
                new_p = (x, y)
                if new_p in best_points:
                    continue
                
                valid_dist = True
                for p in best_points:
                    d2 = _dist_sq(new_p, p)
                    if not _is_square(d2):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                
                valid_coll = True
                for i in range(len(best_points)):
                    for j in range(i+1, len(best_points)):
                        if _are_collinear(best_points[i], best_points[j], new_p):
                            valid_coll = False
                            break
                    if not valid_coll:
                        break
                if not valid_coll:
                    continue
                
                valid_conc = True
                for i in range(len(best_points)):
                    for j in range(i+1, len(best_points)):
                        for k in range(j+1, len(best_points)):
                            if _are_concyclic(best_points[i], best_points[j], best_points[k], new_p):
                                valid_conc = False
                                break
                        if not valid_conc:
                            break
                    if not valid_conc:
                        break
                if valid_conc:
                    best_points = best_points + [new_p]
                    found_7 = True
                    break
            if found_7:
                break
        
        # If not found systematically, try random search for 7th point
        if not found_7:
            for _ in range(100000):
                if time.time() > deadline:
                    break
                x = random.randint(-200, 200)
                y = random.randint(-200, 200)
                new_p = (x, y)
                if new_p in best_points:
                    continue
                
                valid_dist = True
                for p in best_points:
                    d2 = _dist_sq(new_p, p)
                    if not _is_square(d2):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                
                valid_coll = True
                for i in range(len(best_points)):
                    for j in range(i+1, len(best_points)):
                        if _are_collinear(best_points[i], best_points[j], new_p):
                            valid_coll = False
                            break
                    if not valid_coll:
                        break
                if not valid_coll:
                    continue
                
                valid_conc = True
                for i in range(len(best_points)):
                    for j in range(i+1, len(best_points)):
                        for k in range(j+1, len(best_points)):
                            if _are_concyclic(best_points[i], best_points[j], best_points[k], new_p):
                                valid_conc = False
                                break
                        if not valid_conc:
                            break
                    if not valid_conc:
                        break
                if valid_conc:
                    best_points = best_points + [new_p]
                    found_7 = True
                    break

    # Final verification
    if not _valid_set(best_points):
        # Fallback to 5 points if something went wrong
        best_points = base_points
        if not _valid_set(best_points):
            # Fallback to 4 points
            best_points = base_points[:4]

    # Format output
    # D=1, points are [x, y]
    # x, y are integers, so num=x, den=1
    out_points = []
    for x, y in best_points:
        out_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': out_points}

if __name__ == '__main__':
    # Quick test
    res = construct()
    print(res['D'], len(res['points']))
