import random
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Map from line key to count of points
    # Line key: (A, B, C) where A*x + B*y + C = 0, normalized
    # To avoid floating point, use integer coefficients.
    # For points (x1, y1) and (x2, y2):
    # A = y1 - y2
    # B = x2 - x1
    # C = x1*y2 - x2*y1
    # Normalize: gcd(A, B, C)
    
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # If x1 == x2, vertical line
            if x1 == x2:
                key = (1, 0, -x1)
            elif y1 == y2:
                key = (0, 1, -y1)
            else:
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                
                # Normalize by GCD
                g = __import__('math').gcd(A, __import__('math').gcd(B, C))
                if g < 0:
                    g = -g
                A //= g
                B //= g
                C //= g
                
                # Ensure a consistent sign (e.g., A > 0 or A==0 and B > 0)
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
            
            lines[key] = lines.get(key, 0) + 1
    
    ordinary = 0
    for count in lines.values():
        if count == 1:
            ordinary += 1
            
    return ordinary

def construct():
    start_time = time.time()
    timeout = 140  # 10s margin
    
    # Initial random points
    # Use a small range to start
    best_points = None
    best_score = float('inf')
    
    # Try a few random restarts
    for restart in range(20):
        points = []
        while len(points) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            p = (x, y)
            if p not in points:
                points.append(p)
        
        # Ensure not all collinear
        # Check if all points are on the same line
        def all_collinear(pts):
            if len(pts) < 3: return True
            (x1, y1), (x2, y2), (x3, y3) = pts[0], pts[1], pts[2]
            if (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1):
                # Check all
                for i in range(3, len(pts)):
                    xi, yi = pts[i]
                    if (y2 - y1) * (xi - x1) != (y3 - y1) * (x2 - x1):
                        return False
                return True
            return False
        
        if all_collinear(points):
            # Move one point
            points[0] = (points[0][0] + 1, points[0][1])
            # Re-check distinctness
            if points[0] in points[1:]:
                 points[0] = (points[0][0] + 2, points[0][1])
        
        score = count_ordinary_lines(points)
        
        # Hill climbing
        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            for i in range(12):
                x, y = points[i]
                # Try neighbors
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0: continue
                        nx, ny = x + dx, y + dy
                        # Check bounds and distinctness
                        if abs(nx) > 100 or abs(ny) > 100: continue
                        new_p = (nx, ny)
                        if new_p in points and new_p != (x,y): continue
                        # Note: if new_p is already in points, it's a collision.
                        # We must ensure distinctness.
                        # If new_p is in points, skip.
                        if any(p == new_p for p in points):
                            continue
                        
                        new_points = points.copy()
                        new_points[i] = new_p
                        
                        # Check all collinear
                        if all_collinear(new_points):
                            continue
                            
                        new_score = count_ordinary_lines(new_points)
                        if new_score < score:
                            score = new_score
                            points = new_points
                            improved = True
                            # Break to restart outer loop? Or just continue?
                            # Restarting the neighbor search for this point is good.
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if score <= 6:
                break
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
            
        if best_score <= 6:
            break
            
    # Convert to list of lists
    return [[x, y] for x, y in best_points]
