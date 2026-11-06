import random
import math

def construct():
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def score(points_list):
        pts = points_list
        n = len(pts)
        # Map points to indices for fast lookup? 
        # We just need to count lines with exactly 4 points.
        # A line is defined by 2 points. If 2 other points are on it, it's a 4-point line.
        # If 3 other points are on it, it's a 5-point line (invalid).
        
        lines = set()
        invalid = False
        
        # Optimization: Pre-compute vector differences? No, just loop.
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                
                # Normalize direction? Not needed for collinearity check, 
                # but needed for unique line ID if we used that.
                # We use the set of points on the line as the ID.
                
                count = 2
                line_pts = [i, j]
                
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    # Cross product
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
                        line_pts.append(k)
                        
                if count > 4:
                    return -1
                elif count == 4:
                    # Create a canonical ID for this line
                    # Sort the indices
                    line_pts.sort()
                    lines.add(tuple(line_pts))
                    
        return len(lines)

    best_points = [p[:] for p in points]
    best_score = score(best_points)
    
    # Simulated Annealing
    temp = 10.0
    cool_rate = 0.99
    min_temp = 0.1
    
    # Search box
    min_coord = -5
    max_coord = 10
    
    while temp > min_temp:
        improved = False
        # Try to improve by moving one point
        idx = random.randint(0, 15)
        old_point = best_points[idx]
        
        # Best move selection
        best_candidate = None
        best_candidate_score = best_score
        
        # Try several random moves
        for _ in range(20):
            new_x = random.randint(min_coord, max_coord)
            new_y = random.randint(min_coord, max_coord)
            new_point = [new_x, new_y]
            
            if any(p == new_point for p in best_points if p != old_point):
                continue
            
            candidate = best_points[:]
            candidate[idx] = new_point
            
            s = score(candidate)
            
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = candidate
            elif s == best_candidate_score and best_candidate is None:
                # Keep track of equal score moves for SA
                pass
        
        if best_candidate is not None:
            best_points = best_candidate
            best_score = best_candidate_score
            improved = True
        else:
            # Simulated Annealing: accept worse or equal moves with probability
            # We need to store the best "worse" move? Or just pick a random one?
            # Let's pick a random valid move and accept with prob exp((new-old)/temp)
            new_x = random.randint(min_coord, max_coord)
            new_y = random.randint(min_coord, max_coord)
            new_point = [new_x, new_y]
            
            if not any(p == new_point for p in best_points if p != old_point):
                candidate = best_points[:]
                candidate[idx] = new_point
                s = score(candidate)
                
                if s >= 0: # Valid
                    delta = s - best_score
                    if delta >= 0 or random.random() < math.exp(delta / temp):
                        best_points = candidate
                        best_score = s
                        improved = True
        
        temp *= cool_rate
        if not improved:
            # Force a larger jump
            temp *= 0.9

    return [[int(x), int(y)] for x, y in best_points]
