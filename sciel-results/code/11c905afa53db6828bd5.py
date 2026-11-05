import random
import time
import math
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def score(points):
        """
        Returns the number of lines containing exactly 4 points.
        Returns -1 if any 5 points are collinear or if points are not distinct.
        """
        n = len(points)
        if n != 16:
            return -1
        
        # Check distinct
        if len(set(map(tuple, points))) != 16:
            return -1
            
        # Check bounds
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return -1
                
        lines = defaultdict(list)
        
        # Normalize line representation: (A, B, C) for Ax + By + C = 0
        # To avoid floating point issues, use integers.
        # Line through (x1, y1) and (x2, y2):
        # A = y1 - y2
        # B = x2 - x1
        # C = x1*y2 - x2*y1
        # Normalize by GCD and sign
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                A = y1 - y2
                B = x2 - x1
                C = x1*y2 - x2*y1
                
                if A == 0 and B == 0:
                    continue # Should not happen if distinct
                
                # Normalize
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                
                # Ensure consistent sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                lines[(A, B, C)].append(i)
                
        count_4 = 0
        for line_pts in lines.values():
            if len(line_pts) >= 5:
                return -1
            if len(line_pts) == 4:
                count_4 += 1
                
        return count_4

    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = score(best_points)
    
    # Also try a few other random starts
    candidates = []
    candidates.append(best_points)
    
    for _ in range(5):
        pts = []
        while len(pts) < 16:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in pts:
                pts.append([x, y])
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
        candidates.append(pts)

    # Local Search
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random small perturbation
        # Sometimes large jump to escape
        if random.random() < 0.1:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
        else:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Keep in bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness quickly
        new_point = [new_x, new_y]
        is_distinct = True
        for k in range(16):
            if k == idx:
                continue
            if current_points[k] == new_point:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Update point
        current_points[idx] = new_point
        new_score = score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept with probability exp((new_score - current_score) / T)
            if new_score == -1:
                # Invalid, revert
                current_points[idx] = [old_x, old_y]
            else:
                delta = new_score - current_score
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_points[idx] = [old_x, old_y]
                    
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return best_points
