import random
import math
from math import gcd

def construct():
    # Initial configuration: 4x4 grid
    # This is a known good starting point with 10 four-point lines
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_score(points_list):
        """
        Returns (score, is_valid).
        Score is number of lines with exactly 4 points.
        If any line has >= 5 points, is_valid is False.
        """
        n = len(points_list)
        pts = [tuple(p) for p in points_list]
        if len(set(pts)) != n:
            return -float('inf'), False
            
        line_counts = {}
        
        # Iterate over all pairs to define lines
        for i in range(n):
            for j in range(i + 1, n):
                p1 = pts[i]
                p2 = pts[j]
                # Normalize the line equation: Ax + By + C = 0
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = -(A * p1[0] + B * p1[1])
                
                # Normalize the tuple (A, B, C) to handle scaling
                g = gcd(gcd(abs(A), abs(B)), abs(C))
                if g == 0:
                    continue
                A //= g
                B //= g
                C //= g
                
                # Ensure a consistent sign (e.g., first non-zero is positive)
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                key = (A, B, C)
                
                if key not in line_counts:
                    line_counts[key] = set()
                
                line_counts[key].add(i)
                line_counts[key].add(j)
                
        score = 0
        valid = True
        for key, point_indices in line_counts.items():
            count = len(point_indices)
            if count >= 5:
                valid = False
                return -float('inf'), False
            elif count == 4:
                score += 1
        
        return score, valid

    # Simulated Annealing Parameters
    start_temp = 1.0
    end_temp = 0.001
    cooling_rate = 0.9995
    max_iterations = 10000
    
    current_points = [p[:] for p in points]
    current_score, _ = get_score(current_points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    temp = start_temp
    
    # Search bounds. The solution is likely within a small grid.
    min_coord = -10
    max_coord = 10
    
    for it in range(max_iterations):
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.8:
            # Move one point
            idx = random.randint(0, 15)
            new_x = random.randint(min_coord, max_coord)
            new_y = random.randint(min_coord, max_coord)
            
            # Check distinctness quickly
            new_point = (new_x, new_y)
            if new_point in [tuple(p) for p in current_points if p != current_points[idx]]:
                continue
                
            old_point = current_points[idx][:]
            current_points[idx] = [new_x, new_y]
            
            new_score, valid = get_score(current_points)
            
            if valid:
                delta = new_score - current_score
                if delta > 0:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    # Accept bad moves with probability exp(delta / temp)
                    exponent = min(50.0, max(-50.0, delta / temp)) # Clamp to avoid under/overflow
                    if random.random() < math.exp(exponent):
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current_points]
                    else:
                        # Revert
                        current_points[idx] = old_point
            else:
                # Revert
                current_points[idx] = old_point
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            temp_point = current_points[i][:]
            current_points[i] = current_points[j][:]
            current_points[j] = temp_point
            
            new_score, valid = get_score(current_points)
            
            if valid:
                delta = new_score - current_score
                if delta > 0:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    exponent = min(50.0, max(-50.0, delta / temp))
                    if random.random() < math.exp(exponent):
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current_points]
                    else:
                        # Revert
                        current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
            else:
                # Revert
                current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
                
        temp *= cooling_rate
        if temp < end_temp:
            break
            
    return [[int(x), int(y)] for x, y in best_points]
