import random
import time
import math
from itertools import combinations

def count_lines(points):
    """
    Returns (num_4_lines, num_5_plus_lines) for a set of 16 points.
    """
    n = len(points)
    # Map from line key to list of points on that line
    # Line key: normalized (a, b, c) for ax + by + c = 0
    # Since points are integers, we can use (dx, dy) normalized by gcd.
    # Or just store the set of points for each line.
    
    lines = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
            # Ensure unique representation for direction
            # Actually, (dx, dy) and (-dx, -dy) are same line direction.
            # We normalized such that dx > 0 or (dx==0 and dy>0).
            
            # Calculate line key.
            # Line passing through (x1, y1) with direction (dx, dy).
            # Normal vector is (-dy, dx).
            # Equation: -dy * x + dx * y = C
            # C = -dy * x1 + dx * y1
            C = -dy * x1 + dx * y1
            
            # Normalize C? 
            # Since dx, dy are coprime, C is unique for the line.
            # However, different (dx, dy) could represent the same line if not careful?
            # No, the direction is unique. The line is unique.
            
            key = (dx, dy, C)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    num_4 = 0
    num_5_plus = 0
    for key, pts in lines.items():
        if len(pts) == 4:
            num_4 += 1
        elif len(pts) >= 5:
            num_5_plus += 1
            
    return num_4, num_5_plus

def is_valid(points):
    # Check bounds and distinctness
    if len(set(map(tuple, points))) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    # Check no 5 collinear
    _, num_5_plus = count_lines(points)
    if num_5_plus > 0:
        return False
    return True

def score(points):
    if not is_valid(points):
        return -1000
    num_4, _ = count_lines(points)
    return num_4

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial configuration: 4x4 grid
    # Shift to center to allow movement
    points = [[x, y] for x in range(4) for y in range(4)]
    # Center it around 0
    points = [[x-1.5, y-1.5] for x, y in points] # Non-integer? 
    # Task requires integer points.
    # Let's use 0..3
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in points]
    best_score = score(best_points)
    
    # Simulated Annealing
    current_points = [p[:] for p in points]
    current_score = best_score
    
    temp = 1.0
    cooling = 0.999
    
    # Define move radius
    # We can move points within a box of size R
    R = 5
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Generate new position
        # Random integer in range
        new_x = old_x + random.randint(-R, R)
        new_y = old_y + random.randint(-R, R)
        
        # Keep within bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check if point already exists
        new_point = [new_x, new_y]
        if new_point in current_points:
            continue
            
        # Apply move
        current_points[idx] = new_point
        
        new_score = score(current_points)
        
        # Acceptance criteria
        if new_score >= current_score:
            # Always accept improvements or equal scores (to explore)
            # But if new_score is -1000 (invalid), we should reject unless we want to escape?
            # If invalid, score is -1000.
            if new_score < 0:
                # Reject invalid moves mostly, but sometimes accept to escape?
                # If we accept invalid, we might get stuck.
                # Let's only accept valid moves or moves that improve.
                # If current is valid and new is invalid, reject.
                # If current is invalid and new is valid, accept.
                if current_score >= 0 and new_score < 0:
                    # Reject
                    current_points[idx] = [old_x, old_y]
                    # Revert
                    pass
                else:
                    # Accept
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
            else:
                # Both valid or both invalid?
                # If both valid, accept if new >= old
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    # Worse valid score. Accept with probability exp((new-old)/temp)
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / temp):
                        current_score = new_score
                    else:
                        # Revert
                        current_points[idx] = [old_x, old_y]
        else:
            # New score is worse (and valid)
            delta = new_score - current_score
            if random.random() < math.exp(delta / temp):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
                
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return best_points
