import random
import time
import math
from itertools import combinations

def _count_lines_exactly_4(points):
    n = len(points)
    if n != 16:
        return 0, False
        
    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            # Line equation: A(x - x1) + B(y - y1) = 0
            # A = y1 - y2
            # B = x2 - x1
            A = y1 - y2
            B = x2 - x1
            # C = -(A*x1 + B*y1)
            # Normalize
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(A*x1 + B*y1))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C = -(A*x1 + B*y1) // g
            
            # Sign normalization
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0 and B < 0:
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C

            key = (A, B, C)
            
            # Count points on this line
            # We already know i and j are on it.
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if A*(xk - x1) + B*(yk - y1) == 0:
                    count += 1
                    
            # Store the count for this line
            # We only need to know the max count for each line?
            # No, the count is the same for all pairs on the same line.
            # So we can just update the dict with the count.
            # If it's already there, it should be the same.
            if key not in lines:
                lines[key] = count
            # else: lines[key] should be same as count
            
    score = 0
    for key, count in lines.items():
        if count >= 5:
            return 0, False
        elif count == 4:
            score += 1
            
    return score, True

def construct():
    start_time = time.time()
    max_time = 140.0
    
    # Initial best
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _count_lines_exactly_4(best_points)
    
    # Local Search
    current_points = best_points
    current_score = best_score
    
    while time.time() - start_time < max_time:
        new_points = [p[:] for p in current_points]
        move_type = random.random()
        
        if move_type < 0.7:
            idx = random.randint(0, 15)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_points[idx][0] += dx
            new_points[idx][1] += dy
            new_points[idx][0] = max(-1000, min(1000, new_points[idx][0]))
            new_points[idx][1] = max(-1000, min(1000, new_points[idx][1]))
        else:
            idx = random.randint(0, 15)
            new_points[idx][0] = random.randint(-1000, 1000)
            new_points[idx][1] = random.randint(-1000, 1000)
            
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
            
        s, v = _count_lines_exactly_4(new_points)
        if v:
            if s > current_score:
                current_score = s
                current_points = new_points
                if current_score >= 15:
                    break
            elif s == current_score and random.random() < 0.2:
                current_points = new_points
                
    return current_points
