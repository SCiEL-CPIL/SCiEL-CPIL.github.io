import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds to be safe
    
    def evaluate(points):
        counts = {}
        n = 30
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area:
                        counts[area] = counts.get(area, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    def make_line_config(num_lines, counts_per_line, spacing=1, y_start=0):
        points = []
        for l in range(num_lines):
            y = y_start + l
            for i in range(counts_per_line[l]):
                x = i * spacing
                points.append((x, y))
        # Pad or trim to 30
        if len(points) < 30:
            # Add random points? Or just return what we have? 
            # Must be exactly 30.
            # If we define counts to sum to 30, this is fine.
            pass
        return points[:30]

    candidates = []
    
    # Grids
    def centered_grid(rows, cols):
        points = []
        start_x = -((cols - 1) * 2) // 2
        start_y = -((rows - 1) * 2) // 2
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * 2
                y = start_y + i * 2
                points.append((x, y))
        return points

    candidates.append(centered_grid(6, 5))
    candidates.append(centered_grid(5, 6))
    candidates.append(centered_grid(3, 10)) # 3x10
    candidates.append(centered_grid(10, 3)) # 10x3
    
    # 2-Line Configs
    # 15 on y=0, 15 on y=1
    p2 = make_line_config(2, [15, 15], 1, 0)
    candidates.append(p2)
    
    # 3-Line Configs
    # 10 on y=0, 10 on y=1, 10 on y=2
    p3 = make_line_config(3, [10, 10, 10], 1, 0)
    candidates.append(p3)
    
    # 4-Line Configs
    # 8, 8, 8, 6
    p4 = make_line_config(4, [8, 8, 8, 6], 1, 0)
    candidates.append(p4)
    
    # 5-Line Configs
    # 6, 6, 6, 6, 6
    p5 = make_line_config(5, [6, 6, 6, 6, 6], 1, 0)
    candidates.append(p5)

    # Add some random points to break symmetry in line configs?
    # The local search will do that.

    best_score = 0
    best_points = candidates[0]
    
    for cand in candidates:
        score = evaluate(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]

    # Local Search
    current_points = best_points[:]
    current_score = best_score
    random.seed(42)
    
    stuck_count = 0
    max_stuck = 5000
    
    while time.time() < deadline:
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        if random.random() < 0.8:
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        else:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
            
        new_point = (new_x, new_y)
        if new_point == (old_x, old_y):
            continue
        if new_point in current_points:
            continue
            
        current_points[idx] = new_point
        new_score = evaluate(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            stuck_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            current_points[idx] = (old_x, old_y)
            stuck_count += 1
            
            if stuck_count > max_stuck:
                current_points = best_points[:]
                current_score = best_score
                stuck_count = 0
                for _ in range(5):
                    i = random.randint(0, 29)
                    x, y = current_points[i]
                    nx = x + random.randint(-1, 1)
                    ny = y + random.randint(-1, 1)
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) != (x, y) and (nx, ny) not in current_points:
                             current_points[i] = (nx, ny)
                current_score = evaluate(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]

    return best_points
