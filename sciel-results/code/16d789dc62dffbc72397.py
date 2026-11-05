import math
import random
import time
from collections import defaultdict

def count_four_lines(points):
    n = len(points)
    line_counts = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                slope_key = (0, 1)
            elif dy == 0:
                slope_key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                slope_key = (dx_n, dy_n)
            
            c = slope_key[1] * x1 - slope_key[0] * y1
            line_id = (slope_key[0], slope_key[1], c)
            line_counts[line_id].append(i)
    
    four_count = 0
    for line_id, pts in line_counts.items():
        if len(pts) >= 5:
            return -1
        if len(pts) == 4:
            four_count += 1
    
    return four_count

def construct():
    start_time = time.time()
    time_limit = 30  # seconds
    
    # Start with 4x4 grid
    best = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_four_lines(best)
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(16):
            if time.time() - start_time >= time_limit:
                break
            orig_x, orig_y = best[i]
            # Try random nearby moves
            for _ in range(50):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = orig_x + dx, orig_y + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if any(best[j][0] == nx and best[j][1] == ny for j in range(16) if j != i):
                    continue
                new_pts = [list(p) for p in best]
                new_pts[i] = [nx, ny]
                score = count_four_lines(new_pts)
                if score > best_score:
                    best = new_pts
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try a larger random perturbation
            for _ in range(10):
                if time.time() - start_time >= time_limit:
                    break
                i = random.randint(0, 15)
                j = random.randint(0, 15)
                if i == j:
                    continue
                # Swap or move
                new_pts = [list(p) for p in best]
                nx = random.randint(-10, 10)
                ny = random.randint(-10, 10)
                if any(new_pts[k][0] == nx and new_pts[k][1] == ny for k in range(16) if k != i):
                    continue
                new_pts[i] = [nx, ny]
                score = count_four_lines(new_pts)
                if score > best_score:
                    best = new_pts
                    best_score = score
                    improved = True
                    break
    
    return best
