import math
import random
import time
from collections import defaultdict

def line_hash(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y1)
    dy = y2 - y1
    dx = x2 - x1
    g = math.gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    elif dx_n == 0 and dy_n < 0:
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g = math.gcd(abs(a), abs(b), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def evaluate(points):
    n = len(points)
    if n != 16:
        return -1
    if len(set(map(tuple, points))) != 16:
        return -1
    line_counts = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_counts[key].append((i, j))
    
    score = 0
    for key, pairs in line_counts.items():
        pts_on_line = set()
        for i, j in pairs:
            pts_on_line.add(i)
            pts_on_line.add(j)
        k = len(pts_on_line)
        if k >= 5:
            return -1
        if k == 4:
            score += 1
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial candidate: 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    random.seed(42)
    
    def get_line_data(points):
        """
        Returns:
        line_to_pts: dict mapping line_hash to list of point indices
        """
        line_to_pts = defaultdict(list)
        for i in range(16):
            for j in range(i + 1, 16):
                key = line_hash(points[i], points[j])
                if i not in line_to_pts[key]:
                    line_to_pts[key].append(i)
                if j not in line_to_pts[key]:
                    line_to_pts[key].append(j)
        return line_to_pts

    def local_search(points, max_iters=2000):
        current_points = [p[:] for p in points]
        current_score = evaluate(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        
        temp = 1.0
        cooling = 0.995
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # 1. Get current line data
            line_to_pts = get_line_data(current_points)
            
            # Identify 3-point lines (candidates for completion)
            three_point_lines = []
            for key, pts in line_to_pts.items():
                if len(pts) == 3:
                    three_point_lines.append((key, pts))
            
            move_made = False
            
            # Strategy A: Try to complete a 3-point line
            if three_point_lines and random.random() < 0.5:
                key, pts = random.choice(three_point_lines)
                # The line is defined by any two points in pts
                p1, p2 = pts[0], pts[1]
                x1, y1 = current_points[p1]
                x2, y2 = current_points[p2]
                
                # Calculate direction
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                step_x = dx // g
                step_y = dy // g
                
                # Find the 3rd point to determine the segment
                p3 = pts[2]
                x3, y3 = current_points[p3]
                
                # We want to find a point P such that P, p1, p2, p3 are collinear.
                # Actually, any point on the line works.
                # Let's try extending from the ends.
                
                # Get sorted points on line by x (or y if vertical)
                sorted_pts = sorted(pts, key=lambda i: (current_points[i][0], current_points[i][1]))
                min_idx = sorted_pts[0]
                max_idx = sorted_pts[-1]
                
                candidates = []
                
                # Extend from min
                mx, my = current_points[min_idx]
                for k in range(1, 10):
                    nx = mx - k * step_x
                    ny = my - k * step_y
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        candidates.append([nx, ny])
                    
                # Extend from max
                mx, my = current_points[max_idx]
                for k in range(1, 10):
                    nx = mx + k * step_x
                    ny = my + k * step_y
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        candidates.append([nx, ny])
                
                if candidates:
                    new_pt = random.choice(candidates)
                    # Pick a point to move. Ideally one that is NOT on this line.
                    # Or one that is "free".
                    # Let's pick a random point not on the line.
                    available_indices = [i for i in range(16) if i not in pts]
                    if available_indices:
                        idx = random.choice(available_indices)
                        old_pt = current_points[idx][:]
                        
                        # Check if new_pt is already taken
                        if new_pt not in current_points:
                            current_points[idx] = new_pt
                            new_score = evaluate(current_points)
                            
                            if new_score == -1:
                                current_points[idx] = old_pt
                                no_improve += 1
                            else:
                                delta = new_score - current_score
                                if delta > 0:
                                    current_score = new_score
                                    no_improve = 0
                                    if current_score > best_local_score:
                                        best_local_score = current_score
                                        best_local = [p[:] for p in current_points]
                                else:
                                    if delta == 0:
                                        current_score = new_score
                                        no_improve += 1
                                    else:
                                        prob = math.exp(delta / temp)
                                        if random.random() < prob:
                                            current_score = new_score
                                        else:
                                            current_points[idx] = old_pt
                                        no_improve += 1
                                move_made = True
            
            if not move_made:
                # Strategy B: Random move
                idx = random.randint(0, 15)
                old_pt = current_points[idx][:]
                
                step = random.randint(1, 5)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
                
                if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                    continue
                
                new_pt = [new_x, new_y]
                
                if new_pt in current_points:
                    continue
                
                current_points[idx] = new_pt
                new_score = evaluate(current_points)
                
                if new_score == -1:
                    current_points[idx] = old_pt
                    no_improve += 1
                else:
                    delta = new_score - current_score
                    if delta > 0:
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local = [p[:] for p in current_points]
                    else:
                        if delta == 0:
                            current_score = new_score
                            no_improve += 1
                        else:
                            prob = math.exp(delta / temp)
                            if random.random() < prob:
                                current_score = new_score
                            else:
                                current_points[idx] = old_pt
                            no_improve += 1
                
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
                
            if no_improve > 1000:
                break
                
        return best_local, best_local_score

    # Run local search
    pts, score = local_search(initial_points, max_iters=5000)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in pts]
        
    # Try a few random restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        # Perturb the grid
        pts = [p[:] for p in initial_points]
        for i in range(16):
            if random.random() < 0.5:
                pts[i] = [pts[i][0] + random.randint(-3, 3), pts[i][1] + random.randint(-3, 3)]
        
        s = evaluate(pts)
        if s < 0:
            continue
            
        pts, score = local_search(pts, max_iters=2000)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]

    return best_points
