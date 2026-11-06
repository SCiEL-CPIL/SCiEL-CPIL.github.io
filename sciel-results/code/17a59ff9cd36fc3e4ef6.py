import random
import time
import math
from itertools import combinations

def count_lines(points):
    """
    Returns (num_4_lines, num_5_plus_lines, lines_dict) for a set of 16 points.
    """
    n = len(points)
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
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
            
            C = -dy * x1 + dx * y1
            key = (dx, dy, C)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    num_4 = 0
    num_5_plus = 0
    for key, pts in lines.items():
        size = len(pts)
        if size == 4:
            num_4 += 1
        elif size >= 5:
            num_5_plus += 1
            
    return num_4, num_5_plus, lines

def is_valid(points):
    if len(set(map(tuple, points))) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, num_5_plus, _ = count_lines(points)
    if num_5_plus > 0:
        return False
    return True

def score(points):
    if not is_valid(points):
        return -1000
    num_4, _, _ = count_lines(points)
    return num_4

def generate_candidates_on_line(p1, p2):
    """
    Generate candidate integer positions on the line defined by p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    candidates = []
    
    if dx == 0:
        # Vertical
        y_min = min(y1, y2) - 15
        y_max = max(y1, y2) + 15
        for y in range(y_min, y_max + 1):
            if abs(y) > 1000: continue
            candidates.append((x1, y))
    elif dy == 0:
        # Horizontal
        x_min = min(x1, x2) - 15
        x_max = max(x1, x2) + 15
        for x in range(x_min, x_max + 1):
            if abs(x) > 1000: continue
            candidates.append((x, y1))
    else:
        g = math.gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        
        max_k = 20
        for k in range(-max_k, max_k + 1):
            x = x1 + k * dx_n
            y = y1 + k * dy_n
            if abs(x) > 1000 or abs(y) > 1000:
                continue
            candidates.append((x, y))
            
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in points]
    best_score = score(best_points)
    
    current_points = [p[:] for p in points]
    current_num_4, current_num_5, current_lines = count_lines(current_points)
    current_score = current_num_4 if current_num_5 == 0 else -1000
    
    temp = 1.0
    cooling = 0.999
    min_temp = 0.01
    R = 5
    
    while time.time() < deadline:
        move_made = False
        new_point = None
        new_idx = -1
        old_point = None
        
        # 1. Handle Invalidity
        if current_num_5 > 0:
            target_line_pts = None
            for key, pts in current_lines.items():
                if len(pts) >= 5:
                    target_line_pts = pts
                    break
            
            if target_line_pts:
                idx = random.choice(list(target_line_pts))
                old_point = [current_points[idx][0], current_points[idx][1]]
                new_x = old_point[0] + random.randint(-20, 20)
                new_y = old_point[1] + random.randint(-20, 20)
                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                    if [new_x, new_y] not in current_points:
                        new_point = [new_x, new_y]
                        new_idx = idx
                        move_made = True
        
        # 2. Targeted Improvement (if valid)
        if not move_made and current_num_5 == 0:
            three_lines = [key for key, pts in current_lines.items() if len(pts) == 3]
            
            if three_lines and random.random() < 0.8:
                target_key = random.choice(three_lines)
                target_pts = current_lines[target_key]
                
                candidates_for_move = [i for i in range(16) if i not in target_pts]
                
                if candidates_for_move:
                    idx = random.choice(candidates_for_move)
                    p_list = list(target_pts)
                    p1 = current_points[p_list[0]]
                    p2 = current_points[p_list[1]]
                    
                    candidates = generate_candidates_on_line(p1, p2)
                    
                    valid_cands = []
                    for cand in candidates:
                        collides = False
                        for k in range(16):
                            if k == idx: continue
                            if current_points[k][0] == cand[0] and current_points[k][1] == cand[1]:
                                collides = True
                                break
                        if not collides:
                            valid_cands.append(cand)
                    
                    if valid_cands:
                        new_point = random.choice(valid_cands)
                        new_idx = idx
                        old_point = [current_points[idx][0], current_points[idx][1]]
                        move_made = True
        
        # 3. Random Move
        if not move_made:
            idx = random.randint(0, 15)
            old_point = [current_points[idx][0], current_points[idx][1]]
            new_x = old_point[0] + random.randint(-R, R)
            new_y = old_point[1] + random.randint(-R, R)
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                if [new_x, new_y] not in current_points:
                    new_point = [new_x, new_y]
                    new_idx = idx
                    move_made = True
        
        if not move_made:
            continue
            
        current_points[new_idx] = new_point
        
        new_num_4, new_num_5, new_lines = count_lines(current_points)
        new_score = new_num_4 if new_num_5 == 0 else -1000
        
        if new_score >= current_score:
            current_score = new_score
            current_num_4 = new_num_4
            current_num_5 = new_num_5
            current_lines = new_lines
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            delta = new_score - current_score
            if delta > -100:
                prob = math.exp(delta / temp)
            else:
                prob = 0.0
                
            if random.random() < prob:
                current_score = new_score
                current_num_4 = new_num_4
                current_num_5 = new_num_5
                current_lines = new_lines
            else:
                current_points[new_idx] = old_point
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp

    return best_points
