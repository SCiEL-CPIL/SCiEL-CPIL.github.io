import time
import random
from math import gcd
from collections import defaultdict

def get_line_key(p1, p2):
    """
    Returns a canonical key for the line passing through p1 and p2.
    Key is (a, b, c) for ax + by = c, normalized.
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    if a == 0 and b == 0:
        return None
        
    # Normalize a, b, c
    g = gcd(abs(a), abs(b))
    if g > 0:
        a //= g
        b //= g
        c //= g
        
    # Ensure canonical sign: first non-zero of a, b is positive
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def score_configuration(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    # Map line key to set of point indices on that line
    line_to_points = defaultdict(set)
    
    # Iterate over all pairs
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_to_points[key].add(i)
            line_to_points[key].add(j)
            
    score = 0
    for key, pts in line_to_points.items():
        count = len(pts)
        if count >= 5:
            return -1
        if count == 4:
            score += 1
            
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Standard 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    candidates.append(grid)
    
    # 2. Shifted 4x4 grid (to break symmetries)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            shifted = [[x + dx, y + dy] for x in range(4) for y in range(4)]
            # Check bounds
            if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in shifted):
                candidates.append(shifted)
                
    # 3. Some random configurations (sparse)
    random.seed(42)
    for _ in range(5):
        pts = []
        while len(pts) < 16:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if [x, y] not in pts:
                pts.append([x, y])
        candidates.append(pts)
        
    best_score = -1
    best_points = None
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        s = score_configuration(cand)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in cand]
            
    if best_points is None:
        return [[0,0]] * 16 # Fallback, should not happen
        
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Local Search
    # Strategy: Try to move each point to a nearby location.
    # Use a window that expands if stuck.
    window_size = 5
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        improved = False
        
        for i in range(16):
            if time.time() > deadline:
                break
                
            orig_x, orig_y = current_points[i]
            
            # Try moves in a window around the current point
            # To save time, we can be smarter about which moves to try.
            # But for now, let's try all points in a box of size window_size.
            # We can optimize by only trying moves that might create 4-point lines.
            # For simplicity, we try a subset of moves.
            
            # Adaptive window: if stuck, increase window
            if no_improve_count > max_no_improve // 2:
                w = window_size * 2
            else:
                w = window_size
                
            found_move = False
            for dx in range(-w, w + 1):
                if found_move:
                    break
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                        
                    # Check distinctness
                    is_distinct = True
                    for j in range(16):
                        if j != i and current_points[j] == [new_x, new_y]:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                        
                    # Try the move
                    current_points[i] = [new_x, new_y]
                    new_score = score_configuration(current_points)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        found_move = True
                        no_improve_count = 0
                        # Reset window if we found a good move? Or keep it?
                        # Let's keep it to explore the neighborhood.
                    else:
                        # Revert
                        current_points[i] = [orig_x, orig_y]
                        
            if not improved:
                no_improve_count += 1
            else:
                # If we improved, we might want to restart the loop to re-evaluate all points
                # Or just continue. Let's continue but maybe break inner loop?
                # Breaking inner loop ensures we don't make multiple moves in one pass without re-evaluating.
                pass
                
        # If no improvement in a full pass
        if not improved:
            no_improve_count += 1
            # If stuck for a long time, try a random jump
            if no_improve_count > max_no_improve:
                # Random restart or jump
                i = random.randint(0, 15)
                orig = current_points[i][:]
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    new_x = random.randint(-1000, 1000)
                    new_y = random.randint(-1000, 1000)
                    if [new_x, new_y] in current_points[:i] + current_points[i+1:]:
                        continue
                    current_points[i] = [new_x, new_y]
                    s = score_configuration(current_points)
                    if s > current_score:
                        current_score = s
                        best_score = s
                        best_points = [p[:] for p in current_points]
                        no_improve_count = 0
                        break
                current_points[i] = orig
                no_improve_count = 0 # Reset counter after jump attempt
                
            # Increase window size occasionally
            if no_improve_count % 50 == 0:
                window_size = min(window_size + 2, 20)
                
    return best_points
