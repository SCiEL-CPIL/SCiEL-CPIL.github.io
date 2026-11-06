import time
import random
from collections import defaultdict
from math import gcd

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return None
        
    g = gcd(abs(dx), abs(dy))
    if g != 0:
        dx //= g
        dy //= g
        
    # Normalize direction
    if dx < 0:
        dx = -dx
        dy = -dy
    elif dx == 0 and dy < 0:
        dy = -dy
        
    # Constant c for dy*x - dx*y = c
    c = dy * x1 - dx * y1
    
    return (dx, dy, c)

def count_4_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns (score, valid) where valid is False if any line has >= 5 points.
    """
    n = len(points)
    if n < 4:
        return 0, True
        
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = get_line_key(points[i], points[j])
            if key is not None:
                line_counts[key] += 1
                
    score = 0
    for count_val in line_counts.values():
        if count_val == 6: # C(4,2)
            score += 1
        elif count_val >= 10: # C(5,2)
            return score, False
            
    return score, True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial configuration: 4x4 grid
    # This gives score 10
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = count_4_lines(best_points)
    
    # Also try a 5x5 grid subset? 
    # A 5x5 grid has 25 points. We need 16.
    # Let's just stick to 4x4 as a strong seed and rely on search.
    
    current_points = [p[:] for p in best_points]
    current_score, _ = count_4_lines(current_points)
    
    # Helper to check validity and score
    def evaluate(pts):
        s, v = count_4_lines(pts)
        if not v:
            return -1000, False
        return s, True

    # Simulated Annealing / Local Search
    # We will try random moves and targeted moves.
    
    # Precompute some random offsets
    offsets = []
    for r in range(1, 20):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx == 0 and dy == 0: continue
                offsets.append((dx, dy))
                
    # Sort offsets by distance to try small moves first? 
    # Or just random. Let's mix.
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # 1. Random Local Move
        idx = random.randint(0, 15)
        orig_x, orig_y = current_points[idx]
        
        # Choose a move size
        # Smaller moves are more frequent
        move_radius = random.choice([1, 1, 1, 2, 2, 3, 5, 10])
        new_x = orig_x + random.randint(-move_radius, move_radius)
        new_y = orig_y + random.randint(-move_radius, move_radius)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check collision
        is_valid_move = True
        for k in range(16):
            if k != idx and current_points[k] == [new_x, new_y]:
                is_valid_move = False
                break
                
        if not is_valid_move:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score, is_valid = evaluate(current_points)
        
        if is_valid:
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                no_improve_count = 0
            else:
                # Revert if not better (or accept with prob for SA, but let's keep it simple first)
                # If score is equal, we might accept to explore, but it's risky for validity.
                # Let's revert if not strictly better to maintain "valid" state easily.
                current_points[idx] = [orig_x, orig_y]
                no_improve_count += 1
        else:
            current_points[idx] = [orig_x, orig_y]
            no_improve_count += 1
            
        # 2. Targeted Move: Try to complete 3-point lines
        if no_improve_count > 100:
             # Find lines with exactly 3 points
             # Reuse logic from count_4_lines but return points
             line_info = defaultdict(list)
             for i in range(16):
                 x1, y1 = current_points[i]
                 for j in range(i + 1, 16):
                     x2, y2 = current_points[j]
                     key = get_line_key(current_points[i], current_points[j])
                     if key:
                         line_info[key].append(i)
                         
             three_point_lines = [pts for pts in line_info.values() if len(pts) == 3]
             
             if three_point_lines:
                 # Pick a random 3-point line
                 line_pts = random.choice(three_point_lines)
                 p1_idx, p2_idx, p3_idx = line_pts
                 p1 = current_points[p1_idx]
                 p2 = current_points[p2_idx]
                 p3 = current_points[p3_idx]
                 
                 # Get line key
                 key = get_line_key(p1, p2)
                 dx, dy, c = key
                 
                 # We want to find an integer point (nx, ny) on this line
                 # such that moving some point to (nx, ny) creates a 4-point line.
                 # Ideally, we move a point NOT on this line to (nx, ny).
                 # And (nx, ny) must not be occupied.
                 
                 # Generate candidate positions on the line near the existing points
                 # Parametrize: x = x1 + t*dx, y = y1 + t*dy
                 # Since dx, dy are coprime, this generates all integer points on the line.
                 # We check t in a range around the existing points.
                 
                 # Find min/max t for existing points
                 # t = (x - x1) / dx if dx != 0 else (y - y1) / dy
                 # Let's just iterate t from -10 to 10 relative to p1
                 # Actually, better to iterate t such that the point is within bounds and close.
                 
                 candidates = []
                 # Determine a reference t for p1
                 # Let's just try t from -20 to 20
                 for t in range(-20, 21):
                     nx = p1[0] + t * dx
                     ny = p1[1] + t * dy
                     
                     if abs(nx) > 1000 or abs(ny) > 1000:
                         continue
                     
                     # Check if this point is one of the existing 3
                     if [nx, ny] in [p1, p2, p3]:
                         continue
                     
                     # Check if this point is occupied by any other point
                     occupied = False
                     for k in range(16):
                         if current_points[k] == [nx, ny]:
                             occupied = True
                             break
                     if occupied:
                         continue
                     
                     # This is a candidate position for the 4th point.
                     # We need to move SOME point to this position.
                     # Which point? Any point not on this line.
                     # Moving a point on the line would break the 3-point line structure or create 5.
                     # So we pick a random point NOT in line_pts.
                     
                     candidates.append((nx, ny))
                     
                 if candidates:
                     nx, ny = random.choice(candidates)
                     # Pick a point to move
                     # Indices not in line_pts
                     movable_indices = [i for i in range(16) if i not in line_pts]
                     if movable_indices:
                         idx_to_move = random.choice(movable_indices)
                         orig_x, orig_y = current_points[idx_to_move]
                         
                         current_points[idx_to_move] = [nx, ny]
                         new_score, is_valid = evaluate(current_points)
                         
                         if is_valid and new_score > current_score:
                             current_score = new_score
                             if current_score > best_score:
                                 best_score = current_score
                                 best_points = [p[:] for p in current_points]
                             no_improve_count = 0
                         else:
                             current_points[idx_to_move] = [orig_x, orig_y]
                             no_improve_count += 1
             
             # Reset no_improve counter occasionally or let it drive restarts
             if no_improve_count > 500:
                 # Perturb best solution
                 current_points = [p[:] for p in best_points]
                 for _ in range(5):
                     idx = random.randint(0, 15)
                     current_points[idx][0] += random.randint(-10, 10)
                     current_points[idx][1] += random.randint(-10, 10)
                     if abs(current_points[idx][0]) > 1000: current_points[idx][0] = 0
                     if abs(current_points[idx][1]) > 1000: current_points[idx][1] = 0
                 current_score, is_valid = evaluate(current_points)
                 if not is_valid:
                     current_points = [p[:] for p in best_points]
                     current_score = best_score
                 no_improve_count = 0

    return best_points
