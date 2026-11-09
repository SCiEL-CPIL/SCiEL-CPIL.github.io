# EVOLVE-BLOCK-START
import math
import random
import time

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    ordinary_keys = []
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
            ordinary_keys.append(key)
            
    return ordinary, ordinary_keys, lines

def is_valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in points):
        return False
    # Check not all collinear
    (x0, y0) = points[0]
    (x1, y1) = points[1]
    (x2, y2) = points[2]
    if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
        all_collinear = True
        for i in range(3, 12):
            (x3, y3) = points[i]
            if (x1 - x0) * (y3 - y0) != (y1 - y0) * (x3 - x0):
                all_collinear = False
                break
        if all_collinear:
            return False
    return True

def get_line_intersection(p1, p2, p3, p4):
    """Returns integer point if lines p1p2 and p3p4 intersect at integer coords, else None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Line 1: a1*x + b1*y = c1
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    
    # Line 2: a2*x + b2*y = c2
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None # Parallel or coincident
        
    x_num = c1 * b2 - c2 * b1
    y_num = a1 * c2 - a2 * c1
    
    if x_num % det != 0 or y_num % det != 0:
        return None
        
    x = x_num // det
    y = y_num // det
    
    if abs(x) > 10**9 or abs(y) > 10**9:
        return None
        
    return [x, y]

def construct():
    best_points = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 115
    
    # Initial set: Random points
    # We will use multiple restarts with different initializations
    
    for restart in range(100):
        # Initialization: Start with a grid-like structure to have many collinearities
        # 3x4 grid
        points = [[x, y] for x in range(4) for y in range(3)]
        
        # Randomly perturb to break symmetries and create diverse starting points
        for i in range(12):
            # Small perturbation
            points[i][0] += random.randint(-2, 2)
            points[i][1] += random.randint(-2, 2)
            
        # Ensure distinctness
        for i in range(12):
            if i > 0:
                while points[i] in points[:i]:
                    points[i][0] += random.randint(-1, 1)
                    points[i][1] += random.randint(-1, 1)
        
        if not is_valid(points):
            continue

        current_points = [p[:] for p in points]
        current_score, _, _ = count_ordinary_lines(current_points)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            current_score, ordinary_keys, lines = count_ordinary_lines(current_points)
            
            if not ordinary_keys:
                break
                
            # Pick a random ordinary line to fix
            # Or pick the one that allows the best improvement
            # Greedy: Try to fix each ordinary line, pick best move
            
            best_move = None
            best_new_score = current_score
            
            for key in ordinary_keys:
                pts_on_line = lines[key]
                p_i_idx, p_j_idx = list(pts_on_line)
                
                # Try moving p_i or p_j to a new location
                for move_idx in [p_i_idx, p_j_idx]:
                    # Try to move this point to lie on another line
                    # Candidate lines: All lines with >= 2 points
                    # We want to place the point on a line to increase its count
                    # This effectively "merges" the point into a denser structure
                    
                    # To limit computation, try snapping to lines defined by pairs of OTHER points
                    # But we can just try to find a new position that lies on an existing line
                    # with at least 2 points (other than the one being moved).
                    
                    # Optimization: Pre-compute lines? No, lines dict is available.
                    # Iterate over lines that do NOT contain move_idx
                    for l_key, l_pts in lines.items():
                        if move_idx in l_pts:
                            continue
                        if len(l_pts) < 2:
                            continue
                        
                        # We want to move points[move_idx] to a point on line l_key
                        # such that it is distinct from existing points.
                        # Ideally, we pick a point on l_key that is "close" or "nice".
                        # Let's just pick a point on the line defined by two points in l_pts.
                        # Actually, any integer point on that line works.
                        # Let's pick the one closest to the current position to minimize disruption?
                        # Or just pick a random integer point on the line?
                        
                        # Let's get two points on this line to define it
                        l_pts_list = list(l_pts)
                        a_idx, b_idx = l_pts_list[0], l_pts_list[1]
                        a = current_points[a_idx]
                        b = current_points[b_idx]
                        
                        # Line equation: dy*x - dx*y = c
                        dx = b[0] - a[0]
                        dy = b[1] - a[1]
                        c = dy * a[0] - dx * a[1]
                        
                        # Find an integer point on this line.
                        # We can just take a and b, and maybe another point if len > 2.
                        # To avoid moving to an existing point, we can try a = a + k*(dx/g, dy/g)
                        # Let's try a few small offsets.
                        g = math.gcd(abs(dx), abs(dy))
                        if g == 0: continue
                        step_x = dx // g
                        step_y = dy // g
                        
                        # Try points a + k*step for k in -5..5
                        for k in range(-5, 6):
                            new_x = a[0] + k * step_x
                            new_y = a[1] + k * step_y
                            
                            # Check bounds
                            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                                continue
                            
                            # Check distinctness
                            new_pt = [new_x, new_y]
                            if new_pt in current_points:
                                continue
                            
                            # Apply move
                            old_pt = current_points[move_idx]
                            current_points[move_idx] = new_pt
                            
                            if not is_valid(current_points):
                                current_points[move_idx] = old_pt
                                continue
                                
                            new_score, _, _ = count_ordinary_lines(current_points)
                            
                            if new_score < best_new_score:
                                best_new_score = new_score
                                best_move = (move_idx, new_pt, old_pt)
                            
                            current_points[move_idx] = old_pt
                            
                        # If we found a good move, break early?
                        # No, we want the best move for this line.
                        
                    # End for l_key
                # End for move_idx
            # End for key in ordinary_keys
            
            if best_move:
                move_idx, new_pt, old_pt = best_move
                current_points[move_idx] = new_pt
                current_score = best_new_score
                improved = True
            else:
                # No greedy improvement found. Try a random perturbation to escape local opt
                idx = random.randint(0, 11)
                old_x, old_y = current_points[idx]
                new_x = old_x + random.randint(-20, 20)
                new_y = old_y + random.randint(-20, 20)
                
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    if [new_x, new_y] not in current_points:
                        current_points[idx] = [new_x, new_y]
                        if is_valid(current_points):
                            new_score, _, _ = count_ordinary_lines(current_points)
                            # Accept if better or random (annealing-like but simple)
                            if new_score <= current_score:
                                current_score = new_score
                                improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            break
            
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END
