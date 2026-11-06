import random
import math
import time
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize direction vector
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    a = dy // g
    b = -dx // g
    
    # Normalize sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        
    # Calculate c such that ax + by + c = 0
    c = -(a * x1 + b * y1)
    
    return (a, b, c)

def _build_line_map(points):
    """Build a map from line key to set of point indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _evaluate(points):
    """
    Returns (score, is_valid, line_map).
    score: number of lines with exactly 4 points.
    is_valid: False if any line has >= 5 points.
    line_map: dict from line key to set of indices.
    """
    line_map = _build_line_map(points)
    score = 0
    is_valid = True
    
    # Check validity first: if any line has >= 5, it's invalid
    # We can do this in a single pass over the line map
    for key, pts in line_map.items():
        if len(pts) >= 5:
            is_valid = False
            # We still need to return line_map for debugging or further steps, 
            # but score is 0
            return 0, False, line_map
            
    if is_valid:
        for key, pts in line_map.items():
            if len(pts) == 4:
                score += 1
            
    return score, is_valid, line_map

def _find_integer_points_on_line(a, b, c, exclude_set, x_range, y_range):
    """Find integer points (x,y) on line ax+by+c=0 within bounds, excluding points in exclude_set."""
    candidates = []
    if a == 0 and b == 0:
        return candidates
    
    if a == 0:
        # by + c = 0 => y = -c/b (must be integer)
        if b != 0 and (-c) % b == 0:
            y = -c // b
            if y_range[0] <= y <= y_range[1]:
                for x in range(x_range[0], x_range[1] + 1):
                    if (x, y) not in exclude_set:
                        candidates.append([x, y])
    elif b == 0:
        # ax + c = 0 => x = -c/a (must be integer)
        if (-c) % a == 0:
            x = -c // a
            if x_range[0] <= x <= x_range[1]:
                for y in range(y_range[0], y_range[1] + 1):
                    if (x, y) not in exclude_set:
                        candidates.append([x, y])
    else:
        # ax + by + c = 0
        # We need to find integer solutions. 
        # Let's iterate over a small range of x values and check if y is integer.
        # To limit the search, we'll use a smaller range.
        search_x_range = range(max(x_range[0], -50), min(x_range[1], 50) + 1)
        for x in search_x_range:
            num = -(a * x + c)
            if num % b == 0:
                y = num // b
                if y_range[0] <= y <= y_range[1]:
                    if (x, y) not in exclude_set:
                        candidates.append([x, y])
    
    return candidates

def _greedy_construct(deadline):
    """
    Greedy construction that tries to build a configuration with many 4-point lines.
    Strategy: Start with a 4x4 grid, then iteratively try to swap points to create more 4-lines.
    """
    # Start with a 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    score, is_valid, line_map = _evaluate(points)
    best_score = score
    best_points = [p[:] for p in points]
    
    step = 0
    while time.time() < deadline and step < 5000:
        step += 1
            
        if not is_valid:
            # Reset to best valid
            points = [p[:] for p in best_points]
            score, is_valid, line_map = _evaluate(points)
            continue
            
        # Find lines with exactly 3 points - we want to add a 4th
        three_lines = [key for key, pts in line_map.items() if len(pts) == 3]
            
        if not three_lines:
            # No 3-lines to extend, try random moves
            idx = random.randint(0, 15)
            cx, cy = points[idx]
            for _ in range(5): # Reduced from 20 to 5 for speed
                nx = cx + random.randint(-5, 5)
                ny = cy + random.randint(-5, 5)
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    occupied = False
                    for i, p in enumerate(points):
                        if i != idx and p[0] == nx and p[1] == ny:
                            occupied = True
                            break
                    if not occupied:
                        new_points = [p[:] for p in points]
                        new_points[idx] = [nx, ny]
                        new_score, new_valid, _ = _evaluate(new_points)
                        if new_valid and new_score > score:
                            points = new_points
                            score = new_score
                            line_map = _build_line_map(points)
                            if score > best_score:
                                best_score = score
                                best_points = [p[:] for p in points]
                            break
            continue
        
        # Pick a random 3-line
        line_key = random.choice(three_lines)
        pts_on_line = line_map[line_key]
        a, b, c = line_key
        
        # Find points NOT on this line
        not_on_line = [i for i in range(16) if i not in pts_on_line]
        
        if not not_on_line:
            continue
        
        # For each point not on the line, check if moving it to the line helps
        # We need to find a new position on the line that doesn't create a 5-line
        exclude_set = set((points[i][0], points[i][1]) for i in pts_on_line)
        candidates = _find_integer_points_on_line(a, b, c, exclude_set, (-50, 50), (-50, 50))
        
        if not candidates:
            continue
        
        # Try a few random candidates
        random.shuffle(candidates)
        for new_pos in candidates[:5]:
            # Check if this position is already occupied
            occupied = False
            for i, p in enumerate(points):
                if p[0] == new_pos[0] and p[1] == new_pos[1]:
                    occupied = True
                    break
            if occupied:
                continue
            
            # Try moving a random point not on the line to this position
            idx = random.choice(not_on_line)
            new_points = [p[:] for p in points]
            new_points[idx] = new_pos
            
            new_score, new_valid, new_line_map = _evaluate(new_points)
            if new_valid and new_score > score:
                points = new_points
                score = new_score
                line_map = new_line_map
                if score > best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
                break
    
    return best_points, best_score

def _swap_local_search(points, deadline):
    """
    Local search using swaps: exchange positions of two points.
    This preserves the set of occupied positions but changes which point is where.
    """
    score, is_valid, line_map = _evaluate(points)
    best_score = score
    best_points = [p[:] for p in points]
    
    step = 0
    while time.time() < deadline and step < 10000:
        step += 1
            
        if not is_valid:
            points = [p[:] for p in best_points]
            score, is_valid, line_map = _evaluate(points)
            continue
            
        # Pick two random points to swap
        i, j = random.sample(range(16), 2)
            
        new_points = [p[:] for p in points]
        new_points[i], new_points[j] = new_points[j], new_points[i]
            
        new_score, new_valid, new_line_map = _evaluate(new_points)
            
        if new_valid and new_score > score:
            points = new_points
            score = new_score
            line_map = new_line_map
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in points]
        elif new_valid and new_score == score:
            if random.random() < 0.2:
                points = new_points
                line_map = new_line_map
    
    return best_points, best_score

def _perturb_local_search(points, deadline):
    """
    Local search using perturbations: move one point to a nearby random location.
    """
    score, is_valid, line_map = _evaluate(points)
    best_score = score
    best_points = [p[:] for p in points]
    
    step = 0
    while time.time() < deadline and step < 10000:
        step += 1
            
        if not is_valid:
            points = [p[:] for p in best_points]
            score, is_valid, line_map = _evaluate(points)
            continue
            
        idx = random.randint(0, 15)
        cx, cy = points[idx]
            
        # Try several nearby positions
        found_improvement = False
        for _ in range(10): # Reduced from 30 to 10
            nx = cx + random.randint(-8, 8)
            ny = cy + random.randint(-8, 8)
            if not (-1000 <= nx <= 1000 and -1000 <= ny <= 1000):
                continue
                
            # Check if occupied
            occupied = False
            for i, p in enumerate(points):
                if i != idx and p[0] == nx and p[1] == ny:
                    occupied = True
                    break
            if occupied:
                continue
                
            new_points = [p[:] for p in points]
            new_points[idx] = [nx, ny]
                
            new_score, new_valid, new_line_map = _evaluate(new_points)
                
            if new_valid and new_score > score:
                points = new_points
                score = new_score
                line_map = new_line_map
                if score > best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
                found_improvement = True
                break
            elif new_valid and new_score == score:
                if random.random() < 0.15:
                    points = new_points
                    line_map = new_line_map
    
    return best_points, best_score

def _multi_start_search(deadline):
    """
    Multi-start local search with diverse initial configurations.
    """
    start_time = time.time()
    remaining = deadline - start_time
    
    # Define diverse initial configurations
    configs = []
    
    # 4x4 grid
    configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid with different spacing
    configs.append([[x * 3, y * 3] for x in range(4) for y in range(4)])
    
    # 4x4 grid centered and scaled
    configs.append([[x * 2 - 3, y * 2 - 3] for x in range(4) for y in range(4)])
    
    # 5x4 grid subset (remove 4 points)
    base_5x4 = [[x, y] for x in range(5) for y in range(4)]
    random.shuffle(base_5x4)
    configs.append(base_5x4[:16])
    
    # 6x3 grid subset
    base_6x3 = [[x, y] for x in range(6) for y in range(3)]
    random.shuffle(base_6x3)
    configs.append(base_6x3[:16])
    
    # Random points in a small area
    configs.append([[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)])
    
    best_score = 0
    best_points = None
    
    # Allocate time to each config
    time_per_config = remaining / len(configs)
    
    for i, config in enumerate(configs):
        if time.time() >= deadline - 5:
            break
        
        # Check for duplicates
        seen = set()
        valid_config = []
        for p in config:
            if (p[0], p[1]) not in seen and -1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000:
                seen.add((p[0], p[1]))
                valid_config.append(p)
        
        if len(valid_config) != 16:
            continue
        
        config = valid_config
        score, is_valid, _ = _evaluate(config)
        
        if score > best_score and is_valid:
            best_score = score
            best_points = [p[:] for p in config]
        
        # Run greedy construction
        if time.time() < deadline - 10:
            sub_deadline = min(deadline - 5, time.time() + time_per_config * 0.4)
            pts, sc = _greedy_construct(sub_deadline)
            if sc > best_score:
                best_score = sc
                best_points = pts
        
        # Run swap local search
        if time.time() < deadline - 10 and best_points:
            sub_deadline = min(deadline - 5, time.time() + time_per_config * 0.3)
            pts, sc = _swap_local_search(best_points, sub_deadline)
            if sc > best_score:
                best_score = sc
                best_points = pts
        
        # Run perturb local search
        if time.time() < deadline - 10 and best_points:
            sub_deadline = min(deadline - 5, time.time() + time_per_config * 0.3)
            pts, sc = _perturb_local_search(best_points, sub_deadline)
            if sc > best_score:
                best_score = sc
                best_points = pts
    
    return best_points

def construct():
    start_time = time.time()
    # 120s limit. Use 110s deadline to be safe.
    deadline = start_time + 110.0  
    
    # Phase 1: Multi-start search
    best_points = _multi_start_search(deadline)
    
    # Phase 2: Final polish with extended search
    if best_points and time.time() < deadline - 10:
        # Run swap search
        sub_deadline = min(deadline - 5, time.time() + 30)
        pts, sc = _swap_local_search(best_points, sub_deadline)
        if sc > 0:
            best_points = pts
        
        # Run perturb search
        if time.time() < deadline - 5:
            sub_deadline = min(deadline - 2, time.time() + 30)
            pts, sc = _perturb_local_search(best_points, sub_deadline)
            if sc > 0:
                best_points = pts
    
    # Final validation
    if best_points:
        score, is_valid, _ = _evaluate(best_points)
        if not is_valid or score == 0:
            # Fallback to simple 4x4 grid
            best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points

# EVOLVE-BLOCK-END
