import random
import time
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
    
    for key, pts in line_map.items():
        if len(pts) >= 5:
            is_valid = False
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
        search_x_range = range(max(x_range[0], -100), min(x_range[1], 100) + 1)
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
    Aggressive local search focused on extending 3-point lines to 4-point lines.
    """
    # Start with a 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    score, is_valid, line_map = _evaluate(points)
    best_score = score
    best_points = [p[:] for p in points]
    
    step = 0
    no_improve_count = 0
    
    while time.time() < deadline and step < 20000:
        step += 1
            
        if not is_valid:
            points = [p[:] for p in best_points]
            score, is_valid, line_map = _evaluate(points)
            continue
            
        # 1. Try to extend 3-point lines
        three_lines = [key for key, pts in line_map.items() if len(pts) == 3]
        
        found_improvement = False
        if three_lines:
            # Shuffle to randomize order
            random.shuffle(three_lines)
            for line_key in three_lines:
                if time.time() >= deadline:
                    break
                pts_on_line = line_map[line_key]
                a, b, c = line_key
                
                # Find points NOT on this line
                not_on_line = [i for i in range(16) if i not in pts_on_line]
                
                if not not_on_line:
                    continue
                
                # Find candidate positions on the line
                exclude_set = set((points[i][0], points[i][1]) for i in pts_on_line)
                # Search range expanded slightly to allow more flexibility
                candidates = _find_integer_points_on_line(a, b, c, exclude_set, (-100, 100), (-100, 100))
                
                if not candidates:
                    continue
                
                random.shuffle(candidates)
                for new_pos in candidates:
                    # Check if this position is already occupied
                    occupied = False
                    for i, p in enumerate(points):
                        if p[0] == new_pos[0] and p[1] == new_pos[1]:
                            occupied = True
                            break
                    if occupied:
                        continue
                    
                    # Try moving each point not on the line to this position
                    # We prioritize moving points that are part of fewest 4-lines to avoid breaking them
                    # But for simplicity, we just try random ones first
                    random.shuffle(not_on_line)
                    for idx in not_on_line:
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
                            found_improvement = True
                            break
                        elif new_valid and new_score == score:
                            # Accept equal score moves with some probability to explore
                            if random.random() < 0.3:
                                points = new_points
                                line_map = new_line_map
                                found_improvement = True
                                break
                    if found_improvement:
                        break
                if found_improvement:
                    break
        
        # 2. If no improvement from 3-lines, try random swaps or perturbations
        if not found_improvement:
            no_improve_count += 1
            # Occasionally do a random swap
            if no_improve_count % 10 == 0:
                i, j = random.sample(range(16), 2)
                new_points = [p[:] for p in points]
                new_points[i], new_points[j] = new_points[j], new_points[i]
                
                new_score, new_valid, new_line_map = _evaluate(new_points)
                if new_valid:
                    if new_score > score:
                        points = new_points
                        score = new_score
                        line_map = new_line_map
                        if score > best_score:
                            best_score = score
                            best_points = [p[:] for p in points]
                        no_improve_count = 0
                    elif new_score == score:
                        if random.random() < 0.5:
                            points = new_points
                            line_map = new_line_map
                            no_improve_count = 0
            else:
                # Small perturbation
                idx = random.randint(0, 15)
                cx, cy = points[idx]
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
                        
                        new_score, new_valid, new_line_map = _evaluate(new_points)
                        if new_valid:
                            if new_score > score:
                                points = new_points
                                score = new_score
                                line_map = new_line_map
                                if score > best_score:
                                    best_score = score
                                    best_points = [p[:] for p in points]
                                no_improve_count = 0
                            elif new_score == score:
                                if random.random() < 0.3:
                                    points = new_points
                                    line_map = new_line_map
                                    no_improve_count = 0
        else:
            no_improve_count = 0
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = 0
    
    # Run multiple starts with different seeds/configs
    for i in range(5):
        if time.time() >= deadline - 10:
            break
        
        # Different initial configs
        if i == 0:
            pts = [[x, y] for x in range(4) for y in range(4)]
        elif i == 1:
            pts = [[x * 2, y * 2] for x in range(4) for y in range(4)]
        elif i == 2:
            pts = [[x, y] for x in range(4) for y in range(4)]
            # Swap a few points to break symmetry
            random.shuffle(pts)
        elif i == 3:
            pts = [[x, y] for x in range(4) for y in range(4)]
            # Rotate
            pts = [[-y, x] for x, y in pts]
        else:
            pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
            # Ensure distinct
            seen = set()
            valid_pts = []
            for p in pts:
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    valid_pts.append(p)
            while len(valid_pts) < 16:
                p = [random.randint(-10, 10), random.randint(-10, 10)]
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    valid_pts.append(p)
            pts = valid_pts
        
        score, is_valid, _ = _evaluate(pts)
        if not is_valid:
            continue
            
        # Run greedy construct
        sub_deadline = min(deadline - 5, time.time() + (deadline - start_time) / 5)
        pts, sc = _greedy_construct(sub_deadline)
        
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    # Final polish
    if best_points and time.time() < deadline - 5:
        sub_deadline = deadline - 2
        pts, sc = _greedy_construct(sub_deadline)
        if sc > best_score:
            best_points = pts
            best_score = sc
            
    if best_points:
        return best_points
    else:
        # Fallback
        return [[x, y] for x in range(4) for y in range(4)]

# EVOLVE-BLOCK-END
