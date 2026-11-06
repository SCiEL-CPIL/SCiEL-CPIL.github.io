"""Line-packing formulation: select 4-point lines to maximize count with 16 points, no 5 collinear."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _normalize_line(a, b, c):
    """Normalize line equation ax + by = c to canonical form."""
    if a == 0 and b == 0:
        return None
    g = math.gcd(abs(a), abs(b))
    if g > 1:
        a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _line_from_points(p1, p2):
    """Get normalized line through two points."""
    a, b = p1
    c, d = p2
    A = d - b
    B = a - c
    C = A * a + B * b
    return _normalize_line(A, B, C)

def _points_on_line(line, points):
    """Return indices of points on the given line."""
    a, b, c = line
    result = []
    for i, (x, y) in enumerate(points):
        if a * x + b * y == c:
            result.append(i)
    return result

def _compute_score(points):
    """
    Compute score: number of lines with exactly 4 points.
    Returns (score, has_5, line_map) where line_map maps line -> set of point indices.
    """
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line is None:
                continue
            if line not in line_map:
                line_map[line] = set()
            line_map[line].add(i)
            line_map[line].add(j)
    
    score = 0
    has_5 = False
    for line, pts in line_map.items():
        cnt = len(pts)
        if cnt == 4:
            score += 1
        elif cnt >= 5:
            has_5 = True
    return score, has_5, line_map

def _find_4point_lines(points):
    """Find all lines containing exactly 4 of the given points."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line is None:
                continue
            if line not in line_map:
                line_map[line] = set()
            line_map[line].add(i)
            line_map[line].add(j)
    
    result = []
    for line, pts in line_map.items():
        if len(pts) == 4:
            result.append((line, pts))
    return result

def _generate_candidate_lines(rng, n_lines=200):
    """
    Generate candidate 4-point lines with integer points in [-1000, 1000].
    Each line is defined by choosing 4 collinear integer points.
    """
    candidates = []
    
    # Strategy: pick a base point, direction, and generate 4 points
    for _ in range(n_lines):
        # Choose base point
        x0 = rng.randint(-500, 500)
        y0 = rng.randint(-500, 500)
        
        # Choose direction (dx, dy) with small values to keep points in range
        dx = rng.randint(-20, 20)
        dy = rng.randint(-20, 20)
        if dx == 0 and dy == 0:
            continue
        
        # Ensure direction is primitive
        g = math.gcd(abs(dx), abs(dy))
        dx, dy = dx // g, dy // g
        
        # Generate 4 points: base + k*direction for k=0,1,2,3
        points = []
        valid = True
        for k in range(4):
            x = x0 + k * dx
            y = y0 + k * dy
            if abs(x) > 1000 or abs(y) > 1000:
                valid = False
                break
            points.append((x, y))
        
        if not valid or len(set(points)) < 4:
            continue
        
        # Get the line through these points
        line = _line_from_points(points[0], points[1])
        if line is None:
            continue
        
        # Store line with its 4 points
        candidates.append({
            'line': line,
            'points': points,
            'point_set': set(points)
        })
    
    return candidates

def _select_compatible_lines(candidate_lines, max_lines=15):
    """
    Select a compatible set of lines such that:
    - Total distinct points = 16
    - No 5 points are collinear
    - Maximize number of 4-point lines
    """
    best_selection = []
    best_score = 0
    
    # Sort by some heuristic (e.g., prefer lines with fewer overlaps)
    rng = random.Random(42)
    shuffled = candidate_lines[:]
    rng.shuffle(shuffled)
    
    current_selection = []
    current_points = set()
    
    for cand in shuffled:
        # Check if adding this line keeps total points <= 16
        new_points = current_points | cand['point_set']
        if len(new_points) > 16:
            continue
        
        # Check for 5 collinear: for each existing line, check if intersection creates 5+
        valid = True
        for existing in current_selection:
            # Find intersection points
            inter = existing['point_set'] & cand['point_set']
            if len(inter) >= 2:
                # Two lines share 2+ points, they're the same line
                # Check if combined they have 5+ points
                combined = existing['point_set'] | cand['point_set']
                if len(combined) >= 5:
                    valid = False
                    break
            elif len(inter) == 1:
                # They share exactly 1 point, that's fine
                pass

        if valid:
            current_selection.append(cand)
            current_points = new_points
            
            # Check if we have 16 points
            if len(current_points) == 16:
                # Verify no 5 collinear
                pts_list = list(current_points)
                score, has_5, _ = _compute_score(pts_list)
                if not has_5 and score > best_score:
                    best_score = score
                    best_selection = current_selection[:]
                break
        
        # Early stop if we have enough lines
        if len(current_selection) >= max_lines:
            break
    
    return best_selection, best_score

def _construct_from_lines(line_selection):
    """
    Construct 16 points from selected lines.
    If we have fewer than 16 points, add more to reach 16 while maintaining constraints.
    """
    if not line_selection:
        return None
    
    # Collect all points from selected lines
    all_points = set()
    for cand in line_selection:
        all_points.update(cand['point_set'])
    
    pts_list = list(all_points)
    
    # If we have exactly 16, verify and return
    if len(pts_list) == 16:
        score, has_5, _ = _compute_score(pts_list)
        if not has_5:
            return pts_list
        return None
    
    # If we have fewer than 16, try to add points
    if len(pts_list) < 16:
        rng = random.Random(123)
        existing_set = set(pts_list)
        
        while len(pts_list) < 16:
            # Try to add a point that extends a 3-point line to 4
            # First, find all 3-point lines
            line_map = {}
            for i in range(len(pts_list)):
                for j in range(i + 1, len(pts_list)):
                    line = _line_from_points(pts_list[i], pts_list[j])
                    if line is None:
                        continue
                    if line not in line_map:
                        line_map[line] = set()
                    line_map[line].add(i)
                    line_map[line].add(j)
            
            three_lines = [line for line, pts in line_map.items() if len(pts) == 3]
            
            if three_lines and rng.random() < 0.7:
                # Try to complete a 3-point line
                target_line = rng.choice(three_lines)
                a, b, c = target_line
                
                # Find the 3 points on this line
                pts_on_line = [p for i, p in enumerate(pts_list) if a * p[0] + b * p[1] == c]
                if len(pts_on_line) != 3:
                    continue
                
                # Find the missing point
                # The 3 points should be evenly spaced if they're part of a 4-point line
                # Try to find a point that makes a 4-point line
                # Get the line direction
                if len(pts_on_line) >= 2:
                    p1, p2 = pts_on_line[0], pts_on_line[1]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g > 0:
                        dx, dy = dx // g, dy // g
                        # Try points in both directions
                        for k in [-1, 1, -2, 2]:
                            new_x = pts_on_line[0][0] + k * dx
                            new_y = pts_on_line[0][1] + k * dy
                            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                                new_pt = (new_x, new_y)
                                if new_pt not in existing_set:
                                    # Check if this creates 5 collinear
                                    test_pts = pts_list + [new_pt]
                                    _, h5, _ = _compute_score(test_pts)
                                    if not h5:
                                        pts_list.append(new_pt)
                                        existing_set.add(new_pt)
                                        break
                        else:
                            # Try random points on the line
                            # Parametrize: x = x0 + t*dx, y = y0 + t*dy
                            x0, y0 = pts_on_line[0]
                            for t in range(-10, 11):
                                if t == 0:
                                    continue
                                new_x = x0 + t * dx
                                new_y = y0 + t * dy
                                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                                    new_pt = (new_x, new_y)
                                    if new_pt not in existing_set:
                                        test_pts = pts_list + [new_pt]
                                        _, h5, _ = _compute_score(test_pts)
                                        if not h5:
                                            pts_list.append(new_pt)
                                            existing_set.add(new_pt)
                                            break
            else:
                # Random point
                while True:
                    x = rng.randint(-500, 500)
                    y = rng.randint(-500, 500)
                    new_pt = (x, y)
                    if new_pt not in existing_set:
                        test_pts = pts_list + [new_pt]
                        _, h5, _ = _compute_score(test_pts)
                        if not h5:
                            pts_list.append(new_pt)
                            existing_set.add(new_pt)
                            break
                        elif len(pts_list) >= 14:
                            break
        
        if len(pts_list) == 16:
            score, has_5, _ = _compute_score(pts_list)
            if not has_5:
                return pts_list
    
    return None

def _local_search_points(points, deadline, rng):
    """
    Local search to improve the score by moving points.
    """
    current = [p[:] for p in points]
    current_score, has_5, _ = _compute_score(current)
    if has_5:
        return current, 0
    
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 2.0
    T_min = 0.1
    cooling = 0.999
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Pick a point to move
        idx = rng.randint(0, 15)
        old_pt = current[idx][:]
        
        # Try moving to a nearby position
        step = rng.randint(1, 30)
        new_x = old_pt[0] + rng.randint(-step, step)
        new_y = old_pt[1] + rng.randint(-step, step)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        new_pt = (new_x, new_y)
        if new_pt in set(tuple(p) for p in current if p != old_pt):
            continue
        
        current[idx] = [new_x, new_y]
        new_score, new_h5, _ = _compute_score(current)
        
        if not new_h5:
            delta = new_score - current_score
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = [p[:] for p in current]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current[idx] = old_pt
                no_improve += 1
        else:
            current[idx] = old_pt
            no_improve += 1
        
        T = max(T_min, T * cooling)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Generate candidate lines and select compatible ones
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        
        # Generate candidate lines
        candidates = _generate_candidate_lines(rng, n_lines=300)
        
        if not candidates:
            continue
        
        # Select compatible lines
        selection, sel_score = _select_compatible_lines(candidates)
        
        if selection:
            pts = _construct_from_lines(selection)
            if pts is not None:
                score, has_5, _ = _compute_score(pts)
                if not has_5 and score > best_score:
                    best_score = score
                    best_points = [list(p) for p in pts]
    
    # Strategy 2: Start from a grid and improve
    if time.time() < deadline - 20:
        # 4x4 grid
        base = [[x, y] for x in range(4) for y in range(4)]
        s, h5, _ = _compute_score(base)
        if not h5 and s > best_score:
            best_score = s
            best_points = [p[:] for p in base]
        
        # Perturbed grids
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            rng2 = random.Random(seed * 1000 + 7)
            current = [p[:] for p in base]
            for _ in range(rng2.randint(2, 6)):
                idx = rng2.randint(0, 15)
                new_x = current[idx][0] + rng2.randint(-10, 10)
                new_y = current[idx][1] + rng2.randint(-10, 10)
                new_pt = [new_x, new_y]
                if any(new_pt == p for p in current):
                    continue
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                current[idx] = new_pt
            s3, h5_3, _ = _compute_score(current)
            if not h5_3 and s3 > best_score:
                best_score = s3
                best_points = [p[:] for p in current]
    
    # Local search from best
    if best_points is not None and time.time() < deadline - 10:
        opt_pts, opt_score = _local_search_points(best_points, deadline, rng)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Final verification
    if best_points is not None:
        _, final_h5, _ = _compute_score(best_points)
        if final_h5:
            # Fallback
            best_points = [[x, y] for x in range(4) for y in range(4)]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
