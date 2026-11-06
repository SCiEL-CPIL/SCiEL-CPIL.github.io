import time
import random
from collections import defaultdict

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def count_ordinary_lines_fast(points):
    """Count ordinary lines using line grouping by normalized slope/intercept."""
    n = len(points)
    if n < 2:
        return 0
    
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            a = dy
            b = -dx
            c = -(a * x1 + b * y1)
            
            g = gcd(a, b)
            if g > 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            line_key = (a, b, c)
            line_points[line_key].add(i)
            line_points[line_key].add(j)
    
    ordinary_count = 0
    for line_key, indices in line_points.items():
        if len(indices) == 2:
            ordinary_count += 1
    
    return ordinary_count

def _line_key_from_points(p1, p2):
    """Compute normalized line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    g = gcd(a, b)
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _find_integer_points_on_line(line_key, used_set, max_coord=200):
    """Find integer points on a line not in used_set."""
    a, b, c = line_key
    results = []
    if a == 0 and b == 0:
        return results
    if a == 0:
        # by = c -> y = c/b
        if b != 0 and c % b == 0:
            y_val = c // b
            for x in range(-max_coord, max_coord + 1):
                pt = (x, y_val)
                if pt not in used_set:
                    results.append(pt)
        return results
    if b == 0:
        # ax = c -> x = c/a
        if a != 0 and c % a == 0:
            x_val = c // a
            for y in range(-max_coord, max_coord + 1):
                pt = (x_val, y)
                if pt not in used_set:
                    results.append(pt)
        return results
    # ax + by = c
    # Find one solution, then parameterize
    # Extended Euclidean for gcd(a,b)
    def ext_gcd(aa, bb):
        if bb == 0:
            return (aa, 1, 0)
        g, x, y = ext_gcd(bb, aa % bb)
        return (g, y, x - (aa // bb) * y)
    
    g_ab, x0, y0 = ext_gcd(a, b)
    if c % g_ab != 0:
        return results
    # General solution: x = x0*c/g_ab + (b/g_ab)*t, y = y0*c/g_ab - (a/g_ab)*t
    x_base = x0 * (c // g_ab)
    y_base = y0 * (c // g_ab)
    dx_step = b // g_ab
    dy_step = -a // g_ab
    
    for t in range(-max_coord, max_coord + 1):
        x_val = x_base + dx_step * t
        y_val = y_base + dy_step * t
        if abs(x_val) <= max_coord and abs(y_val) <= max_coord:
            pt = (x_val, y_val)
            if pt not in used_set:
                results.append(pt)
    return results

def _try_pappus_extension():
    """
    Pappus configuration: 9 points on 9 lines, each line has exactly 3 points.
    Extend to 12 points by adding 3 points that lie on existing lines or create new multi-point lines.
    
    Classic Pappus: Two lines L1, L2 with 3 points each.
    L1: A1, A2, A3
    L2: B1, B2, B3
    Cross lines: AiBj for i≠j, their intersections give 3 more points C1, C2, C3 on line L3.
    
    We'll use a concrete integer realization.
    """
    # One known integer Pappus configuration
    # Line 1 (y=0): (0,0), (1,0), (2,0)
    # Line 2 (y=1): (0,1), (1,1), (2,1)
    # This gives cross intersections... let me use a verified one.
    
    # Alternative: use the Hesse configuration or other known finite geometry
    # For now, try a 4x3 grid with modifications
    
    configs = []
    
    # Config A: 3x4 grid (12 points)
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config B: Pappus-like with 9 core + 3 extensions
    # Use points from a known Pappus realization
    pappus = [
        [0, 0], [1, 0], [2, 0],   # Line y=0
        [0, 1], [1, 1], [2, 1],   # Line y=1  
        [0, 2], [1, 2], [2, 2],   # Line y=2
        [3, 0], [3, 1], [3, 2],   # Line x=3
    ]
    configs.append(pappus)
    
    # Config C: Two parallel lines with 4 points each + 4 more
    configs.append([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
    ])
    
    # Config D: Desargues configuration (10 points) + 2 more
    desargues = [
        [0, 0], [1, 0], [0, 1],
        [2, 2], [3, 2], [2, 3],
        [1, 1], [0, 2], [2, 0], [3, 3],
        [1, 2], [2, 1]
    ]
    configs.append(desargues)
    
    # Config E: 6 points on one line, 6 on another (parallel)
    configs.append([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
    ])
    
    # Config F: 4x3 grid shifted
    configs.append([[x + 1, y + 1] for x in range(4) for y in range(3)])
    
    # Config G: Hexagonal lattice points
    configs.append([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
    ])
    
    # Config H: 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # Config I: 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # Config J: Random but structured - points on few lines
    random.seed(123)
    for seed in range(20):
        random.seed(seed)
        pts = []
        used = set()
        # Place points on a few lines
        num_lines = random.randint(3, 5)
        for _ in range(num_lines):
            # Pick a line through origin with random slope
            slope_num = random.randint(-5, 5)
            slope_den = random.randint(1, 5)
            # Add points on this line
            for _ in range(random.randint(2, 4)):
                t = random.randint(-10, 10)
                x = t * slope_den
                y = t * slope_num
                if (x, y) not in used and abs(x) <= 100 and abs(y) <= 100:
                    used.add((x, y))
                    pts.append([x, y])
        # Fill remaining with random
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        if len(pts) == 12:
            configs.append(pts)
    
    return configs

def _local_search_line_aware(points, deadline):
    """
    Local search that tries to move points onto existing multi-point lines.
    Uses incremental scoring for speed.
    """
    current_pts = [list(p) for p in points]
    current_score = count_ordinary_lines_fast(current_pts)
    best_score = current_score
    best_pts = [list(p) for p in current_pts]
    
    used_set = set(tuple(p) for p in current_pts)
    
    # Precompute all lines and their point counts
    def compute_lines(pts):
        lines = defaultdict(set)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                key = _line_key_from_points(pts[i], pts[j])
                lines[key].add(i)
                lines[key].add(j)
        return lines
    
    lines = compute_lines(current_pts)
    
    temp = 1.0
    cooling = 0.999
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 500:
            break
        
        idx = random.randint(0, 11)
        old_pt = tuple(current_pts[idx])
        old_x, old_y = old_pt
        
        # Generate smart candidates
        candidates = []
        
        # 1. Points on lines with 3+ other points
        for line_key, point_indices in lines.items():
            other_count = len(point_indices - {idx})
            if other_count >= 2:  # Adding idx makes 3+
                a, b, c = line_key
                # Find integer points on this line
                pts_on_line = _find_integer_points_on_line(line_key, used_set, max_coord=50)
                for pt in pts_on_line[:5]:
                    candidates.append(pt)
        
        # 2. Small perturbations
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = old_x + dx, old_y + dy
                if (nx, ny) not in used_set:
                    candidates.append((nx, ny))
        
        # 3. Random far moves
        for _ in range(3):
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            if (nx, ny) not in used_set:
                candidates.append((nx, ny))
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        moved = False
        for nx, ny in candidates[:30]:
            new_pt = (nx, ny)
            if new_pt in used_set:
                continue
            
            # Temporarily place
            current_pts[idx] = list(new_pt)
            used_set.discard(old_pt)
            used_set.add(new_pt)
            
            new_score = count_ordinary_lines_fast(current_pts)
            
            if new_score < current_score:
                current_score = new_score
                moved = True
                if new_score < best_score:
                    best_score = new_score
                    best_pts = [list(p) for p in current_pts]
                lines = compute_lines(current_pts)
                no_improve_count = 0
                break
            else:
                delta = new_score - current_score
                accept_prob = min(1.0, pow(2.718, -delta / max(temp, 0.01)))
                if random.random() < accept_prob:
                    current_score = new_score
                    moved = True
                    lines = compute_lines(current_pts)
                    no_improve_count = 0
                    break
                else:
                    current_pts[idx] = list(old_pt)
                    used_set.discard(new_pt)
                    used_set.add(old_pt)
        
        if not moved:
            current_pts[idx] = list(old_pt)
            used_set.discard(new_pt)
            used_set.add(old_pt)
            no_improve_count += 1
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate candidate configurations from known structures
    configs = _try_pappus_extension()
    
    best_pts = None
    best_score = float('inf')
    
    # Evaluate all initial configs
    for config in configs:
        if len(config) != 12:
            continue
        # Check distinctness
        pts_tuples = [tuple(p) for p in config]
        if len(set(pts_tuples)) != 12:
            continue
        score = count_ordinary_lines_fast(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    if best_pts is None:
        # Fallback: simple grid
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines_fast(best_pts)
    
    # Run local search from best config
    refined_pts, refined_score = _local_search_line_aware(best_pts, deadline)
    if refined_score < best_score:
        best_score = refined_score
        best_pts = refined_pts
    
    # Try local search from a few other good configs too
    for config in configs[:5]:
        if time.time() > deadline:
            break
        if len(config) != 12:
            continue
        pts_tuples = [tuple(p) for p in config]
        if len(set(pts_tuples)) != 12:
            continue
        score = count_ordinary_lines_fast(config)
        if score <= best_score + 1:  # Only try if close to best
            refined_pts2, refined_score2 = _local_search_line_aware(config, deadline)
            if refined_score2 < best_score:
                best_score = refined_score2
                best_pts = refined_pts2
    
    return best_pts
