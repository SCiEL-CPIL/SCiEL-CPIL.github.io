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
    
    # For each pair, compute the line in normalized form
    # Line: ax + by = c where gcd(a,b)=1, and (a,b,c) is normalized
    # Two points define a line; we group points by their line
    
    line_points = defaultdict(list)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            
            # Normal vector (perpendicular)
            a = dy
            b = -dx
            c = -(a * x1 + b * y1)
            
            # Normalize (a, b, c) so gcd(a,b,c)=1 and a>0 or (a==0 and b>0)
            g = gcd(a, b)
            if g > 0:
                a //= g
                b //= g
                c //= g
            
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            line_key = (a, b, c)
            line_points[line_key].extend([i, j])
    
    # Now count how many points are on each line
    # Each line_key has a list of point indices (with duplicates)
    ordinary_count = 0
    for line_key, indices in line_points.items():
        unique_indices = set(indices)
        if len(unique_indices) == 2:
            ordinary_count += 1
    
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Try several structured initial configurations
    initial_configs = []
    
    # 1. 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    initial_configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. Pappus-like configuration (6 lines, 9 intersection points, add 3 more)
    # Pappus: two lines with 3 points each, 6 connecting lines, 3 intersection points
    # Let's use a simpler structured config
    pappus_pts = [
        [0, 0], [1, 0], [2, 0],  # Line 1: y=0
        [0, 1], [1, 1], [2, 1],  # Line 2: y=1
        [0, 2], [1, 2], [2, 2],  # Line 3: y=2
        [3, 0], [3, 1], [3, 2],  # Line 4: x=3
        [4, 0]                    # Extra point
    ]
    initial_configs.append(pappus_pts)
    
    # 4. Desargues-like: two triangles in perspective
    desargues_pts = [
        [0, 0], [1, 0], [0, 1],  # Triangle 1
        [2, 2], [3, 2], [2, 3],  # Triangle 2
        [1, 1], [2, 1], [1, 2],  # Center of perspectivity area
        [0, 2], [2, 0], [3, 3]   # Extra points
    ]
    initial_configs.append(desargues_pts)
    
    # 5. 2x6 grid
    initial_configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6. 6x2 grid  
    initial_configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 7. Hexagonal-ish
    hex_pts = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [3, 0], [3, 1], [3, 2]
    ]
    initial_configs.append(hex_pts)
    
    # 8. Random structured
    random.seed(42)
    for _ in range(5):
        pts = []
        used = set()
        for _ in range(12):
            for _ in range(100):
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append([x, y])
                    break
        initial_configs.append(pts)
    
    # Find best initial config
    best_pts = None
    best_score = float('inf')
    
    for config in initial_configs:
        score = count_ordinary_lines_fast(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    # Now do local search with line-aware moves
    current_pts = [list(p) for p in best_pts]
    current_score = best_score
    
    # Precompute: for each point, which lines does it lie on, and how many points are on those lines?
    # This helps us know where to move a point to improve things
    
    # Simulated annealing with structured moves
    temp = 2.0
    cooling = 0.995
    
    # Time-based loop
    while time.time() < deadline:
        # Pick a random point
        idx = random.randint(0, 11)
        old_x, old_y = current_pts[idx]
        
        # Generate candidate moves
        candidates = []
        
        # Move 1: Small local perturbation
        for _ in range(3):
            nx = old_x + random.randint(-2, 2)
            ny = old_y + random.randint(-2, 2)
            candidates.append((nx, ny))
        
        # Move 2: Try to place on an existing line with 3+ points
        # Find lines with 3+ points and try to place on them
        line_points = defaultdict(list)
        for i in range(12):
            for j in range(i + 1, 12):
                if i == idx or j == idx:
                    continue
                x1, y1 = current_pts[i]
                x2, y2 = current_pts[j]
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
                line_points[(a, b, c)].append(i)
        
        # Try lines with 3+ other points (so adding idx makes 4+)
        for line_key, other_indices in line_points.items():
            if len(other_indices) >= 3:
                # Try to find an integer point on this line that's not already used
                a, b, c = line_key
                # Find a point on line ax + by = c
                # Try small x values
                for x in range(-15, 16):
                    # Solve for y: by = c - ax
                    if b == 0:
                        # a*x = c, so x = c/a
                        if c % a == 0:
                            x_candidate = c // a
                            y_candidate = 0  # y is free, pick 0
                        else:
                            continue
                    else:
                        val = c - a * x
                        if val % b == 0:
                            y_candidate = val // b
                        else:
                            continue
                    
                    new_pt = (x_candidate if b == 0 else x, y_candidate)
                    if new_pt == (old_x, old_y):
                        continue
                    # Check distinctness
                    if any(new_pt == tuple(p) for p in current_pts):
                        continue
                    candidates.append(new_pt)
                    if len(candidates) > 20:
                        break
                if len(candidates) > 20:
                    break
        
        # Move 3: Global random
        for _ in range(2):
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            candidates.append((nx, ny))
        
        # Try candidates
        moved = False
        for nx, ny in candidates:
            new_point = [nx, ny]
            if any(new_point == p for p in current_pts):
                continue
            
            current_pts[idx] = new_point
            new_score = count_ordinary_lines_fast(current_pts)
            
            if new_score < current_score:
                current_score = new_score
                moved = True
                if new_score < best_score:
                    best_score = new_score
                    best_pts = [list(p) for p in current_pts]
                break
            else:
                delta = new_score - current_score
                if random.random() < pow(2.718, -delta / temp):
                    current_score = new_score
                    moved = True
                    break
                else:
                    current_pts[idx] = [old_x, old_y]
        
        if not moved:
            current_pts[idx] = [old_x, old_y]
        
        temp *= cooling
        if temp < 0.05:
            temp = 0.05
    
    return best_pts
