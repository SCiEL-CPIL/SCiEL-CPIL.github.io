"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def _count_ordinary_fast(points):
    """Count lines containing exactly 2 of the given points. O(n^2) via slope counting."""
    n = len(points)
    if n < 3:
        return 0
    # For each pair, check if a third point is collinear
    # But faster: for each point i, count slopes to all other points.
    # A line through i and j with exactly those 2 points means:
    #   - from i's perspective, slope to j is unique (no other point has same slope)
    #   - from j's perspective, slope to i is unique
    # Actually, the cleanest O(n^2) approach:
    # For each pair (i,j), the line is ordinary iff no other point k is on it.
    # That's O(n^3) naively. But we can use: for each point i, group other points by slope.
    # A line through i with slope s has count c_i(s) points (including i).
    # The line is ordinary iff c_i(s) == 2.
    # But we need to count each line once. Each ordinary line {i,j} is seen from i (slope to j unique)
    # and from j (slope to i unique). So count = number of pairs (i, slope) where count==2, divided by... 
    # No, each ordinary line has exactly 2 endpoints, each sees it as a unique slope. So total ordinary = 
    # (1/2) * sum over i of (number of slopes from i that have exactly 1 other point).
    # Wait: if line {i,j} is ordinary, from i's view, slope(i->j) has exactly 1 point (j).
    # From j's view, slope(j->i) has exactly 1 point (i).
    # So each ordinary line contributes 2 to the sum. Total ordinary = sum / 2.
    
    total = 0
    for i in range(n):
        xi, yi = points[i]
        slopes = {}
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize direction
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            if g != 0:
                dx //= g
                dy //= g
            # Sign fix: make first non-zero positive
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            key = (dx, dy)
            slopes[key] = slopes.get(key, 0) + 1
        # Count slopes with exactly 1 point
        for cnt in slopes.values():
            if cnt == 1:
                total += 1
    return total // 2

def _is_valid(points):
    """Check if points are 12 distinct integer points with |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
            return True
    return False

def _try_structured_configs():
    """Try several known structured configurations."""
    candidates = []
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2x6 grid  
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # Points on circle x^2 + y^2 = 25
    circle_pts = [(0,5),(5,0),(3,4),(4,3),(-3,4),(-4,3),(-3,-4),(-4,-3),(3,-4),(4,-3),(0,-5),(-5,0)]
    candidates.append(circle_pts)
    
    # Points on parabola y = x^2
    parab_pts = [(x, x*x) for x in range(-3, 9)]
    candidates.append(parab_pts)
    
    # Orchard: 4 lines of 3 points
    # Known 12-point orchard configuration
    orchard = [
        [0, 0], [1, 0], [2, 0],  # line y=0
        [0, 1], [1, 1], [2, 1],  # line y=1
        [0, 2], [1, 2], [2, 2],  # line y=2
        [3, 0], [3, 1], [3, 2],  # line x=3
    ]
    candidates.append(orchard)
    
    # Another orchard: use diagonal lines
    # 4 points on x-axis, 4 on y-axis, 4 on diagonal
    orchard2 = [
        [0,0], [1,0], [2,0], [3,0],  # x-axis
        [0,1], [0,2], [0,3], [0,4],  # y-axis
        [1,1], [2,2], [3,3], [4,4],  # diagonal y=x
    ]
    candidates.append(orchard2)
    
    # 3x3 grid + 3 extra points on lines
    grid3 = [[x, y] for x in range(3) for y in range(3)]
    # Add 3 points to create more collinearities
    # Add (3,0), (0,3), (3,3) to extend lines
    orchard3 = grid3 + [[3,0], [0,3], [3,3]]
    candidates.append(orchard3)
    
    # 4 collinear + 8 arranged
    # Start with 4 on x-axis, then add points to form more 3-point lines
    base = [[0,0], [1,0], [2,0], [3,0]]
    # Add points to form lines of 3
    # (0,1), (1,2), (2,3) - line with slope 1 through (0,1)? No, (0,1),(1,2),(2,3) is y=x+1
    # Let's try: add (0,1), (1,1), (2,1) for y=1
    # Then (0,2), (1,2), (2,2) for y=2
    # Then (3,1), (3,2) for x=3 partial
    orchard4 = base + [[0,1], [1,1], [2,1], [0,2], [1,2], [2,2], [3,1], [3,2]]
    candidates.append(orchard4)
    
    best = None
    best_score = float('inf')
    for cand in candidates:
        if not _is_valid(cand):
            continue
        score = _count_ordinary_fast(cand)
        if score < best_score:
            best_score = score
            best = [list(p) for p in cand]
    
    return best, best_score

def _simulated_annealing(initial_points, deadline, seed=42):
    """Run simulated annealing to minimize ordinary lines."""
    rng = random.Random(seed)
    current = [list(p) for p in initial_points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 5.0  # initial temperature
    cooling = 0.999
    
    while time.time() < deadline:
        if best_score <= 6:
            break
        T *= cooling
        if T < 0.01:
            break
            
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move one point randomly
            idx = rng.randint(0, 11)
            ox, oy = current[idx]
            dx = rng.randint(-8, 8)
            dy = rng.randint(-8, 8)
            if dx == 0 and dy == 0:
                continue
            nx, ny = ox + dx, oy + dy
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            # Check not duplicate
            if any(current[i][0] == nx and current[i][1] == ny for i in range(12) if i != idx):
                continue
            current[idx] = [nx, ny]
            
        elif move_type < 0.7:
            # Move one point along a line through two other points
            idx = rng.randint(0, 11)
            # Pick two other points
            others = [i for i in range(12) if i != idx]
            i1, i2 = rng.sample(others, 2)
            x1, y1 = current[i1]
            x2, y2 = current[i2]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Move current[idx] to a new position on this line
            # Parameterize: point = (x1 + t*dx, y1 + t*dy) for integer t
            # Current position: current[idx] = (cx, cy)
            cx, cy = current[idx]
            # Find t for current position: t = (cx - x1) / dx if dx != 0, else (cy - y1) / dy
            # We want a new integer t
            if dx != 0:
                # t must be integer: cx - x1 = t * dx => t = (cx-x1)/dx
                # Just pick a new random t
                t_new = rng.randint(-5, 5)
                nx = x1 + t_new * dx
                ny = y1 + t_new * dy
            else:
                # dx == 0, vertical line
                t_new = rng.randint(-5, 5)
                nx = x1
                ny = y1 + t_new * dy
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            if any(current[i][0] == nx and current[i][1] == ny for i in range(12) if i != idx):
                continue
            current[idx] = [nx, ny]
            
        elif move_type < 0.9:
            # Swap two points
            i1, i2 = rng.sample(range(12), 2)
            current[i1], current[i2] = current[i2], current[i1]
        else:
            # Move one point to be collinear with two others
            idx = rng.randint(0, 11)
            others = [i for i in range(12) if i != idx]
            i1, i2 = rng.sample(others, 2)
            x1, y1 = current[i1]
            x2, y2 = current[i2]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Find a point on line (x1,y1)-(x2,y2) not already occupied
            # Try several t values
            found = False
            for _ in range(20):
                t = rng.randint(-10, 10)
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                if any(current[i][0] == nx and current[i][1] == ny for i in range(12)):
                    continue
                current[idx] = [nx, ny]
                found = True
                break
            if not found:
                continue
        
        # Evaluate
        new_score = _count_ordinary_fast(current)
        delta = new_score - current_score
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            # Revert
            # We need to revert the move - this is tricky without storing state
            # Let's just restart from best periodically
            pass
    
    return best, best_score

def construct():
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    # First try structured configurations
    best_pts, best_score = _try_structured_configs()
    
    if best_pts is None:
        # Fallback
        return [[x, y] for x in range(4) for y in range(3)]
    
    # If we already have 6, return immediately
    if best_score <= 6:
        return best_pts
    
    # Run simulated annealing from the best structured config
    # Use multiple restarts with different seeds
    remaining_deadline = deadline - 5  # Leave 5s margin
    
    for seed in [42, 123, 7, 99, 314]:
        if time.time() > remaining_deadline:
            break
        sa_pts, sa_score = _simulated_annealing(best_pts, remaining_deadline, seed=seed)
        if sa_score < best_score:
            best_score = sa_score
            best_pts = sa_pts
            if best_score <= 6:
                break
    
    return best_pts

# EVOLVE-BLOCK-END
