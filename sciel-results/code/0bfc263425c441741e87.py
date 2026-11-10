"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _normalize_line(p1, p2):
    """Return a canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Ensure consistent sign: first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dx*y - dy*x = c
    c = dx * y1 - dy * x1
    return (dx, dy, c)

def _point_on_line(point, line):
    """Check if point is on line defined by canonical tuple (dx, dy, c)."""
    a, b, c = line
    x, y = point
    # Equation: a*y - b*x = c
    return a * y - b * x == c

def _line_counts_from_points(points):
    """Calculate point counts per line for a given set of points."""
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            line = _normalize_line(points[i], points[j])
            line_counts[line] += 1
    
    # Convert pair counts to point counts
    point_counts = {}
    for line, pairs in line_counts.items():
        disc = 1 + 8 * pairs
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc and (1 + sqrt_disc) % 2 == 0:
            k = (1 + sqrt_disc) // 2
            point_counts[line] = k
    return point_counts

def _score_and_validity(points):
    """
    Returns (score, is_valid).
    Score: number of lines with exactly 4 points.
    is_valid: False if any line has >= 5 points.
    """
    if len(points) != 16:
        return -1, False
    
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return -1, False
        if abs(x) > 1000 or abs(y) > 1000:
            return -1, False
        if (x, y) in seen:
            return -1, False
        seen.add((x, y))
        
    pc = _line_counts_from_points(points)
    score = 0
    for line, count in pc.items():
        if count >= 5:
            return 0, False
        if count == 4:
            score += 1
    return score, True

def _build_csp_points(deadline, seed=None):
    """
    Build 16 points using constraint satisfaction search.
    At each step, try candidate positions and prune if they create 5+ collinear.
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with a base configuration and add points one by one
    # We'll try to build from a 4x4 grid pattern but with perturbations
    
    # Candidate positions: a grid from -15 to 15
    candidates = []
    for x in range(-15, 16):
        for y in range(-15, 16):
            candidates.append([x, y])
    
    random.shuffle(candidates)
    
    best_points = None
    best_score = -1
    
    # Try multiple random orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        points = []
        line_counts = defaultdict(int)  # line -> number of points on it
        
        for candidate in candidates:
            if len(points) >= 16:
                break
            
            x, y = candidate
            if (x, y) in [tuple(p) for p in points]:
                continue
            
            # Check if adding this point would create 5+ collinear
            valid = True
            new_lines = {}
            for existing in points:
                line = _normalize_line(candidate, existing)
                new_lines[line] = new_lines.get(line, 0) + 1
            
            # Check if any line would have 5+ points
            for line, count in new_lines.items():
                current_count = line_counts.get(line, 0)
                if current_count + count >= 5:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also check if this point is on a line that already has 4 points
            # (adding would make 5)
            for line in new_lines:
                if line_counts.get(line, 0) >= 4:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Accept this point
            points.append([x, y])
            for line in new_lines:
                line_counts[line] = line_counts.get(line, 0) + 1
        
        if len(points) == 16:
            score, is_valid = _score_and_validity(points)
            if is_valid and score > best_score:
                best_score = score
                best_points = [list(p) for p in points]
    
    return best_points, best_score

def _local_search(points, max_iters=2000, time_limit=None):
    """Local search to improve a valid configuration."""
    current = [list(p) for p in points]
    current_score, is_valid = _score_and_validity(current)
    if not is_valid:
        return current, current_score

    best = [list(p) for p in current]
    best_score = current_score
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_point = list(current[idx])
        
        # Try to move to a nearby position
        new_point = [old_point[0] + random.randint(-8, 8), 
                     old_point[1] + random.randint(-8, 8)]
        new_point[0] = max(-1000, min(1000, new_point[0]))
        new_point[1] = max(-1000, min(1000, new_point[1]))
        
        if new_point == old_point or new_point in current:
            continue
        
        current[idx] = new_point
        new_score, new_valid = _score_and_validity(current)
        
        if new_valid and new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            current[idx] = old_point
    
    return best, best_score

def _generate_grid_config(w, h, offset_x=0, offset_y=0):
    """Generate a w x h grid configuration."""
    points = []
    for i in range(w):
        for j in range(h):
            points.append([i + offset_x, j + offset_y])
    
    if len(points) > 16:
        random.shuffle(points)
        points = points[:16]
    elif len(points) < 16:
        attempts = 0
        while len(points) < 16 and attempts < 1000:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if [x, y] not in points:
                points.append([x, y])
            attempts += 1
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = -1
    
    # Strategy 1: CSP-based construction
    csp_pts, csp_score = _build_csp_points(deadline - 20, seed=42)
    if csp_pts and csp_score > best_score:
        best_score = csp_score
        best_points = csp_pts
    
    # Strategy 2: Try grid-based initializations with local search
    initializers = []
    initializers.append(_generate_grid_config(4, 4, -2, -2))
    initializers.append(_generate_grid_config(3, 5, -1, -2))
    initializers.append(_generate_grid_config(5, 3, -2, -1))
    initializers.append(_generate_grid_config(2, 8, -1, -4))
    initializers.append(_generate_grid_config(8, 2, -4, -1))
    
    # Add some random dense configurations
    for _ in range(3):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-12, 12), random.randint(-12, 12)]
            if p not in pts:
                pts.append(p)
        initializers.append(pts)
    
    for init_pts in initializers:
        if time.time() > deadline - 10:
            break
        
        improved, score = _local_search(init_pts, max_iters=1500, time_limit=deadline - 5)
        
        if score > best_score:
            best_score = score
            best_points = [list(p) for p in improved]
    
    # Strategy 3: If we have a good solution, do deeper polish
    if best_points and best_score >= 10:
        polished, pol_score = _local_search(best_points, max_iters=3000, time_limit=deadline - 5)
        if pol_score > best_score:
            best_score = pol_score
            best_points = [list(p) for p in polished]
    
    # Final validation
    if best_points is None:
        return _generate_grid_config(4, 4, -2, -2)
        
    final_score, is_valid = _score_and_validity(best_points)
    if not is_valid:
        return _generate_grid_config(4, 4, -2, -2)
        
    return best_points

# EVOLVE-BLOCK-END
