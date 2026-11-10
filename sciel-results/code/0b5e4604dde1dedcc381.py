"""Optimized construction for 16 integer points maximizing 4-collinear lines."""
import time
import random
from itertools import combinations

# Deadline guard: return best found before 150s wall-clock
_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _count_four_lines(points):
    """Count lines containing exactly 4 of the 16 points."""
    n = len(points)
    four_count = 0
    # Use a set of normalized line representations to avoid double counting
    # A line is determined by any two distinct points on it
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = abs(_gcd(dx, dy))
                dx //= g
                dy //= g
                # Ensure canonical sign
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
            # Line: point (x1,y1) with direction (dx,dy)
            # Normalize the point to a canonical representative
            # For a line with direction (dx,dy), we can use the "first" point
            # in some ordering, or use the line equation ax + by + c = 0
            # Let's use the line equation approach
            # a*x + b*y + c = 0 where a = dy, b = -dx, c = dx*y1 - dy*x1
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            # Normalize (a, b, c)
            g = _gcd(_gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Ensure canonical sign: first non-zero of (a,b) is positive
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_key = (a, b, c)
            if line_key not in seen_lines:
                seen_lines.add(line_key)
                # Count points on this line
                cnt = 0
                for k in range(n):
                    px, py = points[k]
                    if a * px + b * py + c == 0:
                        cnt += 1
                if cnt == 4:
                    four_count += 1
    return four_count

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_valid(points):
    """Check: 16 distinct points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    # Distinct
    if len(set(tuple(p) for p in points)) != 16:
        return False
    # Bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    # No five collinear
    n = 16
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            cnt = 0
            for k in range(n):
                px, py = points[k]
                # Cross product to check collinearity
                if (px - x1) * dy - (py - y1) * dx == 0:
                    cnt += 1
            if cnt >= 5:
                return False
    return True

def _score(points):
    """Return (number of 4-lines, validity). Higher 4-lines is better."""
    if not _is_valid(points):
        return -1
    return _count_four_lines(points)

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _random_config():
    """Generate a random valid configuration (may fail, retry)."""
    for _ in range(100):
        points = []
        for _ in range(16):
            p = _random_point()
            if all(tuple(p) != tuple(q) for q in points):
                points.append(p)
        if len(points) == 16:
            # Check no 5-collinear
            valid = True
            n = 16
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        continue
                    cnt = 0
                    for k in range(n):
                        px, py = points[k]
                        if (px - x1) * dy - (py - y1) * dx == 0:
                            cnt += 1
                    if cnt >= 5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return points
    return None

def _simulated_annealing(initial_points, max_iters=50000, t_start=10.0, t_end=0.01):
    """Simulated annealing to maximize 4-collinear lines."""
    points = [list(p) for p in initial_points]
    current_score = _score(points)
    if current_score < 0:
        return points, current_score
    
    best_points = [list(p) for p in points]
    best_score = current_score
    
    n = 16
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Generate a move: pick a random point and perturb it
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        
        # Small perturbation
        delta_x = random.randint(-50, 50)
        delta_y = random.randint(-50, 50)
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        
        # Clamp to bounds
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        # Check distinctness
        if any(tuple([new_x, new_y]) == tuple(points[k]) for k in range(n) if k != idx):
            continue
        
        points[idx] = [new_x, new_y]
        
        new_score = _score(points)
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        if new_score >= current_score:
            # Accept
            points[idx] = [new_x, new_y]
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [list(p) for p in points]
        else:
            # Reject or accept with probability exp((new-old)/t)
            # For discrete objective, use simple acceptance
            if random.random() < 0.05:  # Small probability to accept worse
                points[idx] = [new_x, new_y]
                current_score = new_score
            else:
                # Revert
                points[idx] = [old_x, old_y]
    
    return best_points, best_score

def _local_search(points):
    """Local search: try moving each point to nearby positions."""
    points = [list(p) for p in points]
    current_score = _score(points)
    if current_score < 0:
        return points, current_score
    
    best_points = [list(p) for p in points]
    best_score = current_score
    n = 16
    improved = True
    
    while improved and _time_left() > 5.0:
        improved = False
        for idx in range(n):
            if _time_left() < 5.0:
                break
            x0, y0 = points[idx]
            # Try a set of candidate moves
            candidates = []
            for dx in range(-100, 101, 20):
                for dy in range(-100, 101, 20):
                    nx = x0 + dx
                    ny = y0 + dy
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        if all(tuple([nx, ny]) != tuple(points[k]) for k in range(n) if k != idx):
                            candidates.append((nx, ny))
            
            best_cand_score = -1
            best_cand = None
            for nx, ny in candidates:
                points[idx] = [nx, ny]
                s = _score(points)
                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = (nx, ny)
            
            # Revert
            points[idx] = [x0, y0]
            
            if best_cand and best_cand_score > current_score:
                points[idx] = list(best_cand)
                current_score = best_cand_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [list(p) for p in points]
                improved = True
    
    return best_points, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    best_points = None
    best_score = -1
    
    # Start with the 4x4 grid
    grid_points = [[x, y] for x in range(4) for y in range(4)]
    grid_score = _score(grid_points)
    if grid_score > best_score:
        best_score = grid_score
        best_points = [list(p) for p in grid_points]
    
    # Try multiple random starts with simulated annealing
    num_starts = 20
    for start in range(num_starts):
        if _time_left() < 10.0:
            break
        
        # Generate a random initial configuration
        init = _random_config()
        if init is None:
            # Fallback: perturb the grid
            init = [list(p) for p in grid_points]
            for i in range(16):
                init[i][0] += random.randint(-5, 5)
                init[i][1] += random.randint(-5, 5)
                init[i][0] = max(-1000, min(1000, init[i][0]))
                init[i][1] = max(-1000, min(1000, init[i][1]))
        
        # Run simulated annealing
        sa_points, sa_score = _simulated_annealing(init, max_iters=30000)
        
        # Run local search on the result
        ls_points, ls_score = _local_search(sa_points)
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = [list(p) for p in ls_points]
        
        # Also try local search directly on the initial config
        ls2_points, ls2_score = _local_search(init)
        if ls2_score > best_score:
            best_score = ls2_score
            best_points = [list(p) for p in ls2_points]
    
    # Final polish: run local search on the best found
    if best_points is not None and _time_left() > 10.0:
        final_points, final_score = _local_search(best_points)
        if final_score > best_score:
            best_score = final_score
            best_points = [list(p) for p in final_points]
    
    if best_points is None:
        best_points = grid_points
    
    return best_points
