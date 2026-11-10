import time
import random
import math
from collections import defaultdict

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return (0, 0, 0)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _count_4_lines(points):
    n = len(points)
    if n < 4:
        return 0, False
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] += 1
    count_4 = 0
    has_5 = False
    for c in line_counts.values():
        if c == 4:
            count_4 += 1
        elif c >= 5:
            has_5 = True
    return count_4, has_5

def _evaluate(points):
    return _count_4_lines(points)

def _make_grid_5x5_subset(seed_offset=0):
    """Generate a 5x5 grid subset with 16 points, biased to avoid 5-collinear."""
    random.seed(42 + seed_offset)
    # Start with full 5x5 grid (25 points), remove 9 points strategically
    # To avoid 5-collinear, remove at least one point from each row, col, and main diagonals
    # Strategy: remove points to break long lines
    grid = [[x, y] for x in range(5) for y in range(5)]
    # Remove 9 points: ensure no 5 in a row/col/diag
    # Simple heuristic: remove every 3rd point in a pattern
    to_remove = set()
    # Remove corners and some strategic points
    to_remove.add((0, 0))
    to_remove.add((4, 4))
    to_remove.add((0, 4))
    to_remove.add((4, 0))
    # Remove more to get to 16
    remaining = [p for p in grid if (p[0], p[1]) not in to_remove]
    # Need to remove 5 more
    random.shuffle(remaining)
    for p in remaining[:5]:
        to_remove.add((p[0], p[1]))
    points = [p for p in grid if (p[0], p[1]) not in to_remove]
    return points

def _perturb_point(points, idx, step_size=1):
    """Create a new point list with point idx moved slightly."""
    new_points = [p[:] for p in points]
    ox, oy = new_points[idx]
    dx = random.randint(-step_size, step_size)
    dy = random.randint(-step_size, step_size)
    nx, ny = ox + dx, oy + dy
    if abs(nx) > 1000 or abs(ny) > 1000:
        return None
    # Check distinctness
    for k in range(16):
        if k != idx and new_points[k][0] == nx and new_points[k][1] == ny:
            return None
    new_points[idx] = [nx, ny]
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -1
    
    # Multi-start: try multiple diverse initial configurations
    num_starts = 8
    points_per_start_budget = (deadline - (time.time() - start_time)) / num_starts
    
    for start_idx in range(num_starts):
        if time.time() >= deadline:
            break
        
        # Generate diverse initial config
        if start_idx == 0:
            # 4x4 grid
            current_points = [[x, y] for x in range(4) for y in range(4)]
        elif start_idx == 1:
            # 5x5 grid subset
            current_points = _make_grid_5x5_subset(0)
        elif start_idx == 2:
            # 5x5 grid subset different seed
            current_points = _make_grid_5x5_subset(1)
        elif start_idx == 3:
            # Random near-grid
            random.seed(100 + start_idx)
            current_points = []
            for i in range(16):
                x = random.randint(-3, 3)
                y = random.randint(-3, 3)
                while [x, y] in current_points:
                    x = random.randint(-3, 3)
                    y = random.randint(-3, 3)
                current_points.append([x, y])
        else:
            # More random starts
            random.seed(200 + start_idx * 17)
            current_points = []
            for i in range(16):
                x = random.randint(-5, 5)
                y = random.randint(-5, 5)
                while [x, y] in current_points:
                    x = random.randint(-5, 5)
                    y = random.randint(-5, 5)
                current_points.append([x, y])
        
        # Evaluate initial
        cur_score, is_invalid = _evaluate(current_points)
        if is_invalid:
            # Try to fix by moving points
            for _ in range(100):
                idx = random.randint(0, 15)
                new_pts = _perturb_point(current_points, idx, 2)
                if new_pts is None:
                    continue
                s, inv = _evaluate(new_pts)
                if not inv:
                    current_points = new_pts
                    cur_score = s
                    break
            else:
                continue  # Skip this start if can't fix
        
        if cur_score > best_score:
            best_score = cur_score
            best_points = [p[:] for p in current_points]
        
        # Local search for this start
        start_local_time = time.time()
        local_deadline = min(deadline, start_local_time + points_per_start_budget * 0.8)
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < local_deadline:
            if no_improve_count > max_no_improve:
                break
            
            # Pick a point to move
            idx = random.randint(0, 15)
            # Vary step size
            step = random.choice([1, 1, 2, 2, 3])
            new_points = _perturb_point(current_points, idx, step)
            if new_points is None:
                no_improve_count += 1
                continue
            
            new_score, is_invalid = _evaluate(new_points)
            if is_invalid:
                no_improve_count += 1
                continue
            
            if new_score > cur_score:
                current_points = new_points
                cur_score = new_score
                no_improve_count = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = [p[:] for p in current_points]
            else:
                # Accept worse with small probability
                if random.random() < 0.05:
                    current_points = new_points
                    cur_score = new_score
                else:
                    no_improve_count += 1
    
    # Final cleanup: ensure valid
    if best_points is None:
        # Fallback: 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    # Verify and return
    final_score, is_invalid = _evaluate(best_points)
    if is_invalid:
        # Should not happen, but fallback
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
