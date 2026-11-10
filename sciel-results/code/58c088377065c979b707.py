"""Minimize ordinary lines for 12 integer points via local search."""
import time
import random
import math
from collections import defaultdict

DEADLINE = time.time() + 135  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)  # vertical line: x = x1
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build dict: line_key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for v in line_counts.values() if v == 1)
    return ordinary

def _count_ordinary_from_line_map(line_map):
    """Count ordinary lines from a line_map (line_key -> set of indices)."""
    return sum(1 for s in line_map.values() if len(s) == 2)

def _is_valid(points):
    """Check: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    p0, p1, p2 = points[0], points[1], points[2]
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
        return True
    for i in range(3, 12):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _clamp(p, limit=10**9):
    x, y = p
    x = max(-limit, min(limit, x))
    y = max(-limit, min(limit, y))
    return [x, y]

def _fast_move(points, idx, new_pt, line_map):
    """
    Try moving point idx to new_pt. Returns (new_score, updated_line_map) or (None, None).
    Makes a copy of line_map to avoid mutation on rejection.
    """
    n = len(points)
    old_pt = points[idx]
    if new_pt == old_pt:
        return None, None
    # Check distinctness
    for i in range(n):
        if i != idx and points[i] == new_pt:
            return None, None
    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
        return None, None
    
    # Copy line_map
    new_line_map = {k: set(s) for k, s in line_map.items()}
    
    # Collect all lines involving point idx
    affected_lines = set()
    for j in range(n):
        if j != idx:
            key = _line_key(old_pt, points[j])
            affected_lines.add(key)
    
    # Remove idx from those lines
    for key in affected_lines:
        if key in new_line_map:
            new_line_map[key].discard(idx)
            if not new_line_map[key]:
                del new_line_map[key]
    
    # Add new lines
    for j in range(n):
        if j != idx:
            key = _line_key(new_pt, points[j])
            if key not in new_line_map:
                new_line_map[key] = set()
            new_line_map[key].add(idx)
            new_line_map[key].add(j)
    
    # Count ordinary lines
    score = sum(1 for s in new_line_map.values() if len(s) == 2)
    return score, new_line_map

def construct():
    random.seed(42)
    
    # Expanded structured seeds based on known extremal configurations
    structured_seeds = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Points on two lines
        [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)],
        # Points on three parallel lines
        [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)],
        # Hexagon-like
        [[3, 0], [1, 2], [-1, 2], [-3, 0], [-1, -2], [1, -2],
         [6, 0], [2, 4], [-2, 4], [-6, 0], [-2, -4], [2, -4]],
        # 12 points on a circle (approximate with integers)
        [[7, 24], [15, 8], [8, 15], [-7, 24], [-15, 8], [-8, 15],
         [7, -24], [15, -8], [8, -15], [-7, -24], [-15, -8], [-8, -15]],
        # Böröczky-like: 4 points on a line, 4 on another, 4 on a third
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 3], [1, 3], [2, 3], [3, 3]],
        # Points forming many diagonal lines
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 2], [1, 2], [2, 2], [3, 2]],
        # 2x6 grid
        [[x, y] for x in range(6) for y in range(2)],
        # 6x2 grid
        [[x, y] for x in range(2) for y in range(6)],
        # Points on 4 lines of 3
        [[0, 0], [1, 0], [2, 0],
         [0, 1], [1, 1], [2, 1],
         [0, 2], [1, 2], [2, 2],
         [0, 3], [1, 3], [2, 3]],
        # Böröczky configuration: 6 on one line, 6 on another with shared structure
        # Try: 3x3 grid plus 3 extra points to create more collinearities
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2],
         [3, 0], [3, 1], [3, 2]],
        # 4x3 grid
        [[x, y] for x in range(4) for y in range(3)],
        # Symmetric cross pattern
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
         [1, -1], [-1, 1], [2, 0], [0, 2], [0, -2]],
        # 3-3-3-3 on 4 lines forming a grid with diagonals
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
         [0, 2], [1, 2], [2, 2], [0, 3], [1, 3], [2, 3]],
        # Böröczky-type: 5 on line y=0, 5 on line y=1, 2 connecting
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
         [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
         [5, 0], [5, 1]],
        # 12 points: 4 on x-axis, 4 on y-axis, 4 off
        [[-3, 0], [-1, 0], [1, 0], [3, 0],
         [0, -3], [0, -1], [0, 1], [0, 3],
         [2, 2], [-2, 2], [2, -2], [-2, -2]],
        # 3x4 with shifted rows for more diagonal collinearities
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [1, 1], [2, 1], [3, 1], [4, 1],
         [2, 2], [3, 2], [4, 2], [5, 2]],
    ]
    
    best_points = None
    best_score = float('inf')
    
    for s in structured_seeds:
        if time.time() > DEADLINE - 5:
            break
        if not _is_valid(s):
            continue
        sc = count_ordinary_lines(s)
        if sc < best_score:
            best_score = sc
            best_points = [p[:] for p in s]
    
    # Try random starting configurations
    for seed in range(200):
        if time.time() > DEADLINE - 5:
            break
        random.seed(seed + 42)
        points = []
        for _ in range(12):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            points.append([x, y])
        if not _is_valid(points):
            continue
        score = count_ordinary_lines(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)
    
    # Local search with incremental line_map updates
    points = [p[:] for p in best_points]
    line_map = _build_line_map(points)
    score = _count_ordinary_from_line_map(line_map)
    
    # Hill climbing with single-point moves using fast incremental scoring
    step_size = 10
    for iteration in range(80000):
        if time.time() > DEADLINE - 5:
            break
        if score <= 6:
            break
        
        if random.random() < 0.85:
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            dx = random.randint(-step_size, step_size)
            dy = random.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue
            new_pt = [old_x + dx, old_y + dy]
            new_score, new_line_map = _fast_move(points, idx, new_pt, line_map)
            if new_score is None:
                continue
            if new_score <= score:
                points[idx] = new_pt
                line_map = new_line_map
                score = new_score
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
        else:
            # Two-point move
            idx1, idx2 = random.sample(range(12), 2)
            x1, y1 = points[idx1]
            x2, y2 = points[idx2]
            dx1 = random.randint(-step_size, step_size)
            dy1 = random.randint(-step_size, step_size)
            dx2 = random.randint(-step_size, step_size)
            dy2 = random.randint(-step_size, step_size)
            if (dx1 == 0 and dy1 == 0) and (dx2 == 0 and dy2 == 0):
                continue
            new_pt1 = [x1 + dx1, y1 + dy1]
            new_pt2 = [x2 + dx2, y2 + dy2]
            if abs(new_pt1[0]) > 10**9 or abs(new_pt1[1]) > 10**9:
                continue
            if abs(new_pt2[0]) > 10**9 or abs(new_pt2[1]) > 10**9:
                continue
            # Check distinctness
            if new_pt1 == new_pt2:
                continue
            for i in range(12):
                if i != idx1 and i != idx2 and (points[i] == new_pt1 or points[i] == new_pt2):
                    break
            else:
                new_pts = [p[:] for p in points]
                new_pts[idx1] = new_pt1
                new_pts[idx2] = new_pt2
                new_score = count_ordinary_lines(new_pts)
                if new_score <= score:
                    points = new_pts
                    line_map = _build_line_map(points)
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
        
        if iteration % 10000 == 0 and step_size > 1:
            step_size = max(1, step_size // 2)
    
    # Final polish with small steps
    if score > 6 and time.time() < DEADLINE - 5:
        points = [p[:] for p in best_points]
        line_map = _build_line_map(points)
        score = _count_ordinary_from_line_map(line_map)
        for _ in range(50000):
            if time.time() > DEADLINE - 3:
                break
            if score <= 6:
                break
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_pt = [old_x + dx, old_y + dy]
            new_score, new_line_map = _fast_move(points, idx, new_pt, line_map)
            if new_score is None:
                continue
            if new_score <= score:
                points[idx] = new_pt
                line_map = new_line_map
                score = new_score
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
    
    return best_points
