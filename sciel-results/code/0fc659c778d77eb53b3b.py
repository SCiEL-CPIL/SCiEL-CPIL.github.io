import time
import sys
from math import gcd
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    if len(set(tuple(p) for p in points)) != 12:
        return None
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def _hexagon_points():
    """Generate 12 points from a hexagonal lattice pattern."""
    # Regular hexagon vertices scaled by 2, plus center, plus 6 midpoints of edges
    # Hexagon vertices (scaled by 2): (2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2)
    # Center: (0,0)
    # Edge midpoints: (1.5, 1), (-0.5, 2), (-1.5, 1), (-1.5, -1), (-0.5, -2), (0.5, -1)
    # But we need integer coords. Let's use a different hexagonal arrangement.
    
    # Better: Use the hexagonal lattice points within a hexagon of radius 2
    # Points: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  [7 points, radius 1]
    # Plus radius 2: (2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2)  [6 points]
    # That's 13 points. Remove one to get 12.
    
    pts = [
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 2], [-1, 2], [-2, 0], [-1, -2], [1, -2]
    ]
    # Remove one point to get exactly 12
    # Remove (1,-2) to break some symmetry
    return pts[:12]

def _greedy_local_search(points, max_time=60):
    """Greedy local search: try moving each point to a nearby integer position
    that reduces the number of ordinary lines."""
    start = time.time()
    best = list(points)
    best_count = _count_ordinary_lines(best)
    
    # Define a set of candidate moves: small integer displacements
    moves = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx != 0 or dy != 0:
                moves.append((dx, dy))
    
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        for i in range(12):
            if (time.time() - start) > max_time:
                break
            for dx, dy in moves:
                if (time.time() - start) > max_time:
                    break
                new_pts = list(best)
                new_pts[i] = [best[i][0] + dx, best[i][1] + dy]
                
                # Check validity
                if len(set(tuple(p) for p in new_pts)) != 12:
                    continue
                if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in new_pts):
                    continue
                
                count = _count_ordinary_lines(new_pts)
                if count < best_count:
                    best = new_pts
                    best_count = count
                    improved = True
                    break  # Restart the inner loop for this point
    
    return best, best_count

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses hexagonal lattice initialization with greedy local search.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with hexagonal lattice configuration
    init_points = _hexagon_points()
    
    # Validate initial config
    init_count = _try_config(init_points)
    if init_count is None:
        # Fallback to grid
        init_points = [[x, y] for x in range(4) for y in range(3)]
        init_count = _try_config(init_points)
    
    best_points = list(init_points)
    best_count = init_count
    
    # Run greedy local search
    if time.time() < deadline - 10:
        search_points, search_count = _greedy_local_search(best_points, max_time=min(100, deadline - time.time() - 5))
        if search_count < best_count:
            best_points = search_points
            best_count = search_count
    
    # Try a few additional starting configurations with local search
    additional_starts = [
        # 3x4 grid
        [[x, y] for x in range(3) for y in range(4)],
        # 4x3 grid
        [[x, y] for x in range(4) for y in range(3)],
        # Hexagon with different point removed
        [
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 2], [-1, 2], [-2, 0], [1, -2]
        ],
    ]
    
    for pts in additional_starts:
        if time.time() > deadline - 15:
            break
        cnt = _try_config(pts)
        if cnt is not None and cnt < best_count:
            best_points = list(pts)
            best_count = cnt
        if time.time() < deadline - 15:
            sp, sc = _greedy_local_search(pts, max_time=min(20, deadline - time.time() - 5))
            if sc < best_count:
                best_points = sp
                best_count = sc
    
    # Final validation
    final_count = _try_config(best_points)
    if final_count is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
