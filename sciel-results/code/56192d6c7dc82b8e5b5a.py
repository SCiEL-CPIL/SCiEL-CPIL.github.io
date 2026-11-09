"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    a = dy
    b = -dx
    c = -a * x1 - b * y1
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _count_ordinary(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    
    ordinary = 0
    for key, cnt in lines.items():
        if cnt == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
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
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if cross_i != 0:
                return True
        return False
    return True

def _get_line_members(points):
    """Return dict mapping line key -> list of point indices on that line."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _delta_count(points, idx, new_pos):
    """
    Compute the change in ordinary line count when moving point idx to new_pos.
    Returns the new total count, or None if invalid.
    """
    if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
        return None
    
    # Check collision
    for j in range(12):
        if j != idx and points[j][0] == new_pos[0] and points[j][1] == new_pos[1]:
            return None
    
    # Build line membership for current config
    line_members = _get_line_members(points)
    
    # Find all lines that point idx is on
    old_lines = set()
    for key, members in line_members.items():
        if idx in members:
            old_lines.add(key)
    
    # Build new line membership
    new_points = [p[:] for p in points]
    new_points[idx] = list(new_pos)
    
    # Check validity (not all collinear)
    if not _valid(new_points):
        return None
    
    # Recompute line members for new config
    new_line_members = _get_line_members(new_points)
    
    # Find all lines that point idx is on in new config
    new_lines = set()
    for key, members in new_line_members.items():
        if idx in members:
            new_lines.add(key)
    
    # Compute old ordinary count
    old_ordinary = 0
    for key, members in line_members.items():
        if len(members) == 2:
            old_ordinary += 1
    
    # Compute new ordinary count
    new_ordinary = 0
    for key, members in new_line_members.items():
        if len(members) == 2:
            new_ordinary += 1
    
    return new_ordinary

def _generate_strategic_candidates(points, idx, limit=200):
    """
    Generate candidate positions for point idx that are likely to create collinearity.
    """
    x, y = points[idx]
    candidates = set()
    
    # 1. Small perturbations
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                candidates.add((nx, ny))
    
    # 2. For each pair of other points (j, k), find integer points on line jk
    #    that are close to current position or extend the line
    others = [j for j in range(12) if j != idx]
    for j in others:
        for k in others:
            if j >= k:
                continue
            xj, yj = points[j]
            xk, yk = points[k]
            dx = xk - xj
            dy = yk - yj
            g = math.gcd(math.gcd(abs(dx), abs(dy)), 1)
            dx_n = dx // g if g != 0 else dx
            dy_n = dy // g if g != 0 else dy
            
            # Generate a few points along this line near the current point
            # Parameterize: (xj + t*dx_n, yj + t*dy_n) for integer t
            # We want points where t is such that the point is near (x, y)
            if dx_n != 0:
                t_approx = (x - xj) / dx_n
            elif dy_n != 0:
                t_approx = (y - yj) / dy_n
            else:
                continue
            
            for t in range(int(t_approx) - 2, int(t_approx) + 3):
                nx = xj + t * dx_n
                ny = yj + t * dy_n
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    # Check it's not too far and not the same as existing points
                    candidates.add((nx, ny))
    
    # 3. Random candidates
    for _ in range(50):
        nx = x + random.randint(-20, 20)
        ny = y + random.randint(-20, 20)
        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
            candidates.add((nx, ny))
    
    return list(candidates)[:limit]

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Uses local search with strategic candidate generation.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    random.seed(42)
    
    # Initial candidates
    candidates_list = [
        [[i, 0] for i in range(11)] + [[0, 1]],  # 11 on line, 1 off
        [[x, y] for x in range(4) for y in range(3)],  # 4x3 grid
        [[x, y] for x in range(6) for y in range(2)],  # 6x2 grid
    ]
    
    best_points = None
    best_count = float('inf')
    
    for pts in candidates_list:
        if not _valid(pts):
            continue
        c = _count_ordinary(pts)
        if c < best_count:
            best_count = c
            best_points = [p[:] for p in pts]
    
    current_points = best_points
    current_count = best_count
    
    # Local search with strategic moves
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        
        for i in range(12):
            if time.time() > deadline:
                break
            
            # Generate strategic candidates
            cand_positions = _generate_strategic_candidates(current_points, i, limit=100)
            
            best_move = None
            best_move_count = current_count
            
            for nx, ny in cand_positions:
                if time.time() > deadline:
                    break
                
                new_count = _delta_count(current_points, i, (nx, ny))
                if new_count is not None and new_count < best_move_count:
                    best_move_count = new_count
                    best_move = (i, nx, ny, new_count)
            
            if best_move:
                idx, nx, ny, new_count = best_move
                current_points[idx] = [nx, ny]
                current_count = new_count
                improved = True
                stuck_count = 0
                
                if current_count <= 6:
                    return current_points
        
        if not improved:
            stuck_count += 1
            
            # Try larger perturbations
            for i in range(12):
                if time.time() > deadline:
                    break
                x, y = current_points[i]
                for _ in range(30):
                    nx = x + random.randint(-50, 50)
                    ny = y + random.randint(-50, 50)
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_count = _delta_count(current_points, i, (nx, ny))
                    if new_count is not None and new_count < current_count:
                        current_points[i] = [nx, ny]
                        current_count = new_count
                        improved = True
                        stuck_count = 0
                        break
                if improved:
                    break
            
            if not improved and stuck_count >= 3:
                # Random restart
                new_pts = []
                for _ in range(12):
                    new_pts.append([random.randint(-15, 15), random.randint(-15, 15)])
                if _valid(new_pts):
                    cnt = _count_ordinary(new_pts)
                    if cnt < current_count:
                        current_points = new_pts
                        current_count = cnt
                        stuck_count = 0
                else:
                    break
    
    return current_points

# EVOLVE-BLOCK-END
