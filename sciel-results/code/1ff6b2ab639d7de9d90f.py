import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary(points):
    """Count ordinary lines."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    # Check not all collinear
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _build_hesse_extended():
    """
    Deterministic construction: Hesse 9 points + 3 strategically placed points.
    
    The 3x3 grid has 12 lines of 3 points each (3 horizontal, 3 vertical, 
    3 with slope 1, 3 with slope -1). We add 3 points to minimize new ordinary lines.
    
    Strategy: Add points that lie on extensions of existing lines to create
    new lines with 3+ points instead of ordinary lines.
    """
    base = [[i, j] for i in range(3) for j in range(3)]
    
    # Try multiple extensions and pick the best
    extensions = [
        [[3, 0], [3, 1], [3, 2]],      # Extend right
        [[0, 3], [1, 3], [2, 3]],      # Extend up
        [[3, 0], [0, 3], [3, 3]],      # Triangle
        [[3, 3], [4, 0], [0, 4]],      # Diagonal + corners
        [[3, 0], [2, 3], [0, 3]],      # Asymmetric triangle
        [[3, 1], [1, 3], [2, 0]],      # Scattered
        [[3, 2], [2, 3], [1, 0]],      # Another scatter
        [[3, 0], [3, 1], [0, 3]],      # L-shape
        [[3, 2], [3, 3], [2, 3]],      # Corner
        [[0, 3], [3, 0], [3, 3]],      # Big triangle
    ]
    
    best_pts = None
    best_score = float('inf')
    
    for ext in extensions:
        pts = base + ext
        if not _is_valid(pts):
            continue
        score = _count_ordinary(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    return best_pts, best_score

def _build_parallel_bundles():
    """
    Alternative: Use parallel line bundles with careful point placement.
    
    Try configurations where points are distributed on lines from multiple
    parallel classes to minimize ordinary lines.
    """
    candidates = []
    
    # 4 lines of 3 points in 4 different directions
    # Horizontal: y=0,1,2,3 with 3 points each
    for offset in [0, 1, 2, 3, 4, 5]:
        pts = []
        for y in range(4):
            for x in range(3):
                pts.append([x + offset * (y % 2), y])
        if len(set(tuple(p) for p in pts)) == 12 and _is_valid(pts):
            candidates.append(pts)
    
    # Mixed: some horizontal, some vertical, some diagonal
    # 3 horizontal lines with 4 points, but shifted
    for shift in [0, 1, 2, 3]:
        pts = []
        for y in range(3):
            for x in range(4):
                pts.append([x + shift * (y % 2), y])
        if len(set(tuple(p) for p in pts)) == 12 and _is_valid(pts):
            candidates.append(pts)
    
    best_pts = None
    best_score = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        score = _count_ordinary(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    return best_pts, best_score

def _build_conic_based():
    """
    Try points on a conic (parabola or ellipse) with careful placement.
    Points on a conic have the property that lines through pairs have
    structured intersections.
    """
    candidates = []
    
    # Points on parabola y = x^2
    for start in range(-3, 4):
        pts = [[x, x * x] for x in range(start, start + 12)]
        if len(set(tuple(p) for p in pts)) == 12 and _is_valid(pts):
            candidates.append(pts)
    
    # Points on ellipse-like curve: x^2 + y^2 = r^2 (approximate with integers)
    # Use parametric: x = r*cos(t), y = r*sin(t) but with integer approximations
    # Simpler: use x^2 + y^2 = 25 (circle radius 5)
    circle_pts = [(x, y) for x in range(-5, 6) for y in range(-5, 6) if x*x + y*y == 25]
    if len(circle_pts) >= 12:
        candidates.append([[x, y] for x, y in circle_pts[:12]])
    
    # x^2 + y^2 = 50
    circle_pts = [(x, y) for x in range(-7, 8) for y in range(-7, 8) if x*x + y*y == 50]
    if len(circle_pts) >= 12:
        candidates.append([[x, y] for x, y in circle_pts[:12]])
    
    best_pts = None
    best_score = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        score = _count_ordinary(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    return best_pts, best_score

def _deterministic_search(points, deadline):
    """
    Deterministic local search: systematically try moving each point to
    positions that reduce ordinary lines, prioritizing positions on existing lines.
    """
    current = [tuple(p) for p in points]
    n = len(current)
    
    # Build initial line counts
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(current[i], current[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    ordinary_count = sum(1 for v in line_counts.values() if v == 1)
    
    best_score = ordinary_count
    best_points = [list(p) for p in current]
    
    # Generate candidate positions: focus on lines through existing points
    coords = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Extend in both directions
            for t in range(-5, 6):
                coords.add((x1 + t * dx, y1 + t * dy))
    
    # Also add nearby positions
    for p in current:
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                coords.add((p[0] + dx, p[1] + dy))
    
    candidate_list = list(coords)
    
    # Deterministic: try all candidates for each point, pick best improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            old_pt = current[i]
            best_cand = None
            best_delta = 0
            best_new_ordinary = ordinary_count
            
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == current[j] for j in range(n) if j != i):
                    continue
                
                # Compute delta in ordinary count
                delta = 0
                
                # Remove lines through old_pt
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(old_pt, current[j])
                    if key is not None:
                        c = line_counts[key]
                        if c == 2:
                            delta -= 1
                        elif c == 3:
                            delta += 1
                        line_counts[key] = c - 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                
                # Add lines through new_pt
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(cand, current[j])
                    if key is not None:
                        c = line_counts.get(key, 0)
                        if c == 1:
                            delta += 1
                        elif c == 2:
                            delta -= 1
                        line_counts[key] = c + 1
                
                new_ordinary = ordinary_count + delta
                
                if new_ordinary < best_new_ordinary:
                    best_new_ordinary = new_ordinary
                    best_delta = delta
                    best_cand = cand
                
                # Undo: remove new lines, add old lines back
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(cand, current[j])
                    if key is not None:
                        c = line_counts[key]
                        line_counts[key] = c - 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(old_pt, current[j])
                    if key is not None:
                        line_counts[key] = line_counts.get(key, 0) + 1
            
            if best_cand is not None and best_new_ordinary < ordinary_count:
                # Apply the best move
                # Remove old lines
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(old_pt, current[j])
                    if key is not None:
                        c = line_counts[key]
                        if c == 2:
                            ordinary_count -= 1
                        elif c == 3:
                            ordinary_count += 1
                        line_counts[key] = c - 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                
                # Add new lines
                current[i] = best_cand
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(best_cand, current[j])
                    if key is not None:
                        c = line_counts.get(key, 0)
                        if c == 1:
                            ordinary_count += 1
                        elif c == 2:
                            ordinary_count -= 1
                        line_counts[key] = c + 1
                
                ordinary_count = best_new_ordinary
                improved = True
                if ordinary_count < best_score:
                    best_score = ordinary_count
                    best_points = [list(p) for p in current]
    
    return best_points, best_score

def construct():
    deadline = time.time() + 135  # Safety margin
    
    # Step 1: Try multiple deterministic constructions
    best_points = None
    best_score = float('inf')
    
    # Hesse extension
    pts1, score1 = _build_hesse_extended()
    if pts1 and score1 < best_score:
        best_score = score1
        best_points = pts1
    
    # Parallel bundles
    pts2, score2 = _build_parallel_bundles()
    if pts2 and score2 < best_score:
        best_score = score2
        best_points = pts2
    
    # Conic-based
    pts3, score3 = _build_conic_based()
    if pts3 and score3 < best_score:
        best_score = score3
    
    # Fallback to 3x4 grid if nothing found
    if best_points is None:
        best_points = [[i, j] for i in range(4) for j in range(3)]
        best_score = _count_ordinary(best_points)
    
    # Step 2: Deterministic local search from best construction
    if time.time() < deadline - 20:
        ls_points, ls_score = _deterministic_search(best_points, deadline - 20)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # Final validation
    if not _is_valid(best_points):
        # Fallback to 3x4 grid
        best_points = [[i, j] for i in range(4) for j in range(3)]
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
