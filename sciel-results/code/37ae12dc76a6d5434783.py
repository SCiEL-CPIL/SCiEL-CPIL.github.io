import math
import time
from itertools import combinations, permutations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, dx)
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def _generate_3x4():
    return [[x, y] for x in range(4) for y in range(3)]

def _generate_structured():
    pts = []
    for x in range(4):
        pts.append([x, 0])
    for y in range(1, 5):
        pts.append([0, y])
    for x in range(1, 5):
        pts.append([x, 5 - x])
    seen = set()
    unique = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    return unique

def _generate_line_config(partition):
    """
    Given a partition of 12 into parts >= 3 (collinear groups),
    construct integer points on those lines.
    We use a systematic approach: place each group on a distinct line
    with integer coordinates, ensuring no two points coincide.
    """
    points = []
    # Assign each group to a different line direction/offset
    # Use lines: y = m*x + b for various m, b
    # To keep it simple and ensure integrality, use:
    # Group i: points on line y = i * x + offset_i, spaced out
    
    # Better: use a grid-like approach with distinct slopes
    # For partition [4,4,4]: three lines with 4 points each
    # For partition [3,3,3,3]: four lines with 3 points each
    # For partition [4,3,3,2] - wait, parts must be >=3 for non-ordinary
    # Actually, the partition defines which points are collinear in groups of >=3.
    # Points NOT in any collinear group of >=3 are on ordinary lines.
    
    # Strategy: For each part in partition, create a line with that many points.
    # Use distinct lines (different slopes/intercepts) to avoid accidental collinearity.
    
    line_defs = []
    used_slopes = set()
    for idx, size in enumerate(partition):
        # Choose a unique slope and intercept
        # Use slope = idx, intercept = idx * 100 to separate lines
        m = idx
        b = idx * 100
        # Points: (x, m*x + b) for x in some range
        # Choose x values to keep coordinates small
        x_start = -size // 2
        for k in range(size):
            x = x_start + k
            y = m * x + b
            line_defs.append([x, y])
    
    # Check for duplicates
    seen = set()
    unique = []
    for p in line_defs:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    
    return unique

def _generate_pappus_like():
    """
    Try to construct a configuration with 6 ordinary lines using
    a structured approach based on 3 lines with 4 points each,
    carefully chosen to minimize accidental collinearities.
    """
    # Three lines: L1, L2, L3
    # L1: y = 0, x = 0,1,2,3
    # L2: y = x, x = 0,1,2,3 -> but (0,0) is on L1
    # We need distinct points. Let's use:
    # L1: y = 0, points (0,0), (1,0), (2,0), (3,0)
    # L2: y = x + 10, points (0,10), (1,11), (2,12), (3,13)
    # L3: y = -x + 20, points (0,20), (1,19), (2,18), (3,17)
    
    pts = []
    # L1
    for x in range(4):
        pts.append([x, 0])
    # L2
    for x in range(4):
        pts.append([x, x + 10])
    # L3
    for x in range(4):
        pts.append([x, -x + 20])
    
    # Check distinctness
    seen = set()
    unique = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    
    return unique

def _generate_pencil_config():
    """
    Configuration based on a pencil of lines through a common point,
    plus additional structure to create collinear triples.
    """
    # Use 4 lines through origin with 3 points each, plus 2 more
    # But we need exactly 12 distinct points.
    # Lines through origin: y = m*x
    # For m=0: (1,0), (2,0), (3,0)
    # For m=1: (1,1), (2,2), (3,3)
    # For m=-1: (1,-1), (2,-2), (3,-3)
    # For m=2: (1,2), (2,4), (3,6)
    # That's 12 points. Check for accidental collinearities.
    
    pts = []
    slopes = [0, 1, -1, 2]
    for m in slopes:
        for x in [1, 2, 3]:
            pts.append([x, m * x])
    
    # Check distinctness
    seen = set()
    unique = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    
    return unique

def _generate_grid_perturbed():
    """
    Start with a 3x4 grid and perturb points to break accidental collinearities
    while preserving the main 3+3+3+3 collinear structure.
    """
    pts = [[x, y] for x in range(4) for y in range(3)]
    # The 3x4 grid has rows of 4 and columns of 3.
    # Rows: y=0,1,2 each have 4 points -> 3 lines with 4 points
    # Columns: x=0,1,2,3 each have 3 points -> 4 lines with 3 points
    # Total non-ordinary lines: 3 + 4 = 7
    # But there may be other collinearities.
    # Let's just use the grid as is.
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple candidate configurations
    candidates = []
    
    # Standard 3x4 grid
    candidates.append(_generate_3x4())
    
    # Structured 3-line config
    candidates.append(_generate_structured())
    
    # Pappus-like
    candidates.append(_generate_pappus_like())
    
    # Pencil config
    candidates.append(_generate_pencil_config())
    
    # Grid perturbed (same as 3x4 for now)
    candidates.append(_generate_grid_perturbed())
    
    # Try various partitions
    partitions = [
        [4, 4, 4],
        [3, 3, 3, 3],
        [4, 3, 3, 2],  # Wait, 2 is not >=3, so this has an ordinary point
        [5, 4, 3],
        [6, 3, 3],
        [7, 5],
        [8, 4],
        [9, 3],
        [10, 2],
        [11, 1],
    ]
    # Filter to only valid partitions (all parts >= 3, or handle singles/doubles)
    # Actually, for minimizing ordinary lines, we want as many points as possible
    # in lines with >= 3 points.
    
    for part in partitions:
        if sum(part) == 12:
            pts = _generate_line_config(part)
            if len(pts) == 12:
                candidates.append(pts)
    
    # Evaluate all candidates
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        # Check all distinct
        seen = set()
        valid = True
        for p in cand:
            t = tuple(p)
            if t in seen:
                valid = False
                break
            seen.add(t)
        if not valid:
            continue
        # Check bounds
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in cand):
            continue
        # Check not all collinear
        if len(cand) >= 3:
            p0, p1, p2 = cand[0], cand[1], cand[2]
            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p0[0]
            dy2 = p2[1] - p0[1]
            if dx1 * dy2 == dy1 * dx2:
                # Check if all are collinear
                all_collinear = True
                for i in range(3, 12):
                    dx3 = cand[i][0] - p0[0]
                    dy3 = cand[i][1] - p0[1]
                    if dx1 * dy3 != dy1 * dx3:
                        all_collinear = False
                        break
                if all_collinear:
                    continue
        
        s = _count_ordinary(cand)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        best_points = _generate_3x4()
        best_score = _count_ordinary(best_points)
    
    # If we found 6, we're done
    if best_score <= 6:
        return best_points
    
    # Otherwise, do a limited local search to try to improve
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random point and try to move it
        idx = (it * 7 + 3) % 12  # Deterministic pseudo-random
        old_x, old_y = current_points[idx]
        
        # Try moving to create a new collinearity
        # Pick two other points and try to place on their line
        i = (it * 11 + 5) % 12
        j = (it * 13 + 7) % 12
        if i == idx or j == idx:
            continue
        if i == j:
            continue
        
        p1 = current_points[i]
        p2 = current_points[j]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0 and dy == 0:
            continue
        
        g = _gcd(dx, dy)
        if g == 0:
            continue
        dxr = dx // g
        dyr = dy // g
        
        # Try a few positions on this line
        for t in [-3, -2, -1, 1, 2, 3]:
            nx = p1[0] + t * dxr
            ny = p1[1] + t * dyr
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            
            # Check distinctness
            is_distinct = True
            for k in range(12):
                if k != idx and current_points[k] == [nx, ny]:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            
            current_points[idx] = [nx, ny]
            new_score = _count_ordinary(current_points)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                break
            else:
                current_points[idx] = [old_x, old_y]
        
        if best_score <= 6:
            break
    
    return best_points
