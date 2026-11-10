import numpy as np
import time
from math import gcd
from itertools import combinations

def _canon(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _canon(A, B, C)

def _incidence_count(points, lines):
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    vals = L[:, 0:1] * P[:, 0:1].T + L[:, 1:2] * P[:, 1:2].T + L[:, 2:3]
    return int(np.count_nonzero(vals == 0))

def _objective(points):
    """
    Compute the objective: sum of top-27 line incidences for the given 54 points.
    Returns (score, lines_array).
    """
    n = len(points)
    # Enumerate all lines through pairs
    line_points = {}  # canonical line -> list of point indices
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            if key in line_points:
                line_points[key].append(j)
            else:
                line_points[key] = [i, j]
    
    if not line_points:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # For each line, count how many of the 54 points lie on it
    # A line is defined by two points; we need to check all 54 points
    # But we can be smarter: for a line through points i,j, the incidence is
    # the number of points k such that (i,j,k) are collinear.
    
    # Actually, let's just count properly.
    # For each unique line (by canonical form), count incidences.
    # We can do this by: for each line key, we know at least 2 points are on it.
    # To count all, we check each of the 54 points.
    
    # But this is O(num_lines * n) which is O(n^3).
    # Alternative: for each point, find all lines through it, then count.
    # Let's use a different approach:
    
    # Build a dict: line_key -> set of point indices
    line_to_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    # Now for each line, we have a set of points we know are on it.
    # But there might be more points on the line that we haven't found via pairs.
    # Wait, no: if a point k is on the line through i and j, then (i,k) and (j,k)
    # are also pairs that generate the same line. So the set should be complete.
    
    # Let's verify: if k is collinear with i and j, then line_from_two_points(i,k) 
    # should equal line_from_two_points(i,j). Yes, because they're the same line.
    
    # So line_to_points[key] should contain all points on that line.
    
    # Count incidences
    counts = []
    for key, pts in line_to_points.items():
        counts.append((len(pts), key))
    
    # Sort by count descending
    counts.sort(key=lambda x: -x[0])
    
    # Take top 27
    top27 = counts[:27]
    score = sum(c for c, _ in top27)
    lines = np.array([k for _, k in top27], dtype=np.int64)
    
    return score, lines

def _local_search(points, grid_points, rng, max_iters=500, deadline=None):
    """
    Local search: try swapping one point with a random grid point.
    """
    current_score, _ = _objective(points)
    P = points.copy()
    G = grid_points
    n_grid = len(G)
    
    # Build a set of current points for fast lookup
    current_set = set(map(tuple, P.tolist()))
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random index to swap
        idx = rng.integers(0, 54)
        old_pt = P[idx]
        
        # Try a few random replacements
        best_new_pt = None
        best_new_score = current_score
        
        for _ in range(5):
            new_idx = rng.integers(0, n_grid)
            new_pt = G[new_idx]
            
            # Check if new_pt is already in P (excluding idx)
            if tuple(new_pt.tolist()) in current_set and tuple(new_pt.tolist()) != tuple(old_pt.tolist()):
                continue
            
            # Try the swap
            P[idx] = new_pt
            new_score, _ = _objective(P)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_pt = new_pt.copy()
            
            P[idx] = old_pt
        
        if best_new_pt is not None:
            # Apply the best swap
            current_set.discard(tuple(old_pt.tolist()))
            current_set.add(tuple(best_new_pt.tolist()))
            P[idx] = best_new_pt
            current_score = best_new_score
        else:
            # No improvement found in this iteration
            # Try a more targeted approach: find the point with lowest "contribution"
            # and try to replace it
            pass
    
    return P, current_score

def _greedy_construct(grid_points, rng, deadline=None):
    """
    Greedy construction: start with 0 points, add the point that most increases the objective.
    This is O(n^4) which is too slow for 54 points.
    
    Alternative: start with a random set and do local search.
    """
    # Instead of pure greedy (too slow), use a hybrid:
    # 1. Start with a structured grid
    # 2. Do local search
    
    # Try a 6x9 grid
    # Or try multiple grid shapes
    
    # For now, just return a random set
    idxs = rng.choice(len(grid_points), size=54, replace=False)
    return grid_points[idxs]

def _structured_points(size_x, size_y, offset_x=0, offset_y=0, scale=1):
    """
    Create a structured grid of points.
    """
    xs = np.arange(size_x) * scale + offset_x
    ys = np.arange(size_y) * scale + offset_y
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.issubdtype(points.dtype, np.integer):
        return False
    if not np.issubdtype(lines.dtype, np.integer):
        return False
    if len(np.unique(points, axis=0)) != 54:
        return False
    seen = set()
    for row in lines:
        A, B, C = int(row[0]), int(row[1]), int(row[2])
        if A == 0 and B == 0:
            return False
        key = _canon(A, B, C)
        if key in seen:
            return False
        seen.add(key)
    return True

def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 580  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Define a grid of candidate points
    # Size 31x31 seems reasonable. Coordinates -15 to 15.
    size = 15
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    grid_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try multiple structured grids and random initializations
    initializations = []
    
    # 1. Structured grids
    for sx in range(4, 10):
        for sy in range(4, 10):
            if sx * sy == 54:
                pts = _structured_points(sx, sy, offset_x=-(sx-1)//2, offset_y=-(sy-1)//2)
                initializations.append(pts)
    
    # 2. Random sets
    for _ in range(20):
        idxs = rng.choice(len(grid_points), size=54, replace=False)
        initializations.append(grid_points[idxs])
    
    # 3. Some more structured patterns
    # Diagonal patterns
    for scale in [1, 2, 3]:
        for offset in range(-5, 6):
            pts = []
            for i in range(54):
                x = (i % 9) * scale + offset
                y = (i // 9) * scale + offset
                pts.append((x, y))
            pts = np.array(pts, dtype=np.int64)
            if len(np.unique(pts, axis=0)) == 54:
                initializations.append(pts)
    
    # Process each initialization
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        # Ensure 54 distinct points
        if len(init_pts) != 54:
            continue
        if len(np.unique(init_pts, axis=0)) != 54:
            continue
        
        # Local search
        P, score = _local_search(init_pts, grid_points, rng, max_iters=300, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_points = P.copy()
            _, best_lines = _objective(P)
    
    # Fallback if something went wrong
    if best_points is None:
        points = np.array([(i, j) for i in range(6) for j in range(9)], dtype=np.int64)
        _, lines = _objective(points)
        best_points = points
        best_lines = lines
    
    if not _is_valid(best_points, best_lines):
        # Fallback to valid grid
        points = np.array([(i, j) for i in range(6) for j in range(9)], dtype=np.int64)
        _, lines = _objective(points)
        best_points = points
        best_lines = lines
    
    return best_points.astype(np.int64), best_lines.astype(np.int64)
