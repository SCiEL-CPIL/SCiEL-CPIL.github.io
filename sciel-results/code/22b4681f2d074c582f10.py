# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    # Find GCD of A, B, C
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero entry is positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Count total incidences between points and lines.
    points: (N, 2) array
    lines: (K, 3) array
    Returns: total incidences (int)
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _generate_line_pool_for_points(points, max_coord=15):
    """
    Generate a diverse pool of lines likely to be useful for the given points.
    Includes lines defined by pairs of points and grid-like lines.
    """
    pool = set()
    n = len(points)
    
    # 1. Lines defined by pairs of points
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                pool.add(key)
                
    # 2. Grid lines and diagonals near the points
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    # Expand range slightly
    margin = 5
    min_x -= margin; max_x += margin
    min_y -= margin; max_y += margin
    
    # Vertical and horizontal
    for c in range(min_x, max_x + 1):
        pool.add(_normalize_line(1, 0, -c))
        pool.add(_normalize_line(0, 1, -c))
        
    # Diagonals
    for c in range(min_x + min_y, max_x + max_y + 1):
        pool.add(_normalize_line(1, 1, -c))
    for c in range(min_x - max_y, max_x - min_y + 1):
        pool.add(_normalize_line(1, -1, c))
        
    # Other simple slopes
    slopes = [(2, 1), (1, 2), (3, 1), (1, 3), (3, 2), (2, 3), (2, -1), (1, -2), (3, -1), (1, -3)]
    for A, B in slopes:
        # Estimate C range
        # y = -A/B x - C/B => C = -(A x + B y)
        # Range of A x + B y
        min_val = A * min_x + B * min_y
        max_val = A * max_x + B * max_y
        if A < 0:
            min_val, max_val = max_val, min_val
        if B < 0:
            min_val, max_val = max_val, min_val
            
        c_range = int(max_val - min_val) + 10
        center = int((min_val + max_val) / 2)
        
        for c in range(center - c_range//2, center + c_range//2 + 1):
             pool.add(_normalize_line(A, B, c))

    # 3. Random lines with small coeffs
    for _ in range(1000):
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-30, 30)
        if A != 0 or B != 0:
            pool.add(_normalize_line(A, B, C))
            
    return list(pool)

def _constructive_greedy_with_sa(points, max_time=500):
    """
    1. Generate a pool of candidate lines.
    2. Greedily select 27 lines maximizing total incidences.
    3. Use Simulated Annealing to improve the selection.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    pool = _generate_line_pool_for_points(points)
    if not pool:
        # Fallback
        pool = [tuple([random.randint(-10, 10), random.randint(-10, 10), random.randint(-100, 100)]) for _ in range(100)]
        pool = list(set(pool))
        
    # Precompute incidence counts for all lines in pool
    # pool_counts[line] = number of points on this line
    pool_counts = {}
    for line in pool:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        pool_counts[line] = np.sum(vals == 0)
        
    # Sort pool by count descending
    sorted_pool = sorted(pool_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Greedy initialization
    current_lines = []
    current_score = 0
    used_lines = set()
    
    # We want to maximize sum of counts.
    # However, simply taking top 27 is not always optimal if they cover the same points?
    # No, the objective is sum of incidences. Incidences are additive.
    # Score = sum(count(line_k)).
    # So the optimal solution IS the top 27 lines by individual count!
    # Wait, is that true?
    # "I is the number of (point, line) pairs with A*x + B*y + C == 0"
    # Yes, I = sum_{p in P} sum_{l in L} [p in l] = sum_{l in L} sum_{p in P} [p in l] = sum_{l in L} count(l).
    # Since the lines are independent in the summation, the optimal set of 27 lines is simply the 27 lines with the highest individual counts.
    
    # So why did the previous program do alternating minimization?
    # Because the POINTS were also being optimized.
    # If we fix points, the best lines are just the top 27.
    # If we fix lines, the best points are those that lie on the most lines.
    
    # So the strategy should be:
    # 1. Fix points. Find top 27 lines.
    # 2. Fix those 27 lines. Find best 54 points.
    # 3. Repeat.
    
    # My previous "greedy" was actually just "take top 27".
    # Let's implement the alternating minimization more effectively.
    # The "Simulated Annealing" part should be on the POINTS, not the lines,
    # because given points, lines are deterministic (top 27).
    
    # Let's refine:
    # 1. Start with a grid of points.
    # 2. Get top 27 lines.
    # 3. Optimize points using SA to maximize sum of counts on these 27 lines.
    # 4. Re-get top 27 lines for new points.
    # 5. Repeat.
    
    pass # This function structure is misleading, let's write the real logic in the main search.

def _optimize_points_sa(points, lines, max_time=10):
    """
    Optimize points using Simulated Annealing to maximize incidences with fixed lines.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    def score_point(pt):
        x, y = pt
        vals = A * x + B * y + C
        return np.sum(vals == 0)
    
    def total_score(pts):
        return _count_incidences(pts, lines)
    
    current_points = points.copy()
    current_score = total_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    used_points = set([tuple(p) for p in current_points])
    
    while time.time() < deadline - 1:
        # Pick a random point to move
        idx = random.randint(0, len(current_points) - 1)
        old_pt = current_points[idx]
        
        # Generate new position
        # Try to move to a nearby point or a random point in the bounding box
        if random.random() < 0.5:
            # Local move
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            new_pt = (old_pt[0] + dx, old_pt[1] + dy)
        else:
            # Global move within bounds
            min_x, max_x = np.min(current_points[:, 0]), np.max(current_points[:, 0])
            min_y, max_y = np.min(current_points[:, 1]), np.max(current_points[:, 1])
            new_pt = (random.randint(min_x - 5, max_x + 5), random.randint(min_y - 5, max_y + 5))
            
        if new_pt in used_points:
            T *= cooling
            continue
            
        # Calculate delta
        old_count = score_point(old_pt)
        new_count = score_point(new_pt)
        delta = new_count - old_count
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            used_points.remove(tuple(old_pt))
            used_points.add(new_pt)
            current_points[idx] = new_pt
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        T *= cooling
        
    return best_points, best_score

def _get_top_lines(points, k=27):
    """
    Find the k lines with the most incidences among all lines defined by pairs of points
    and some heuristic lines.
    """
    n = len(points)
    
    # Candidate lines
    candidates = set()
    
    # Lines from pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                candidates.add(key)
                
    # Add grid lines
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    for c in range(min_x - 2, max_x + 3):
        candidates.add(_normalize_line(1, 0, -c))
        candidates.add(_normalize_line(0, 1, -c))
    for c in range(min_x + min_y - 4, max_x + max_y + 5):
        candidates.add(_normalize_line(1, 1, -c))
        candidates.add(_normalize_line(1, -1, c))
        
    # Count incidences
    counts = {}
    for line in candidates:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        counts[line] = np.sum(vals == 0)
        
    # Sort and take top k
    sorted_lines = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_keys = [key for key, _ in sorted_lines[:k]]
    
    # Pad if necessary
    if len(top_keys) < k:
        existing = set(top_keys)
        # Add random lines
        while len(top_keys) < k:
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            C = random.randint(-50, 50)
            key = _normalize_line(A, B, C)
            if key not in existing and key != (0,0,0):
                top_keys.append(key)
                existing.add(key)
                
    return top_keys

def _constructive_search(max_time=580):
    """
    Alternating Minimization:
    1. Start with a good grid (e.g., 6x9 or 9x6).
    2. Get top 27 lines.
    3. Optimize points using SA to maximize incidences with these lines.
    4. Re-get top 27 lines.
    5. Repeat until time runs out.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initial seed: 6x9 grid (or 9x6)
    # Let's try 6x9
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = np.array(_get_top_lines(points, 27), dtype=np.int64)
    
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Alternating optimization loop
    while time.time() < deadline - 20:
        time_left = deadline - time.time()
        if time_left < 30:
            break
            
        # Allocate time for point optimization
        # We want to optimize points for the current lines
        chunk_time = min(15, time_left / 2 - 5)
        
        # Optimize points
        points, score_p = _optimize_points_sa(points, lines, max_time=chunk_time)
        
        # Get new top lines
        lines = np.array(_get_top_lines(points, 27), dtype=np.int64)
        
        # Update best
        current_score = _count_incidences(points, lines)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            best_lines = lines.copy()
            
        # Optional: Optimize lines? No, lines are deterministic given points.
        # But maybe we can perturb the points differently?
        # The SA already handles perturbation.
        
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses alternating minimization with Simulated Annealing for points.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    try:
        best_points, best_lines, best_score = _constructive_search(max_time=timeout - 10)
    except Exception:
        # Fallback to simple grid if something goes wrong
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = np.array(_get_top_lines(best_points, 27), dtype=np.int64)
        
    # Final validation and formatting
    points_out = best_points.astype(np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Ensure shapes
    if points_out.shape != (54, 2):
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    if lines_out.shape != (27, 3):
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for duplicates in points
    pts_set = set([tuple(p) for p in points_out])
    if len(pts_set) < 54:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for invalid lines
    for l in lines_out:
        if l[0] == 0 and l[1] == 0:
            xs = np.arange(1, 7)
            ys = np.arange(1, 10)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
            break
            
    # Check for duplicate lines
    lines_set = set([tuple(l) for l in lines_out])
    if len(lines_set) < 27:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)

    return points_out, lines_out

# EVOLVE-BLOCK-END
