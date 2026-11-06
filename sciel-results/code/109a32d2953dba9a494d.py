"""Search for 30 points maximizing the largest multiplicity of a positive doubled area."""
import time
import random
from collections import Counter

def _score(points):
    """Return the largest multiplicity of a single positive doubled area among all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = abs((xj-xi)*(yk-yi) - (yj-yi)*(xk-xi))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _score_fast(points):
    """Vectorized scoring using numpy."""
    import numpy as np
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    # Compute all doubled areas for triples
    # For each pair (i,j), compute cross product with each k
    best = 0
    area_counter = {}
    for i in range(n):
        for j in range(i+1, n):
            dx = pts[j,0] - pts[i,0]
            dy = pts[j,1] - pts[i,1]
            # Cross product with all other points
            cross = np.abs(dx * (pts[:,1] - pts[i,1]) - dy * (pts[:,0] - pts[i,0]))
            # Only count k > j to avoid duplicates
            valid = cross[j+1:]
            valid = valid[valid > 0]
            if len(valid) > 0:
                vals = valid.tolist()
                for v in vals:
                    area_counter[v] = area_counter.get(v, 0) + 1
                    if area_counter[v] > best:
                        best = area_counter[v]
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Start with a grid configuration that has many equal areas
    # A 6x5 grid gives 30 points
    # Let's try different grid sizes and also some structured configurations
    
    best_points = None
    best_score = 0
    
    # Initial candidates
    candidates = []
    
    # 6x5 grid
    candidates.append([(x, y) for x in range(6) for y in range(5)])
    # 5x6 grid
    candidates.append([(x, y) for x in range(5) for y in range(6)])
    # 3x10 grid
    candidates.append([(x, y) for x in range(3) for y in range(10)])
    # 10x3 grid
    candidates.append([(x, y) for x in range(10) for y in range(3)])
    # 2x15 grid
    candidates.append([(x, y) for x in range(2) for y in range(15)])
    # 15x2 grid
    candidates.append([(x, y) for x in range(15) for y in range(2)])
    
    # Evaluate initial candidates
    for pts in candidates:
        s = _score_fast(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]
    
    # Local search: try to improve by moving individual points
    rng = random.Random(42)
    
    def try_move(points, idx, new_x, new_y):
        """Try replacing point idx with (new_x, new_y). Return new score and whether it's valid."""
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            return -1, False
        # Check distinctness
        for i, (px, py) in enumerate(points):
            if i != idx and px == new_x and py == new_y:
                return -1, False
        new_pts = points[:]
        new_pts[idx] = (new_x, new_y)
        return _score_fast(new_pts), True
    
    # Local search with multiple restarts
    max_iterations = 50000
    iteration = 0
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        pts = best_points[:]
        current_score = best_score
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            # Try random moves
            for _ in range(50):
                if time.time() > deadline:
                    break
                idx = rng.randrange(30)
                # Try moving to a nearby position
                old_x, old_y = pts[idx]
                delta_x = rng.randint(-3, 3)
                delta_y = rng.randint(-3, 3)
                new_x = old_x + delta_x
                new_y = old_y + delta_y
                new_score, valid = try_move(pts, idx, new_x, new_y)
                if valid and new_score > current_score:
                    pts[idx] = (new_x, new_y)
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = pts[:]
        
        # Try larger moves occasionally
        if time.time() < deadline and rng.random() < 0.3:
            idx = rng.randrange(30)
            new_x = rng.randint(-30, 30)
            new_y = rng.randint(-30, 30)
            new_score, valid = try_move(pts, idx, new_x, new_y)
            if valid and new_score > current_score:
                pts[idx] = (new_x, new_y)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = pts[:]
    
    return best_points
