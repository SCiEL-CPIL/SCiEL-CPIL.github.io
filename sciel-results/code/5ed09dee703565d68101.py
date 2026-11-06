"""Optimized: Vectorized evaluation + restricted lattice search + time-bounded local search."""
import time
import numpy as np
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin: return by 55s, limit is 90s.

    # Precompute indices for all triples (i, j, k) with i < j < k
    # This allows vectorized area computation for a given set of points.
    indices = list(combinations(range(30), 3))
    idx_arr = np.array(indices, dtype=np.int64)
    I, J, K = idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]

    def fast_evaluate(points):
        """
        Compute the max multiplicity of any positive doubled area.
        points: list of 30 (x, y) tuples.
        """
        if len(points) != 30:
            return 0
        
        pts = np.array(points, dtype=np.int64)
        # Check distinctness quickly (optional, assumed valid in local search if we check before calling)
        # For safety in initial candidates:
        if len(set(points)) < 30:
            return 0

        x, y = pts[:, 0], pts[:, 1]
        
        # 2*Area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        # Vectorized over all 4060 triples
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Exclude zero areas
        positive_areas = areas[areas > 0]
        
        if len(positive_areas) == 0:
            return 0
            
        # Count multiplicities
        # np.unique is fast enough for 4060 elements
        _, counts = np.unique(positive_areas, return_counts=True)
        return int(counts.max())

    # --- Phase 1: Generate Good Initial Candidates ---
    # We test a small set of structured lattices that are known to produce high multiplicities.
    # Format: (u, v, M, N) where points are i*u + j*v for i in [0, M-1], j in [0, N-1]
    # We center them at (0,0) to maximize space in [-30, 30].
    
    candidates = []
    
    # Common bases: (1,0), (0,1) -> Square grid
    # (1,0), (1,1) -> Sheared grid
    # (1,0), (0,2) -> Stretched grid
    # (1,0), (2,1) -> Another shear
    
    bases = [
        ((1, 0), (0, 1)),
        ((1, 0), (0, 2)),
        ((1, 0), (2, 0)),
        ((1, 0), (1, 1)),
        ((1, 0), (1, 2)),
        ((1, 0), (2, 1)),
        ((1, 1), (1, -1)),
        ((1, 0), (0, 3)),
    ]
    
    dims = [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2)]
    
    best_points = None
    best_score = 0
    
    for u, v in bases:
        for M, N in dims:
            if time.time() > deadline - 10:
                break
                
            pts = []
            valid = True
            # Generate raw lattice points
            for i in range(M):
                for j in range(N):
                    px = i * u[0] + j * v[0]
                    py = i * u[1] + j * v[1]
                    pts.append((px, py))
            
            if len(pts) != 30 or len(set(pts)) < 30:
                continue
                
            # Center the points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            
            centered_pts = [(p[0] - cx, p[1] - cy) for p in pts]
            
            # Check bounds
            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in centered_pts):
                score = fast_evaluate(centered_pts)
                if score > best_score:
                    best_score = score
                    best_points = centered_pts
    
    # Fallback if nothing found (should not happen)
    if best_points is None:
        # Simple grid 5x6
        pts = []
        for i in range(5):
            for j in range(6):
                pts.append((i, j))
        # Center
        cx, cy = 2, 2
        best_points = [(p[0]-cx, p[1]-cy) for p in pts]
        best_score = fast_evaluate(best_points)

    # --- Phase 2: Local Search ---
    # We perform a hill-climbing search.
    # To ensure speed, we limit the number of evaluations.
    # fast_evaluate takes ~2-5ms.
    # In 45s, we can do ~10,000 - 20,000 evaluations.
    
    current_points = list(best_points)
    current_score = best_score
    random.seed(42)
    
    # Neighborhood: move one point by (dx, dy) where dx, dy in [-2, 2]
    # This is a small, dense neighborhood.
    neighborhood = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    
    no_improve_count = 0
    max_no_improve = 3000  # Increased patience to explore more before stopping
    
    while time.time() < deadline - 2:
        if no_improve_count > max_no_improve:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Pick a random move
        dx, dy = random.choice(neighborhood)
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            no_improve_count += 1
            continue
            
        # Check collision
        new_point = (new_x, new_y)
        
        new_points = list(current_points)
        new_points[idx] = new_point
        
        if len(set(new_points)) < 30:
            no_improve_count += 1
            continue
            
        score = fast_evaluate(new_points)
        
        if score > current_score:
            current_score = score
            current_points = new_points
            no_improve_count = 0
        else:
            no_improve_count += 1

    return current_points
