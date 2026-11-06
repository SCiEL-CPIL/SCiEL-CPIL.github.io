"""Optimized: Lattice section approach - select 30 points from dense sublattice, optimize for target area."""
import time
import numpy as np
import random
from itertools import combinations
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin

    # Precompute indices for all triples
    indices = list(combinations(range(30), 3))
    idx_arr = np.array(indices, dtype=np.int64)
    I, J, K = idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]

    def fast_evaluate(points):
        """Compute max multiplicity of any positive doubled area."""
        if len(points) != 30:
            return 0
        if len(set(points)) < 30:
            return 0
        
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        positive_areas = areas[areas > 0]
        
        if len(positive_areas) == 0:
            return 0
            
        max_area = int(positive_areas.max())
        if max_area > 20000:
            _, counts = np.unique(positive_areas, return_counts=True)
            return int(counts.max())
        counts = np.bincount(positive_areas.astype(np.int64))
        return int(counts.max())

    def evaluate_target_area(points, target_area):
        """Count how many triples have exactly the given doubled area."""
        if len(points) != 30:
            return 0
        if len(set(points)) < 30:
            return 0
        
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        return int(np.sum(areas == target_area))

    # --- Phase 1: Generate dense sublattice candidates ---
    # Try various grid sizes and spacings to create dense point sets
    best_points = None
    best_score = 0
    best_target = 0
    
    # Dense grids: try different spacings and sizes
    # We want 30+ points in [-30,30]^2
    lattice_configs = []
    
    # Square grids with different spacings
    for spacing in [1, 2, 3]:
        # Max points that fit: (60//spacing + 1)^2
        max_dim = 60 // spacing + 1
        for side in range(5, min(8, max_dim + 1)):
            lattice_configs.append(('square', spacing, side))
    
    # Rectangular grids
    for spacing in [1, 2]:
        max_w = 60 // spacing + 1
        max_h = 60 // spacing + 1
        for w in range(4, min(8, max_w + 1)):
            for h in range(4, min(8, max_h + 1)):
                if w * h >= 30:
                    lattice_configs.append(('rect', spacing, (w, h)))
    
    # Sheared grids
    for spacing in [1, 2]:
        for shear in [1, 2]:
            for w in range(4, 8):
                for h in range(4, 8):
                    if w * h >= 30:
                        lattice_configs.append(('shear', spacing, (w, h, shear)))
    
    random.seed(42)
    random.shuffle(lattice_configs)
    
    for config in lattice_configs:
        if time.time() > deadline - 15:
            break
        
        kind = config[0]
        
        if kind == 'square':
            _, spacing, side = config
            pts = []
            for i in range(side):
                for j in range(side):
                    pts.append((i * spacing, j * spacing))
        elif kind == 'rect':
            _, spacing, (w, h) = config
            pts = []
            for i in range(w):
                for j in range(h):
                    pts.append((i * spacing, j * spacing))
        else:  # shear
            _, spacing, (w, h, shear) = config
            pts = []
            for i in range(w):
                for j in range(h):
                    px = i * spacing + j * shear
                    py = j * spacing
                    pts.append((px, py))
        
        if len(pts) < 30:
            continue
        
        # Check if all points are distinct
        if len(set(pts)) < len(pts):
            continue
        
        # Center the points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        
        centered_pts = [(p[0] - cx, p[1] - cy) for p in pts]
        
        # Check bounds
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in centered_pts):
            continue
        
        # Select 30 points from this lattice
        # Strategy: Try different subsets - random, greedy, etc.
        if len(centered_pts) == 30:
            selected = centered_pts
        else:
            # Random selection with some structure
            random.shuffle(centered_pts)
            selected = centered_pts[:30]
        
        # Evaluate and find best target area
        score = fast_evaluate(selected)
        if score > best_score:
            best_score = score
            best_points = selected
            # Find the target area that achieves this
            pts_arr = np.array(selected, dtype=np.int64)
            x, y = pts_arr[:, 0], pts_arr[:, 1]
            xi, yi = x[I], y[I]
            xj, yj = x[J], y[J]
            xk, yk = x[K], y[K]
            areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
            positive_areas = areas[areas > 0]
            if len(positive_areas) > 0:
                _, counts = np.unique(positive_areas, return_counts=True)
                best_target = int(positive_areas[np.argmax(counts)])
    
    # Fallback
    if best_points is None:
        pts = [(i, j) for i in range(5) for j in range(6)]
        cx, cy = 2, 2
        best_points = [(p[0]-cx, p[1]-cy) for p in pts]
        best_score = fast_evaluate(best_points)
        best_target = 2

    # --- Phase 2: Local search optimizing for target area ---
    # We now do local search that specifically tries to increase the count
    # of triangles with area == best_target
    
    current_points = list(best_points)
    current_score = best_score
    current_target = best_target
    best_overall_points = list(current_points)
    best_overall_score = current_score
    
    # Precompute which triples have the target area for current configuration
    def get_target_triples(points, target):
        """Return set of triple indices that have the target area."""
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        return set(np.where(areas == target)[0].tolist())
    
    # Neighborhood for local search
    small_moves = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]
    medium_moves = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 3:
        if no_improve > max_no_improve:
            break
        
        # Pick random point and move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Choose move size based on stagnation
        if no_improve > 800:
            moves = medium_moves
        else:
            moves = small_moves
        
        dx, dy = random.choice(moves)
        new_x = old_x + dx
        new_y = old_y + dy
        
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            no_improve += 1
            continue
        
        new_point = (new_x, new_y)
        new_points = list(current_points)
        new_points[idx] = new_point
        
        if len(set(new_points)) < 30:
            no_improve += 1
            continue
        
        # Evaluate: count triples with target area
        target_count = evaluate_target_area(new_points, current_target)
        
        # Also check if we should switch to a different target area
        full_score = fast_evaluate(new_points)
        
        if full_score > current_score:
            current_score = full_score
            current_points = new_points
            no_improve = 0
            # Update target
            pts_arr = np.array(current_points, dtype=np.int64)
            x, y = pts_arr[:, 0], pts_arr[:, 1]
            xi, yi = x[I], y[I]
            xj, yj = x[J], y[J]
            xk, yk = x[K], y[K]
            areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
            positive_areas = areas[areas > 0]
            if len(positive_areas) > 0:
                _, counts = np.unique(positive_areas, return_counts=True)
                current_target = int(positive_areas[np.argmax(counts)])
            
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_points = list(current_points)
        elif target_count > evaluate_target_area(current_points, current_target):
            # Improved target count even if full score didn't improve
            # Accept with some probability (simulated annealing-like)
            if random.random() < 0.3:
                current_points = new_points
                no_improve = 0
        else:
            no_improve += 1
    
    return best_overall_points
