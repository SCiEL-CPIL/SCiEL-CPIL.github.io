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
    n_triples = len(indices)

    # For incremental evaluation: precompute for each point index, which triple indices contain it
    # triple_of_point[i] = list of triple indices where point i appears
    triple_of_point = [[] for _ in range(30)]
    for t_idx in range(n_triples):
        for p in (I[t_idx], J[t_idx], K[t_idx]):
            triple_of_point[p].append(t_idx)

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

    def incremental_evaluate(points, changed_idx, old_area_arr):
        """
        After changing point `changed_idx`, recompute only affected triples.
        Returns (new_full_score, new_target_count_for_current_target, new_area_arr)
        """
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        
        # Recompute areas for triples involving changed_idx
        affected = triple_of_point[changed_idx]
        new_areas = old_area_arr.copy()
        
        for t in affected:
            i, j, k = I[t], J[t], K[t]
            xi, yi = x[i], y[i]
            xj, yj = x[j], y[j]
            xk, yk = x[k], y[k]
            new_areas[t] = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        positive_areas = new_areas[new_areas > 0]
        if len(positive_areas) == 0:
            return 0, 0, new_areas
        
        max_area = int(positive_areas.max())
        if max_area > 20000:
            _, counts = np.unique(positive_areas, return_counts=True)
            full_score = int(counts.max())
        else:
            counts = np.bincount(positive_areas.astype(np.int64))
            full_score = int(counts.max())
        
        return full_score, full_score, new_areas

    # --- Phase 1: Generate dense sublattice candidates ---
    best_points = None
    best_score = 0
    best_target = 0
    
    # Dense grids: try different spacings and sizes
    lattice_configs = []
    
    # Square grids with different spacings
    for spacing in [1, 2, 3]:
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
        
        if len(set(pts)) < len(pts):
            continue
        
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        
        centered_pts = [(p[0] - cx, p[1] - cy) for p in pts]
        
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in centered_pts):
            continue
        
        if len(centered_pts) == 30:
            selected = centered_pts
        else:
            random.shuffle(centered_pts)
            selected = centered_pts[:30]
        
        score = fast_evaluate(selected)
        if score > best_score:
            best_score = score
            best_points = selected
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

    # --- Phase 2: Multi-restart local search with incremental evaluation ---
    best_overall_points = list(best_points)
    best_overall_score = best_score
    
    # Neighborhood for local search
    small_moves = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]
    medium_moves = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    large_moves = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
    extra_large_moves = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if (dx, dy) != (0, 0)]
    
    def run_local_search(initial_points, max_iterations=None):
        """Run local search starting from initial_points."""
        current_points = list(initial_points)
        current_score = fast_evaluate(current_points)
        
        # Compute initial area array
        pts_arr = np.array(current_points, dtype=np.int64)
        x, y = pts_arr[:, 0], pts_arr[:, 1]
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        current_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        no_improve = 0
        max_no_improve = 3000
        iteration = 0
        local_best_points = list(current_points)
        local_best_score = current_score
        
        while time.time() < deadline - 2:
            iteration += 1
            if max_iterations and iteration >= max_iterations:
                break
            if no_improve > max_no_improve:
                break
            
            # Pick random point and move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            
            # Choose move size based on stagnation
            if no_improve > 2000:
                moves = extra_large_moves
            elif no_improve > 1000:
                moves = large_moves
            elif no_improve > 500:
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
            
            # Incremental evaluation
            new_score, _, new_areas = incremental_evaluate(new_points, idx, current_areas)
            
            if new_score > current_score:
                current_score = new_score
                current_points = new_points
                current_areas = new_areas
                no_improve = 0
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_points = list(current_points)
            else:
                if new_score == current_score and random.random() < 0.1:
                    if no_improve > 300:
                        current_points = new_points
                        current_areas = new_areas
                        no_improve = 0
                else:
                    no_improve += 1
        
        return local_best_points, local_best_score
    
    # Run local search from the best initial point
    best_overall_points, best_overall_score = run_local_search(best_points)
    
    # Try additional restarts from slightly perturbed versions of best solution
    restart_attempts = 0
    max_restarts = 20
    
    while time.time() < deadline - 5 and restart_attempts < max_restarts:
        restart_attempts += 1
        
        # Create a perturbed version of the best solution
        perturbed = list(best_overall_points)
        for _ in range(random.randint(3, 8)):
            idx = random.randint(0, 29)
            dx, dy = random.choice(large_moves)
            nx, ny = perturbed[idx][0] + dx, perturbed[idx][1] + dy
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                new_pt = (nx, ny)
                if new_pt not in perturbed[:idx] + perturbed[idx+1:]:
                    perturbed[idx] = new_pt
        
        if len(set(perturbed)) < 30:
            continue
        
        # Run local search from perturbed point
        pts, score = run_local_search(perturbed, max_iterations=2000)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = pts
    
    return best_overall_points
