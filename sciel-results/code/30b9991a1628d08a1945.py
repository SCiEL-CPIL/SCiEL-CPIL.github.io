"""Optimized: Vectorized evaluation + restricted lattice search + time-bounded local search."""
import time
import numpy as np
import random
from itertools import combinations
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin: return by 50s, limit is 90s.

    # Precompute indices for all triples (i, j, k) with i < j < k
    # This allows vectorized area computation for a given set of points.
    indices = list(combinations(range(30), 3))
    idx_arr = np.array(indices, dtype=np.int64)
    I, J, K = idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]
    
    # Precompute which triples contain each point index
    # triples_with_point[i] = list of triple indices that include point i
    triples_with_point = [[] for _ in range(30)]
    for t_idx, (i, j, k) in enumerate(indices):
        triples_with_point[i].append(t_idx)
        triples_with_point[j].append(t_idx)
        triples_with_point[k].append(t_idx)
    # Convert to arrays for fast access
    triples_with_point_arr = [np.array(tp, dtype=np.int64) for tp in triples_with_point]

    def fast_evaluate(points):
        """
        Compute the max multiplicity of any positive doubled area.
        points: list of 30 (x, y) tuples.
        """
        if len(points) != 30:
            return 0
        
        pts = np.array(points, dtype=np.int64)
        # Check distinctness quickly
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
        _, counts = np.unique(positive_areas, return_counts=True)
        return int(counts.max())

    def incremental_evaluate(points, moved_idx, old_pt, new_pt, prev_areas, prev_counts):
        """
        Incrementally update area counts after moving one point.
        Returns (new_max_count, new_areas, new_counts)
        """
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        
        # Get triples that involve the moved point
        t_indices = triples_with_point_arr[moved_idx]
        
        if len(t_indices) == 0:
            return prev_counts.max() if len(prev_counts) > 0 else 0, prev_areas, prev_counts
        
        # Recompute areas for affected triples
        ti = t_indices
        xi, yi = x[ti], y[ti]
        # Need to get the other two points from the triple
        # For each triple (a,b,c) containing moved_idx, get the other two
        triple_arr = idx_arr[ti]
        a_idx = triple_arr[:, 0]
        b_idx = triple_arr[:, 1]
        c_idx = triple_arr[:, 2]
        
        # For each triple, find which position has moved_idx
        # and compute area with the new point
        new_areas = np.zeros(len(ti), dtype=np.int64)
        
        for pos in range(3):
            mask = (triple_arr[:, pos] == moved_idx)
            if np.any(mask):
                m_idx = np.where(mask)[0]
                if pos == 0:
                    other1 = b_idx[m_idx]
                    other2 = c_idx[m_idx]
                elif pos == 1:
                    other1 = a_idx[m_idx]
                    other2 = c_idx[m_idx]
                else:
                    other1 = a_idx[m_idx]
                    other2 = b_idx[m_idx]
                
                # Compute area with new point
                nx, ny = new_pt[0], new_pt[1]
                x1, y1 = x[other1], y[other1]
                x2, y2 = x[other2], y[other2]
                
                # Area = |nx*(y1-y2) + x1*(y2-ny) + x2*(ny-y1)|
                new_areas[m_idx] = np.abs(nx * (y1 - y2) + x1 * (y2 - ny) + x2 * (ny - y1))
        
        # Update areas array
        new_areas_full = prev_areas.copy()
        new_areas_full[ti] = new_areas
        
        # Recompute counts from scratch (still fast for 4060 elements)
        positive_areas = new_areas_full[new_areas_full > 0]
        if len(positive_areas) == 0:
            return 0, new_areas_full, np.array([0])
        
        _, counts = np.unique(positive_areas, return_counts=True)
        return int(counts.max()), new_areas_full, counts

    # --- Phase 1: Generate Good Initial Candidates ---
    # We test a small set of structured lattices that are known to produce high multiplicities.
    
    best_points = None
    best_score = 0
    
    def try_candidate(pts):
        nonlocal best_points, best_score
        if len(pts) != 30 or len(set(pts)) < 30:
            return
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            return
        score = fast_evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
    
    # Lattice-based candidates
    bases = [
        ((1, 0), (0, 1)),
        ((1, 0), (0, 2)),
        ((1, 0), (2, 0)),
        ((1, 0), (1, 1)),
        ((1, 0), (1, 2)),
        ((1, 0), (2, 1)),
        ((1, 1), (1, -1)),
        ((1, 0), (0, 3)),
        ((2, 0), (0, 1)),
        ((1, 1), (0, 1)),
        ((1, -1), (0, 1)),
    ]
    
    dims = [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2)]
    
    for u, v in bases:
        for M, N in dims:
            if time.time() > deadline - 15:
                break
                
            pts = []
            for i in range(M):
                for j in range(N):
                    px = i * u[0] + j * v[0]
                    py = i * u[1] + j * v[1]
                    pts.append((px, py))
            
            # Center the points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            
            centered_pts = [(p[0] - cx, p[1] - cy) for p in pts]
            try_candidate(centered_pts)
    
    # Two-parallel-lines candidates:
    # Place 15 points on line y=0 and 15 points on line y=d
    # Triangles with 2 points on one line and 1 on the other have area = d * |x1 - x2| / 2
    # So doubled area = d * |x1 - x2|
    # To maximize multiplicity, we want many pairs with the same |x1 - x2|
    
    for d in range(1, 20):
        if time.time() > deadline - 15:
            break
        # Try different spacings for the two lines
        for spacing1 in [1, 2, 3]:
            for spacing2 in [1, 2, 3]:
                # Line 1: 15 points at x = 0, spacing1, 2*spacing1, ...
                # Line 2: 15 points at x = offset, offset+spacing2, ...
                # Choose offset to maximize equal |differences|
                
                # Try offset = 0, spacing1//2, etc.
                for offset in [0, 1, 2, 3]:
                    pts = []
                    for i in range(15):
                        pts.append((i * spacing1, 0))
                    for i in range(15):
                        pts.append((offset + i * spacing2, d))
                    
                    # Center
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    cx = (min(xs) + max(xs)) // 2
                    cy = (min(ys) + max(ys)) // 2
                    centered = [(p[0] - cx, p[1] - cy) for p in pts]
                    try_candidate(centered)
    
    # Fallback if nothing found (should not happen)
    if best_points is None:
        pts = []
        for i in range(5):
            for j in range(6):
                pts.append((i, j))
        cx, cy = 2, 2
        best_points = [(p[0]-cx, p[1]-cy) for p in pts]
        best_score = fast_evaluate(best_points)

    # --- Phase 2: Local Search ---
    # We perform a hill-climbing search with multiple restarts.
    
    current_points = list(best_points)
    current_score = best_score
    random.seed(42)
    
    # Larger neighborhood: move one point by (dx, dy) where dx, dy in [-4, 4]
    neighborhood_small = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    neighborhood_large = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)]
    
    def local_search(points, score, time_limit):
        """Run local search until time_limit is reached. Returns (best_points, best_score)."""
        cur_pts = list(points)
        cur_score = score
        no_improve = 0
        max_no_improve = 500
        
        # Use set for fast collision checking
        cur_set = set(cur_pts)
        
        while time.time() < time_limit:
            if no_improve > max_no_improve:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 29)
            old_x, old_y = cur_pts[idx]
            
            # Choose neighborhood size based on stagnation
            if no_improve > 200:
                dx, dy = random.choice(neighborhood_large)
            else:
                dx, dy = random.choice(neighborhood_small)
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Check bounds
            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                no_improve += 1
                continue
            
            new_point = (new_x, new_y)
            
            # Check collision: new_point must not be in cur_set (except the old point)
            if new_point in cur_set and new_point != (old_x, old_y):
                no_improve += 1
                continue
            
            # Update set
            cur_set.remove((old_x, old_y))
            cur_set.add(new_point)
            
            new_pts = list(cur_pts)
            new_pts[idx] = new_point
            
            score = fast_evaluate(new_pts)
            
            if score > cur_score:
                cur_score = score
                cur_pts = new_pts
                no_improve = 0
            else:
                # Revert
                cur_set.remove(new_point)
                cur_set.add((old_x, old_y))
                no_improve += 1
        
        return cur_pts, cur_score
    
    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # First pass: intensive local search
        current_points, current_score = local_search(current_points, current_score, time.time() + min(20, remaining_time - 2))
        
        # If time remains, try a few random restarts
        restarts = 0
        while time.time() < deadline - 3 and restarts < 3:
            # Create a perturbed version of the best solution
            perturbed = list(current_points)
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, 29)
                old_x, old_y = perturbed[idx]
                dx, dy = random.choice(neighborhood_large)
                new_x, new_y = old_x + dx, old_y + dy
                if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                    new_pt = (new_x, new_y)
                    if new_pt not in perturbed or new_pt == (old_x, old_y):
                        perturbed[idx] = new_pt
            
            if len(set(perturbed)) == 30:
                p_score = fast_evaluate(perturbed)
                if p_score >= current_score - 2:  # Only restart if close to best
                    p_pts, p_score = local_search(perturbed, p_score, min(time.time() + 5, deadline - 2))
                    if p_score > current_score:
                        current_points = p_pts
                        current_score = p_score
            restarts += 1

    return current_points
