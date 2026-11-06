"""Combined approach: structured lattice + continuous optimization + local search."""
import numpy as np
from collections import Counter
import random
import time
from itertools import combinations
from scipy.optimize import minimize

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    """
    
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s limit (evaluator kills at 90s, but task says 60s)
    
    def compute_max_multiplicity(points):
        """Compute the maximum multiplicity of any positive doubled area using vectorized ops."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        
        if not np.any(mask):
            return 0
            
        ii = i_idx[mask]
        jj = j_idx[mask]
        kk = k_idx[mask]
        
        xi = x[ii]
        yi = y[ii]
        xj = x[jj]
        yj = y[jj]
        xk = x[kk]
        yk = y[kk]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        areas = areas[areas > 0]
        
        if len(areas) == 0:
            return 0
            
        values, counts = np.unique(areas, return_counts=True)
        return int(np.max(counts))
    
    def count_area_multiplicities(points):
        """Return dict: area -> count for all positive areas."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        
        if not np.any(mask):
            return {}
            
        ii = i_idx[mask]
        jj = j_idx[mask]
        kk = k_idx[mask]
        
        xi = x[ii]
        yi = y[ii]
        xj = x[jj]
        yj = y[jj]
        xk = x[kk]
        yk = y[kk]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        areas = areas[areas > 0]
        
        if len(areas) == 0:
            return {}
            
        values, counts = np.unique(areas, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}
    
    def smooth_objective(params, beta=10.0):
        """
        Smoothed objective: maximize softmax-weighted max area count.
        params: 60-dim vector (30 x, 30 y) continuous coordinates.
        We want to maximize the concentration of triangle areas.
        """
        pts = params.reshape(30, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Compute all triangle areas
        i_idx, j_idx, k_idx = np.meshgrid(np.arange(30), np.arange(30), np.arange(30), indexing='ij')
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        
        ii = i_idx[mask]
        jj = j_idx[mask]
        kk = k_idx[mask]
        
        xi = x[ii]
        yi = y[ii]
        xj = x[jj]
        yj = y[jj]
        xk = x[kk]
        yk = y[kk]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Exclude near-zero areas
        areas = areas[areas > 0.5]
        
        if len(areas) == 0:
            return 0.0
        
        # Softmax over area values to find the "dominant" area
        # We want to maximize the count of the most frequent area
        # Use a soft approximation: for each area value, compute a smooth count
        # and take the max
        
        # Discretize areas into bins for smooth counting
        # Use a kernel density approach
        area_min = np.min(areas)
        area_max = np.max(areas)
        
        if area_max - area_min < 1e-6:
            return float(len(areas))
        
        # Create bins
        n_bins = 50
        bin_edges = np.linspace(area_min, area_max + 1e-6, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_width = bin_edges[1] - bin_edges[0]
        
        # For each area, compute its contribution to each bin (smooth)
        # count_bin[b] = sum_i exp(-beta * (area_i - bin_center_b)^2 / bin_width^2)
        diffs = areas[:, np.newaxis] - bin_centers[np.newaxis, :]
        weights = np.exp(-beta * (diffs ** 2) / (bin_width ** 2))
        bin_counts = np.sum(weights, axis=0)
        
        # Softmax over bin counts to get a smooth max
        max_count = np.max(bin_counts)
        if max_count < 1e-10:
            return 0.0
        
        # Return the smooth max (negative for minimization)
        return -max_count
    
    def round_and_deduplicate(pts_continuous):
        """Round continuous points to integers, ensure distinctness and bounds."""
        pts_int = np.round(pts_continuous).astype(int)
        
        # Clip to bounds
        pts_int[:, 0] = np.clip(pts_int[:, 0], -30, 30)
        pts_int[:, 1] = np.clip(pts_int[:, 1], -30, 30)
        
        # Ensure distinctness: if duplicates, perturb
        seen = set()
        unique_pts = []
        for p in pts_int:
            pt = (int(p[0]), int(p[1]))
            if pt not in seen:
                seen.add(pt)
                unique_pts.append(pt)
        
        # If we lost points, add random valid points
        all_valid = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
        random.shuffle(all_valid)
        for pt in all_valid:
            if len(unique_pts) >= 30:
                break
            if pt not in seen:
                seen.add(pt)
                unique_pts.append(pt)
        
        return unique_pts[:30]
    
    # Generate diverse candidate sets from structured patterns
    candidates = []
    
    # Pattern A: Two parallel rows
    for x_start in range(-15, 16):
        for y0 in range(-15, 16):
            for y1 in [y0+1, y0+2, y0+3]:
                pts = []
                for x in range(x_start, x_start + 15):
                    if -30 <= x <= 30:
                        pts.append((x, y0))
                for x in range(x_start, x_start + 15):
                    if -30 <= x <= 30:
                        pts.append((x, y1))
                if len(pts) == 30 and len(set(pts)) == 30:
                    candidates.append(pts[:])
    
    # Pattern B: Rectangular grids
    for w in range(4, 9):
        for h in range(4, 9):
            if w * h >= 30:
                for ox in range(-10, 10):
                    for oy in range(-10, 10):
                        pts = [(ox + x, oy + y) for x in range(w) for y in range(h)]
                        if len(pts) >= 30:
                            pts = pts[:30]
                            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                                candidates.append(pts)
    
    # Pattern C: Three parallel rows
    for x_start in range(-10, 10):
        for base_y in range(-10, 10):
            for gap in [1, 2]:
                pts = []
                for row in range(3):
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, base_y + row * gap))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Evaluate all candidates
    random.seed(42)
    best_score = 0
    best_points = None
    best_area = None
    
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        score = compute_max_multiplicity(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]
            counts = count_area_multiplicities(cand)
            if counts:
                best_area = max(counts, key=counts.get)
    
    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score = compute_max_multiplicity(best_points)
        counts = count_area_multiplicities(best_points)
        if counts:
            best_area = max(counts, key=counts.get)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_points
    
    # Continuous optimization phase
    # Try L-BFGS-B from multiple initializations
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Initializations: best structured solution + random
    initializations = [np.array(best_points, dtype=float)]
    
    # Add a few random initializations
    for _ in range(3):
        random_pts = random.sample(all_valid_points, 30)
        initializations.append(np.array(random_pts, dtype=float))
    
    bounds = [(-30.0, 30.0)] * 60
    
    for init in initializations:
        if time.time() - start_time > DEADLINE:
            break
        
        try:
            result = minimize(
                smooth_objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if result.success or result.fun < 0:
                rounded = round_and_deduplicate(result.x.reshape(30, 2))
                if len(rounded) == 30 and len(set(rounded)) == 30:
                    score = compute_max_multiplicity(rounded)
                    if score > best_score:
                        best_score = score
                        best_points = rounded
                        counts = count_area_multiplicities(rounded)
                        if counts:
                            best_area = max(counts, key=counts.get)
        except Exception:
            pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_points
    
    # Enhanced local search
    current_counts = count_area_multiplicities(best_points)
    current_best_area = max(current_counts, key=current_counts.get) if current_counts else None
    current_best_count = current_counts[current_best_area] if current_best_area else 0
    
    for iteration in range(15):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        current_set = set(best_points)
        
        for i in range(30):
            if time.time() - start_time > DEADLINE:
                break
            
            old_point = best_points[i]
            
            candidates_to_try = []
            
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ not in current_set and np_ != old_point:
                            candidates_to_try.append(np_)
            
            for _ in range(15):
                np_ = random.choice(all_valid_points)
                if np_ not in current_set and np_ != old_point:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:30]:
                if time.time() - start_time > DEADLINE:
                    break
                
                new_points = best_points[:]
                new_points[i] = new_point
                
                score = compute_max_multiplicity(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    current_set = set(best_points)
                    improved = True
                    current_counts = count_area_multiplicities(best_points)
                    if current_counts:
                        current_best_area = max(current_counts, key=current_counts.get)
                        current_best_count = current_counts[current_best_area]
                    break
        
        if not improved:
            for _ in range(5):
                if time.time() - start_time > DEADLINE:
                    break
                
                new_points = best_points[:]
                for _ in range(random.randint(2, 4)):
                    idx = random.randint(0, 29)
                    old = new_points[idx]
                    for _ in range(50):
                        dx = random.randint(-6, 6)
                        dy = random.randint(-6, 6)
                        np_ = (old[0] + dx, old[1] + dy)
                        if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in new_points:
                            new_points[idx] = np_
                            break
                
                if len(set(new_points)) == 30:
                    score = compute_max_multiplicity(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        current_set = set(best_points)
                        improved = True
                        current_counts = count_area_multiplicities(best_points)
                        if current_counts:
                            current_best_area = max(current_counts, key=current_counts.get)
                            current_best_count = current_counts[current_best_area]
    
    return best_points
