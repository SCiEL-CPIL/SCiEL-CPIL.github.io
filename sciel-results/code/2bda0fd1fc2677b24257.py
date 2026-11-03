import numpy as np
import time
from scipy.spatial import Delaunay

N = 300

def _compute_radius(pts):
    """Compute the packing radius (min of half pair-distances and boundary margins)."""
    if len(pts) < 2:
        return np.min(1.0 - np.linalg.norm(pts, axis=1))
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
    return min(min_pair, min_bound)

def _soft_min_optimize(points, time_budget, k_start=20.0, k_end=500.0, alpha_start=0.05, alpha_end=0.001, n_restarts=1):
    """Soft-min gradient ascent with log-sum-exp for numerical stability and adaptive sharpening."""
    start = time.time()
    deadline = start + time_budget
    
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    n_pairs = len(i_idx)
    
    best_pts = points.copy()
    best_score = _compute_radius(points)
    
    for restart in range(n_restarts):
        # For restarts > 0, add small perturbation
        if restart > 0:
            points = points + np.random.normal(0, 0.01, size=points.shape)
            rad = np.linalg.norm(points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                points[mask] /= rad[mask, np.newaxis]
        
        alpha = alpha_start
        k_soft = k_start
        n_steps = 0
        stall_count = 0
        last_best = best_score
        
        while time.time() < deadline:
            # Compute pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Boundary distances
            rad = np.linalg.norm(points, axis=1)
            bound_dists = 1.0 - rad
            
            # Constraint values
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            
            current_min = np.min(all_vals)
            
            if current_min > best_score + 1e-12:
                best_score = current_min
                best_pts = points.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Adaptive sharpening: increase k as we converge
            progress = min(1.0, n_steps / 5000.0)
            k_soft = k_start + (k_end - k_start) * progress
            
            # Log-sum-exp for stable soft-min weights
            shifted = all_vals - current_min
            # weights proportional to exp(-k * shifted)
            log_w = -k_soft * shifted
            log_w -= np.max(log_w)  # numerical stability
            w = np.exp(log_w)
            w_sum = np.sum(w)
            if w_sum < 1e-300:
                # All weights underflow; fall back to uniform on near-min constraints
                near_min = shifted < (all_vals - current_min).max() * 0.1 + 1e-12
                w = near_min.astype(float)
                w_sum = np.sum(w)
                if w_sum == 0:
                    w = np.ones_like(all_vals) / len(all_vals)
                    w_sum = 1.0
            weights = w / w_sum
            
            # Compute gradients
            grads = np.zeros_like(points)
            
            # Pair contributions
            pair_weights = weights[:n_pairs]
            # Only compute for pairs with significant weight to save time
            active = pair_weights > 1e-8
            if np.any(active):
                ai = i_idx[active]
                aj = j_idx[active]
                aw = pair_weights[active]
                ad = dists[ai, aj]
                ad_safe = np.where(ad > 1e-12, ad, 1.0)
                vecs = (diff[ai, aj] / ad_safe[:, np.newaxis]) * aw[:, np.newaxis]
                np.add.at(grads, ai, vecs)
                np.add.at(grads, aj, -vecs)
            
            # Boundary contributions
            bound_weights = weights[n_pairs:]
            active_b = bound_weights > 1e-8
            if np.any(active_b):
                ab = bound_weights[active_b]
                rad_safe = np.where(rad[active_b] > 1e-9, rad[active_b], 1.0)
                grads[active_b] -= (points[active_b] / rad_safe[:, np.newaxis]) * ab[:, np.newaxis]
            
            # Adaptive step size
            alpha = alpha_start * (alpha_end / alpha_start) ** progress
            alpha *= max(0.1, 1.0 - 0.01 * stall_count)
            
            # Update
            points += alpha * grads
            
            # Project inside circle
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            
            # Occasional perturbation
            if n_steps % 200 == 199 and np.random.rand() < 0.3:
                pert_scale = alpha * 3.0
                points += np.random.normal(0, pert_scale, size=points.shape)
                rad_new = np.linalg.norm(points, axis=1)
                mask = rad_new > 1.0
                if np.any(mask):
                    points[mask] /= rad_new[mask, np.newaxis]
            
            n_steps += 1
            
            # Early exit if stalled for too long
            if stall_count > 3000:
                break
        
        # Update best from this restart
        r = _compute_radius(points)
        if r > best_score:
            best_score = r
            best_pts = points.copy()
    
    return best_pts, best_score

def _delaunay_polish(points, time_budget):
    """High-precision polish using Delaunay triangulation to find active edges."""
    start = time.time()
    deadline = start + time_budget
    
    best_pts = points.copy()
    best_score = _compute_radius(points)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            edge_list = np.array(list(edges))
        except Exception:
            # Fallback: use nearest-neighbor edges
            edge_list = np.array([(i, (i+1) % N) for i in range(N)])
        
        # Compute current radius
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[edge_list[:, 0], edge_list[:, 1]] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        current_min = np.min(all_vals)
        
        if current_min > best_score + 1e-12:
            best_score = current_min
            best_pts = points.copy()
        
        # Use very sharp soft-min (nearly exact min)
        k_soft = 200.0
        shifted = all_vals - current_min
        log_w = -k_soft * shifted
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum < 1e-300:
            w = np.ones_like(all_vals) / len(all_vals)
            w_sum = 1.0
        weights = w / w_sum
        
        # Gradient
        grads = np.zeros_like(points)
        
        # Edge contributions
        ew = weights[:len(edge_list)]
        active = ew > 1e-8
        if np.any(active):
            ai = edge_list[active, 0]
            aj = edge_list[active, 1]
            aw = ew[active]
            ad = dists[ai, aj]
            ad_safe = np.where(ad > 1e-12, ad, 1.0)
            vecs = (diff[ai, aj] / ad_safe[:, np.newaxis]) * aw[:, np.newaxis]
            np.add.at(grads, ai, vecs)
            np.add.at(grads, aj, -vecs)
        
        # Boundary contributions
        bw = weights[len(edge_list):]
        active_b = bw > 1e-8
        if np.any(active_b):
            ab = bw[active_b]
            rad_safe = np.where(rad[active_b] > 1e-9, rad[active_b], 1.0)
            grads[active_b] -= (points[active_b] / rad_safe[:, np.newaxis]) * ab[:, np.newaxis]
        
        # Small step
        alpha = 0.005
        points += alpha * grads
        
        # Project
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
        
        # Check convergence
        if iteration > 50 and current_min < best_score - 1e-10:
            break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]
    
    # 2. Global optimization with soft-min
    elapsed = time.time() - start_time
    global_budget = min(80.0, time_limit - elapsed - 30.0)  # Leave 30s for polish + safety
    global_budget = max(global_budget, 20.0)
    
    best_points, best_score = _soft_min_optimize(points, global_budget, k_start=20.0, k_end=400.0, alpha_start=0.05, alpha_end=0.001, n_restarts=2)
    
    # 3. Delaunay-based polish
    elapsed = time.time() - start_time
    polish_budget = min(25.0, time_limit - elapsed - 5.0)
    polish_budget = max(polish_budget, 5.0)
    
    best_points, best_score = _delaunay_polish(best_points, polish_budget)
    
    # Final validation
    final_r = _compute_radius(best_points)
    if final_r > best_score:
        best_score = final_r
    
    return best_points
