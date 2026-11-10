import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n=13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points, normalized by the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13, 2)
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Leave 40s margin from 630s

    # Precompute triangle indices
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f, j_f, k_f = i_idx[mask], j_idx[mask], k_idx[mask]
    n_tris = len(i_f)

    def compute_ratio(points):
        points = np.asarray(points, dtype=float)
        if points.shape != (n, 2):
            return 0.0
        if np.any(~np.isfinite(points)):
            return 0.0
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 0.0
        except Exception:
            return 0.0

        p1 = points[i_f]
        p2 = points[j_f]
        p3 = points[k_f]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / A_hull

    def _triangle_areas(pts):
        """Compute all triangle areas vectorized."""
        p1 = pts[i_f]
        p2 = pts[j_f]
        p3 = pts[k_f]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def _hull_area(pts):
        try:
            hull = scipy.spatial.ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e10

    def _soft_min_and_grad(areas, beta):
        """
        Compute soft-min of areas and its gradient w.r.t. areas.
        soft_min = -beta * log(sum(exp(-areas/beta)))
        d(soft_min)/d(areas_i) = -exp(-areas_i/beta) / sum(exp(-areas_j/beta))
        """
        # Ensure areas are positive to avoid log(0) or exp issues if area is 0
        areas = np.maximum(areas, 1e-12)
        neg_scaled = -areas / beta
        max_neg = np.max(neg_scaled)
        exp_neg = np.exp(neg_scaled - max_neg)
        sum_exp = np.sum(exp_neg)
        log_sum_exp = max_neg + np.log(sum_exp)
        soft_min = -beta * log_sum_exp
        weights = exp_neg / sum_exp  # d(soft_min)/d(areas_i) = -weights
        return soft_min, -weights

    def _objective_and_grad(x, beta):
        """
        Returns (-soft_min/A_hull, gradient) for L-BFGS-B.
        """
        pts = x.reshape(n, 2)
        
        A_hull = _hull_area(pts)
        if A_hull <= 1e-12:
            return 1e10, np.zeros_like(x)
        
        areas = _triangle_areas(pts)
        soft_min, d_soft_min_d_areas = _soft_min_and_grad(areas, beta)
        
        value = -soft_min / A_hull
        
        # Gradient calculation
        p1 = pts[i_f]
        p2 = pts[j_f]
        p3 = pts[k_f]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        sign_cross = np.sign(cross)
        # Avoid division by zero if cross is 0
        sign_cross[cross == 0] = 1.0
        
        grad = np.zeros_like(x)
        for t in range(n_tris):
            w = d_soft_min_d_areas[t] * sign_cross[t] * 0.5 / A_hull
            ii = i_f[t]
            jj = j_f[t]
            kk = k_f[t]
            
            x1, y1 = pts[ii]
            x2, y2 = pts[jj]
            x3, y3 = pts[kk]
            
            # d(cross)/d(x1) = -(y3-y1), d(cross)/d(y1) = (x3-x1)
            # d(cross)/d(x2) = (y3-y1), d(cross)/d(y2) = -(x3-x1)
            # d(cross)/d(x3) = -(y2-y1), d(cross)/d(y3) = (x2-x1)
            
            grad[ii, 0] += -w * (y3 - y1)
            grad[ii, 1] += w * (x3 - x1)
            
            grad[jj, 0] += w * (y3 - y1)
            grad[jj, 1] += -w * (x3 - x1)
            
            grad[kk, 0] += -w * (y2 - y1)
            grad[kk, 1] += w * (x2 - x1)
        
        return value, grad

    def _regular_polygon(n_sides, radius=1.0, center=(0.0, 0.0), rotation=0.0):
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + rotation
        pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        pts += np.array(center)
        return pts

    def _optimize_local(x0, time_budget):
        """
        Perform local optimization starting from x0.
        Uses beta annealing with L-BFGS-B followed by Nelder-Mead polish.
        """
        current_x = x0.copy()
        current_score = compute_ratio(current_x.reshape(n, 2))
        
        # Beta schedule: Start smooth, end sharp
        betas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        for beta in betas:
            if time.time() - start_time > time_limit - 10:
                break
            
            # L-BFGS-B step
            try:
                res = scipy.optimize.minimize(
                    lambda x: _objective_and_grad(x, beta)[0],
                    lambda x: _objective_and_grad(x, beta)[1],
                    current_x, method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50}
                )
                if res.fun < 1e9: # Check for failure
                    new_pts = res.x.reshape(n, 2)
                    new_score = compute_ratio(new_pts)
                    if new_score > current_score:
                        current_x = res.x
                        current_score = new_score
            except Exception:
                pass
            
            # Nelder-Mead polish with exact objective (or very sharp soft min)
            # Using exact objective for NM is often more robust for final polish
            if time.time() - start_time > time_limit - 5:
                break
                
            try:
                def exact_neg_obj(x):
                    pts = x.reshape(n, 2)
                    s = compute_ratio(pts)
                    return -s if s > 0 else 1e10

                res_nm = scipy.optimize.minimize(
                    exact_neg_obj,
                    current_x, method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                new_pts = res_nm.x.reshape(n, 2)
                new_score = compute_ratio(new_pts)
                if new_score > current_score:
                    current_x = res_nm.x
                    current_score = new_score
            except Exception:
                pass

        return current_x, current_score

    # Generate diverse initial guesses
    guesses = []
    
    # 1. Regular 13-gon
    guesses.append(_regular_polygon(n))
    
    # 2. 12-gon + Center
    poly12 = _regular_polygon(12)
    guesses.append(np.vstack([poly12, [[0.0, 0.0]]]))
    
    # 3. 11-gon + Center + 1 point on edge
    poly11 = _regular_polygon(11)
    guesses.append(np.vstack([poly11, [[0.0, 0.0]], [0.95, 0.0]]))
    
    # 4. Two concentric hexagons (rotated) + Center
    h1 = _regular_polygon(6, 1.0)
    h2 = _regular_polygon(6, 0.5, rotation=np.pi/6)
    guesses.append(np.vstack([h1, h2, [[0.0, 0.0]]]))
    
    # 5. 7-gon + 6 points on inner circle
    poly7 = _regular_polygon(7, 1.0)
    inner6 = _regular_polygon(6, 0.45, rotation=np.pi/7)
    guesses.append(np.vstack([poly7, inner6]))
    
    # 6. 5-gon + 5-gon (rotated) + 3 points
    pent1 = _regular_polygon(5, 1.0)
    pent2 = _regular_polygon(5, 0.55, rotation=np.pi/5)
    extra = np.array([[0.3, 0.3], [-0.3, 0.3], [0.0, -0.4]])
    guesses.append(np.vstack([pent1, pent2, extra]))
    
    # 7. Random initializations
    rng = np.random.default_rng(seed=42)
    for _ in range(8):
        guesses.append(rng.normal(0, 1, (n, 2)))
        
    # 8. Jittered Grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([0.3 + 0.4*i, 0.3 + 0.4*j])
    grid_pts = np.array(grid_pts[:n])
    guesses.append(grid_pts)

    best_score = -np.inf
    best_points = None

    # Phase 1: Multi-start local optimization
    for g in guesses:
        if time.time() - start_time > time_limit - 30:
            break
        
        x0 = g.flatten()
        try:
            # Normalize initial guess to have reasonable scale
            # We can center and scale to unit hull area or just use as is
            # Using as is is fine, but let's ensure no NaNs
            if np.any(~np.isfinite(x0)):
                continue
                
            res_x, res_score = _optimize_local(x0, time_budget=60)
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_x.reshape(n, 2)
        except Exception:
            continue

    if best_points is None:
        best_points = _regular_polygon(n)
        best_score = compute_ratio(best_points)

    # Phase 2: Intensive search around the best point found
    if time.time() - start_time < time_limit - 20:
        rng2 = np.random.default_rng(seed=123)
        for _ in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Perturb the best point
            perturbation_scale = 0.1
            perturbed = best_points + rng2.normal(0, perturbation_scale, (n, 2))
            x0 = perturbed.flatten()
            
            try:
                res_x, res_score = _optimize_local(x0, time_budget=30)
                
                if res_score > best_score:
                    best_score = res_score
                    best_points = res_x.reshape(n, 2)
            except Exception:
                continue

    # Final validation and cleanup
    if np.any(~np.isfinite(best_points)):
        return _regular_polygon(n)
        
    return best_points
