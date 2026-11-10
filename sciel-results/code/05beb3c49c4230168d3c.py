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

    def neg_objective(x):
        pts = x.reshape(n, 2)
        return -compute_ratio(pts)

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
        
        # Gradient: d(-soft_min/A_hull)/dx = -d(soft_min)/dx / A_hull
        # d(soft_min)/dx = sum_i (d_soft_min_d_areas[i] * d(area_i)/dx)
        # d(area_i)/dx for triangle (i,j,k):
        # area_i = 0.5 * |cross_i| where cross = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
        # For non-degenerate: d(0.5*|cross|)/dx = 0.5*sign(cross)*d(cross)/dx
        
        p1 = pts[i_f]
        p2 = pts[j_f]
        p3 = pts[k_f]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        sign_cross = np.sign(cross)
        
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

    def _physics_relax(pts, n_steps=500, dt=0.01):
        """
        Physics-based relaxation: Laplacian smoothing with repulsive forces
        to spread points apart while maintaining convex hull structure.
        """
        pts = pts.copy()
        for step in range(n_steps):
            forces = np.zeros_like(pts)
            for i in range(n):
                # Repulsive force from all other points
                for j in range(n):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-12:
                        dist_sq = 1e-12
                    dist = np.sqrt(dist_sq)
                    # Repulsive force ~ 1/dist^2
                    force_mag = 0.1 / (dist_sq + 0.01)
                    forces[i] += force_mag * diff / dist
                # Laplacian smoothing: pull toward centroid of neighbors
                centroid = np.mean(pts, axis=0)
                forces[i] += 0.05 * (centroid - pts[i])
            pts += dt * forces
        return pts

    # Generate diverse initial guesses
    guesses = []
    
    # Regular 13-gon
    guesses.append(_regular_polygon(n))
    
    # 12-gon + Center
    poly12 = _regular_polygon(12)
    guesses.append(np.vstack([poly12, [[0.0, 0.0]]]))
    
    # 11-gon + Center + 1 point on edge
    poly11 = _regular_polygon(11)
    guesses.append(np.vstack([poly11, [[0.0, 0.0]], [0.95, 0.0]]))
    
    # Two concentric hexagons (rotated) + Center
    h1 = _regular_polygon(6, 1.0)
    h2 = _regular_polygon(6, 0.5, rotation=np.pi/6)
    guesses.append(np.vstack([h1, h2, [[0.0, 0.0]]]))
    
    # 7-gon + 6 points on inner circle
    poly7 = _regular_polygon(7, 1.0)
    inner6 = _regular_polygon(6, 0.45, rotation=np.pi/7)
    guesses.append(np.vstack([poly7, inner6]))
    
    # 5-gon + 5-gon (rotated) + 3 points
    pent1 = _regular_polygon(5, 1.0)
    pent2 = _regular_polygon(5, 0.55, rotation=np.pi/5)
    extra = np.array([[0.3, 0.3], [-0.3, 0.3], [0.0, -0.4]])
    guesses.append(np.vstack([pent1, pent2, extra]))
    
    # Random initializations
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        guesses.append(rng.normal(0, 1, (n, 2)))
    
    best_score = -np.inf
    best_points = None

    # Phase 1: Quick Nelder-Mead on each guess
    for g in guesses:
        if time.time() - start_time > time_limit - 120:
            break
        try:
            x0 = g.flatten()
            res = scipy.optimize.minimize(
                neg_objective, x0, method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            continue

    if best_points is None:
        best_points = _regular_polygon(n)
        best_score = compute_ratio(best_points)

    # Phase 2: Physics-based relaxation to spread points
    if time.time() - start_time < time_limit - 60:
        relaxed = _physics_relax(best_points, n_steps=300, dt=0.005)
        score_relaxed = compute_ratio(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_points = relaxed

    # Phase 3: L-BFGS-B with annealing beta
    betas = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
    for beta in betas:
        if time.time() - start_time > time_limit - 40:
            break
        try:
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                lambda x: _objective_and_grad(x, beta)[1],
                x0, method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-14}
            )
            score = compute_ratio(res.x.reshape(n, 2))
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            continue

    # Phase 4: Final polish with Nelder-Mead
    if time.time() - start_time < time_limit - 20:
        try:
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                neg_objective, x0, method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-14}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 5: Random restarts with full pipeline
    if time.time() - start_time < time_limit - 10:
        rng2 = np.random.default_rng(seed=123)
        for _ in range(4):
            if time.time() - start_time > time_limit - 5:
                break
            perturbed = best_points + rng2.normal(0, 0.05, (n, 2))
            try:
                # Quick NM
                res_nm = scipy.optimize.minimize(
                    neg_objective, perturbed.flatten(), method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                score_nm = -res_nm.fun
                
                # L-BFGS-B with small beta
                res_lb = scipy.optimize.minimize(
                    lambda x: _objective_and_grad(x, 0.01)[0],
                    lambda x: _objective_and_grad(x, 0.01)[1],
                    res_nm.x, method='L-BFGS-B',
                    options={'maxiter': 150, 'gtol': 1e-9, 'ftol': 1e-14}
                )
                score_lb = compute_ratio(res_lb.x.reshape(n, 2))
                
                if score_lb > best_score:
                    best_score = score_lb
                    best_points = res_lb.x.reshape(n, 2)
                elif score_nm > best_score:
                    best_score = score_nm
                    best_points = res_nm.x.reshape(n, 2)
            except Exception:
                continue

    return best_points
