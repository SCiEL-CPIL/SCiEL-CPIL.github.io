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

    def _hexagonal_lattice_init():
        """Generate points on a hexagonal lattice pattern."""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2 (partial to reach 13)
        for i in range(6):
            angle = i * np.pi / 3 + np.pi / 6
            pts.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        return np.array(pts[:13])

    def _local_optimize(pts, beta_start=0.1, beta_end=0.01, n_steps=5):
        """
        Perform local optimization using L-BFGS-B with annealing beta.
        """
        current_pts = pts.copy()
        for step in range(n_steps):
            beta = beta_start * (beta_end / beta_start) ** (step / (n_steps - 1))
            try:
                res = scipy.optimize.minimize(
                    lambda x: _objective_and_grad(x, beta)[0],
                    lambda x: _objective_and_grad(x, beta)[1],
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                current_pts = res.x.reshape(n, 2)
            except Exception:
                pass
        return current_pts

    def _simulated_annealing_step(pts, temperature):
        """
        Perform one step of simulated annealing: perturb points and accept if better or with probability.
        """
        pts_new = pts + np.random.normal(0, temperature, (n, 2))
        score_old = compute_ratio(pts)
        score_new = compute_ratio(pts_new)
        
        if score_new >= score_old:
            return pts_new
        else:
            prob = np.exp((score_new - score_old) / max(temperature, 1e-9))
            if np.random.random() < prob:
                return pts_new
        return pts

    # Generate diverse initial guesses
    guesses = []
    guesses.append(_regular_polygon(n))
    guesses.append(_hexagonal_lattice_init())
    
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        guesses.append(rng.normal(0, 1, (n, 2)))
    
    best_score = -np.inf
    best_points = None

    # Phase 1: Local optimization on each guess
    for g in guesses:
        if time.time() - start_time > time_limit - 120:
            break
        try:
            optimized = _local_optimize(g)
            score = compute_ratio(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            continue

    if best_points is None:
        best_points = _regular_polygon(n)
        best_score = compute_ratio(best_points)

    # Phase 2: Simulated Annealing / Stochastic Search
    # We use a cooling schedule to explore the landscape
    current_pts = best_points.copy()
    current_score = best_score
    temperature = 0.5
    cooling_factor = 0.95
    n_sa_steps = 1000
    
    for step in range(n_sa_steps):
        if time.time() - start_time > time_limit - 60:
            break
        
        # Perturb
        new_pts = current_pts + np.random.normal(0, temperature, (n, 2))
        new_score = compute_ratio(new_pts)
        
        # Acceptance criterion
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
        else:
            prob = np.exp((new_score - current_score) / max(temperature, 1e-9))
            if np.random.random() < prob:
                current_pts = new_pts
                current_score = new_score
        
        temperature *= cooling_factor

    # Phase 3: Final local polish on the best point found
    if time.time() - start_time < time_limit - 20:
        try:
            polished = _local_optimize(best_points, beta_start=0.05, beta_end=0.005, n_steps=5)
            score = compute_ratio(polished)
            if score > best_score:
                best_score = score
                best_points = polished
        except Exception:
            pass

    # Phase 4: Random restarts with local optimization
    if time.time() - start_time < time_limit - 10:
        rng2 = np.random.default_rng(seed=123)
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            perturbed = best_points + rng2.normal(0, 0.1, (n, 2))
            try:
                optimized = _local_optimize(perturbed, beta_start=0.1, beta_end=0.01, n_steps=3)
                score = compute_ratio(optimized)
                if score > best_score:
                    best_score = score
                    best_points = optimized
            except Exception:
                continue

    return best_points
