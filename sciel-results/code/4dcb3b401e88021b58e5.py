import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _epigraph_objective(x):
        """
        Epigraph formulation: minimize t subject to all triangle areas >= t.
        x = [x_0, y_0, ..., x_12, y_12, t]  (27 params)
        We maximize min area / hull area by minimizing -t where t is the min area,
        but with the constraint that the hull area is normalized to 1.
        
        Actually, let's use a different approach:
        We work in a space where the hull area is fixed at 1 (by scaling),
        and we maximize the minimum triangle area.
        
        For the optimizer, we minimize -min_area (or use soft-min).
        """
        pts = x[:2*N].reshape(-1, 2)
        t = x[2*N]
        
        # Normalize to unit hull area
        pts_norm = _normalize_points(pts)
        
        x_n = pts_norm[:, 0]
        y_n = pts_norm[:, 1]
        v1x = x_n[tri_j] - x_n[tri_i]
        v1y = y_n[tri_j] - y_n[tri_i]
        v2x = x_n[tri_k] - x_n[tri_i]
        v2y = y_n[tri_k] - y_n[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Penalty if any area < t
        violations = np.maximum(0, t - areas)
        penalty = np.sum(violations**2)
        
        # We want to maximize t, so minimize -t + penalty
        return -t + penalty * 100.0

    def _soft_min_objective(x, beta=100.0):
        """
        Soft-min objective: minimize -soft_min(areas).
        x = [x_0, y_0, ..., x_12, y_12]  (26 params)
        """
        pts = x.reshape(-1, 2)
        pts_norm = _normalize_points(pts)
        
        x_n = pts_norm[:, 0]
        y_n = pts_norm[:, 1]
        v1x = x_n[tri_j] - x_n[tri_i]
        v1y = y_n[tri_j] - y_n[tri_i]
        v2x = x_n[tri_k] - x_n[tri_i]
        v2y = y_n[tri_k] - y_n[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min: -log(sum(exp(-beta * areas))) / beta
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _coordinate_descent(points, max_iter=200, tol=1e-10):
        """
        Coordinate descent to maximize min triangle area / hull area.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        
        # Adaptive step sizes
        base_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
                
            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = best_score
                
                for step in base_steps:
                    directions = [
                        (step, 0), (-step, 0), (0, step), (0, -step),
                        (step*0.707, step*0.707), (-step*0.707, step*0.707),
                        (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                    ]
                    for dx, dy in directions:
                        new_pt = pts.copy()
                        new_pt[i] = pts[i] + np.array([dx, dy])
                        score = compute_score(new_pt)
                        if score > best_local_score + tol:
                            best_local_score = score
                            best_local = new_pt[i].copy()
                            improved = True
                
                if improved:
                    pts[i] = best_local
                    best_score = best_local_score
            
            if not improved:
                break
        
        return pts, best_score

    def _simulated_annealing(initial_points, max_iter=5000, t_start=0.1, t_end=0.001):
        """
        Simulated annealing to escape local minima.
        """
        pts = initial_points.copy()
        current_score = compute_score(pts)
        best_pts = pts.copy()
        best_score = current_score
        
        # Temperature schedule
        temperatures = np.geomspace(t_start, t_end, max_iter)
        
        rng = np.random.RandomState(12345)
        
        for it in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
                
            T = temperatures[it]
            
            # Random perturbation
            new_pts = pts.copy()
            # Perturb 1-3 random points
            n_perturb = rng.randint(1, 4)
            indices = rng.choice(N, size=n_perturb, replace=False)
            for idx in indices:
                perturbation = rng.randn(2) * T
                new_pts[idx] += perturbation
            
            new_score = compute_score(new_pts)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / T):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
        
        return best_pts, best_score

    # Generate initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 on boundary + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. Various split configurations
    for k_outer in range(5, 13):
        k_inner = N - k_outer
        outer = _regular_polygon(k_outer, 1.0)
        for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if k_inner == 1:
                inner = np.array([[0.0, 0.0]])
            else:
                inner = _regular_polygon(k_inner, r_inner)
            config = np.vstack([outer, inner])
            initializations.append(config)
    
    # 4. Concentric polygons
    for r1 in [0.3, 0.4, 0.5]:
        for r2 in [0.6, 0.7, 0.8]:
            center = np.array([[0.0, 0.0]])
            inner_hex = _regular_polygon(6, r1)
            outer_hex = _regular_polygon(6, r2)
            config = np.vstack([center, inner_hex, outer_hex])
            initializations.append(config)
    
    # 5. 13 = 1 + 3 + 9
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        triangle = _regular_polygon(3, r1)
        nonagon = _regular_polygon(9, 1.0)
        config = np.vstack([center, triangle, nonagon])
        initializations.append(config)
    
    # 6. 13 = 1 + 4 + 8
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        square = _regular_polygon(4, r1)
        octagon = _regular_polygon(8, 1.0)
        config = np.vstack([center, square, octagon])
        initializations.append(config)
    
    # 7. 13 = 1 + 5 + 7
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        pentagon = _regular_polygon(5, r1)
        heptagon = _regular_polygon(7, 1.0)
        config = np.vstack([center, pentagon, heptagon])
        initializations.append(config)
    
    # 8. 13 = 1 + 2 + 10
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        dipole = np.array([[-r1, 0.0], [r1, 0.0]])
        decagon = _regular_polygon(10, 1.0)
        config = np.vstack([center, dipole, decagon])
        initializations.append(config)
    
    # Add random perturbations
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 5):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.05
        initializations.append(perturbed)
    
    # Random initializations
    for _ in range(20):
        rng = np.random.RandomState(np.random.randint(10000))
        pts = rng.randn(N, 2)
        initializations.append(pts)
    
    best_score = -float('inf')
    best_points = None
    seen_scores = set()
    
    # Optimize each initialization
    for idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 25:
            break
            
        current = init.copy()
        current_score = compute_score(current)
        
        # Quick check: skip if this is a known bad config
        score_key = round(current_score, 6)
        if score_key in seen_scores:
            continue
        seen_scores.add(score_key)
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_min_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Simulated annealing to escape local minima
        if time.time() < start_time + time_limit - 20:
            try:
                sa_pts, sa_score = _simulated_annealing(current, max_iter=3000, t_start=0.05, t_end=0.001)
                if sa_score > current_score:
                    current = sa_pts
                    current_score = sa_score
            except Exception:
                pass
        
        # Step 3: Coordinate descent refinement
        if time.time() > start_time + time_limit - 10:
            break
        try:
            current, score_cd = _coordinate_descent(current, max_iter=200)
            if score_cd > best_score:
                best_score = score_cd
                best_points = current.copy()
        except Exception:
            pass
        
        # Step 4: Nelder-Mead with exact objective for final polish
        if time.time() > start_time + time_limit - 8:
            break
        try:
            res = opt.minimize(
                lambda x: -compute_score(x.reshape(-1, 2)),
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 5: Additional coordinate descent pass after Nelder-Mead
        if time.time() > start_time + time_limit - 5:
            break
        try:
            candidate, score_cd2 = _coordinate_descent(candidate, max_iter=100)
            if score_cd2 > best_score:
                best_score = score_cd2
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 6: High-precision final polish with very small steps
        if time.time() < start_time + time_limit - 2:
            try:
                fine_steps = [0.0001, 0.00005, 0.00002, 0.00001]
                fine_pts = best_points.copy() if best_points is not None else current.copy()
                fine_score = compute_score(fine_pts)
                for step in fine_steps:
                    for i in range(N):
                        directions = [
                            (step, 0), (-step, 0), (0, step), (0, -step),
                            (step*0.707, step*0.707), (-step*0.707, step*0.707),
                            (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                        ]
                        for dx, dy in directions:
                            new_pt = fine_pts.copy()
                            new_pt[i] = fine_pts[i] + np.array([dx, dy])
                            score = compute_score(new_pt)
                            if score > fine_score + 1e-12:
                                fine_score = score
                                fine_pts[i] = new_pt[i]
                if fine_score > best_score:
                    best_score = fine_score
                    best_points = fine_pts.copy()
            except Exception:
                pass
    
    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]
    
    return best_points
