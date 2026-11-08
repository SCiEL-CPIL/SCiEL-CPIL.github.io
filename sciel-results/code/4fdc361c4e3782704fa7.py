import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

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

    def _soft_objective(points_flat, beta=100.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
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

    def _exact_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _coordinate_descent_adaptive(points, max_iter=500, tol=1e-10):
        """
        Adaptive coordinate descent: step size adapts per coordinate.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        
        # Per-coordinate adaptive step sizes
        steps = np.ones(N) * 0.1
        min_step = 1e-6
        max_step = 1.0
        n_iter_no_improve = 0
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
            
            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = best_score
                step = steps[i]
                
                # Try multiple step sizes around current adaptive step
                for mult in [1.0, 0.5, 2.0, 0.25, 4.0]:
                    s = step * mult
                    if s < min_step or s > max_step:
                        continue
                    
                    # 8 directions
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        dx = s * np.cos(angle)
                        dy = s * np.sin(angle)
                        new_pt = pts.copy()
                        new_pt[i] = pts[i] + np.array([dx, dy])
                        score = compute_score(new_pt)
                        if score > best_local_score + tol:
                            best_local_score = score
                            best_local = new_pt[i].copy()
                            improved = True
                            # If improvement, keep this step size
                            break
                    else:
                        continue
                    break
                else:
                    # No improvement at this step size, shrink
                    steps[i] = max(min_step, step * 0.5)
                    continue
                
                if improved:
                    pts[i] = best_local
                    best_score = best_local_score
                    # Expand step slightly if we found improvement
                    steps[i] = min(max_step, step * 1.2)
            
            if not improved:
                n_iter_no_improve += 1
                if n_iter_no_improve >= 3:
                    break
            else:
                n_iter_no_improve = 0
        
        return pts, best_score

    # Generate diverse symmetric initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 11 outer + 2 inner (dipole)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(11, 1.0)
        inner = np.array([[-r, 0.0], [r, 0.0]])
        initializations.append(np.vstack([outer, inner]))
        
    # 4. 10 outer + 3 inner (triangle)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 5. 9 outer + 4 inner (square)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 6. 8 outer + 5 inner (pentagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 7. 7 outer + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 8. 6 outer + 1 center + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(6, 1.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, center, inner]))
        
    # 9. 5 outer + 8 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(5, 1.0)
        inner = _regular_polygon(8, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 10. 4 outer + 9 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(4, 1.0)
        inner = _regular_polygon(9, r)
        initializations.append(np.vstack([outer, inner]))
    
    # 11. Add a near-optimal hand-tuned configuration
    # Based on known good configurations: mix of outer ring and inner points
    outer_10 = _regular_polygon(10, 1.0)
    inner_3 = _regular_polygon(3, 0.45)
    inner_3[0, 0] += 0.05  # slight perturbation to break symmetry
    initializations.append(np.vstack([outer_10, inner_3]))
    
    # 12. Another variant
    outer_9 = _regular_polygon(9, 1.0)
    inner_4 = _regular_polygon(4, 0.5)
    inner_4[1, 1] += 0.03
    initializations.append(np.vstack([outer_9, inner_4]))
    
    # Add random perturbations to top symmetric configs
    rng = np.random.RandomState(42)
    for i in [0, 1, 3, 6, 9, 11, 12]:  # key initializations
        if i < len(initializations):
            perturbed = initializations[i].copy()
            perturbed += rng.randn(*perturbed.shape) * 0.08
            initializations.append(perturbed)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 8:
            break
            
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 4:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Adaptive coordinate descent refinement
        if time.time() > start_time + time_limit - 2:
            break
        try:
            current, score_cd = _coordinate_descent_adaptive(current, max_iter=400)
            if score_cd > best_score:
                best_score = score_cd
                best_points = current.copy()
        except Exception:
            pass
        
        # Step 3: Nelder-Mead with exact objective for final polish
        if time.time() > start_time + time_limit - 1:
            break
        try:
            res = opt.minimize(
                _exact_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 4: Final adaptive coordinate descent pass
        if time.time() > start_time + time_limit - 0.3:
            break
        try:
            candidate, score_cd2 = _coordinate_descent_adaptive(candidate, max_iter=200)
            if score_cd2 > best_score:
                best_score = score_cd2
                best_points = candidate.copy()
        except Exception:
            pass

    # Final high-precision polish on the best result
    if best_points is not None and time.time() < start_time + time_limit - 0.5:
        try:
            res = opt.minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # One more CD pass
        if time.time() < start_time + time_limit - 0.2:
            try:
                candidate, score_cd3 = _coordinate_descent_adaptive(best_points, max_iter=100)
                if score_cd3 > best_score:
                    best_score = score_cd3
                    best_points = candidate.copy()
            except Exception:
                pass

    # Fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
