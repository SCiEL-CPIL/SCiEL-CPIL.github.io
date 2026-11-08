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
        Objective: negative soft-min triangle area (normalized to unit hull area).
        Lower is better (we minimize this).
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
        
        # Soft min via log-sum-exp: soft_min = -1/beta * log(sum(exp(-beta * areas)))
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective for final polish: negative min triangle area (normalized).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score <= 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _coordinate_descent(points, max_iter=300, initial_step=0.02):
        """
        Coordinate descent refinement on the hard score.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        step = initial_step
        
        for it in range(max_iter):
            if time.time() > start_time + time_limit - 1:
                break
                
            improved = False
            for i in range(N):
                original = pts[i].copy()
                # Try 8 directions + smaller step
                for s in [step, step * 0.5]:
                    for dx in [-s, 0, s]:
                        for dy in [-s, 0, s]:
                            if dx == 0 and dy == 0:
                                continue
                            pts[i] = original + np.array([dx, dy])
                            new_score = compute_score(pts)
                            if new_score > best_score:
                                best_score = new_score
                                improved = True
                            else:
                                pts[i] = original
                    if improved:
                        break
                if not improved:
                    # Try diagonal directions
                    for s in [step, step * 0.5]:
                        for dx in [-s, s]:
                            for dy in [-s, s]:
                                pts[i] = original + np.array([dx, dy])
                                new_score = compute_score(pts)
                                if new_score > best_score:
                                    best_score = new_score
                                    improved = True
                                else:
                                    pts[i] = original
                        if improved:
                            break
            
            if not improved:
                step *= 0.7
                if step < 1e-7:
                    break
        
        return pts

    # Generate initializations - focus on promising symmetric configs
    initializations = []
    
    # 1. Regular 13-gon (strong baseline)
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 11 outer + 2 inner (dipole)
    for r in [0.4, 0.6]:
        outer = _regular_polygon(11, 1.0)
        inner = np.array([[-r, 0.0], [r, 0.0]])
        initializations.append(np.vstack([outer, inner]))
    
    # 4. 10 outer + 3 inner (triangle)
    for r in [0.4, 0.6]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r)
        initializations.append(np.vstack([outer, inner]))
    
    # 5. 9 outer + 4 inner (square)
    for r in [0.4, 0.6]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r)
        initializations.append(np.vstack([outer, inner]))
    
    # 6. 8 outer + 5 inner
    for r in [0.4, 0.6]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r)
        initializations.append(np.vstack([outer, inner]))
    
    # 7. 7 outer + 6 inner
    for r in [0.4, 0.6]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, inner]))
    
    # 8. 6 outer + 1 center + 6 inner
    for r in [0.4, 0.6]:
        outer = _regular_polygon(6, 1.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, center, inner]))
    
    # 9. Slightly perturbed regular 13-gon to break symmetry
    rng = np.random.RandomState(42)
    perturbed = _regular_polygon(13, 1.0).copy()
    perturbed += rng.randn(*perturbed.shape) * 0.05
    initializations.append(perturbed)
    
    # 10. Another perturbation
    perturbed2 = _regular_polygon(13, 1.0).copy()
    perturbed2 += rng.randn(*perturbed2.shape) * 0.1
    initializations.append(perturbed2)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for init_idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 8:
            break
            
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        # Start with moderate beta to explore, increase to refine
        betas = [30.0, 80.0, 150.0, 300.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 4:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Powell with hard objective
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 150, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 3: Coordinate descent for further refinement
        if time.time() > start_time + time_limit - 1:
            break
        try:
            candidate = _coordinate_descent(current, max_iter=150, initial_step=0.02)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Final: if best_points found, do one more polish pass on the best
    if best_points is not None and time.time() < start_time + time_limit - 3:
        try:
            res = opt.minimize(
                _hard_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-15, 'ftol': 1e-15}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Final coordinate descent
        if time.time() < start_time + time_limit - 1:
            try:
                candidate = _coordinate_descent(best_points, max_iter=100, initial_step=0.01)
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
            except Exception:
                pass

    # Fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
