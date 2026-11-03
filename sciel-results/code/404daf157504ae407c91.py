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
    tri_i = []
    tri_j = []
    tri_k = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)
    n_tris = len(tri_i)

    def _compute_areas(pts):
        """Compute all triangle areas for given points."""
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        return 0.5 * cross

    def _hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

    def _score(pts):
        """Compute min triangle area / convex hull area."""
        try:
            pts_c = pts - pts.mean(axis=0)
            area_hull = _hull_area(pts_c)
            if area_hull < 1e-9:
                return 0.0
            areas = _compute_areas(pts_c)
            min_area = np.min(areas)
            return min_area / area_hull
        except Exception:
            return 0.0

    def _soft_min_area(pts, beta=100.0):
        """Soft minimum of triangle areas using log-sum-exp."""
        areas = _compute_areas(pts)
        if np.any(areas < 1e-12):
            return 0.0
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 0.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        return -log_sum_exp / beta

    def _neg_soft_score(pts_flat, beta=100.0):
        """Negative soft score for minimization."""
        pts = pts_flat.reshape(-1, 2)
        pts_c = pts - pts.mean(axis=0)
        area_hull = _hull_area(pts_c)
        if area_hull < 1e-9:
            return 1e6
        soft_min = _soft_min_area(pts_c, beta)
        return -soft_min / area_hull

    def _neg_score(pts_flat):
        """Negative score for minimization."""
        pts = pts_flat.reshape(-1, 2)
        s = _score(pts)
        if s < 1e-12:
            return 1e6
        return -s

    def _concentric_init(n_outer, n_inner, r_outer=1.0, r_inner=0.5, rot_outer=0.0, rot_inner=0.0):
        """Generate points on two concentric circles."""
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + rot_outer
        outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        
        if n_inner > 0:
            angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + rot_inner
            inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
            return np.vstack([outer, inner])
        else:
            return outer

    def _symmetric_two_ring(params):
        """
        Parameterize 13 points as two concentric rings.
        params = [r_outer, r_inner, rot_outer, rot_inner, n_outer, n_inner]
        Returns 13x2 array of points.
        """
        r_outer, r_inner, rot_outer, rot_inner, n_outer, n_inner = params
        if n_outer + n_inner != 13:
            return None
        return _concentric_init(int(n_outer), int(n_inner), r_outer, r_inner, rot_outer, rot_inner)

    def _symmetric_optimize(params, max_iter=200):
        """Optimize two-ring symmetric configuration using Nelder-Mead."""
        def objective(p):
            pts = _symmetric_two_ring(p)
            if pts is None:
                return 1e6
            pts_c = pts - pts.mean(axis=0)
            area_hull = _hull_area(pts_c)
            if area_hull < 1e-9:
                return 1e6
            soft_min = _soft_min_area(pts_c, beta=80.0)
            return -soft_min / area_hull
        
        try:
            res = opt.minimize(
                objective,
                params,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            return res.x, res.fun
        except Exception:
            return params, 1e6

    def _refine_full_space(pts, max_iter=300):
        """Refine points in full 26D space using Nelder-Mead."""
        try:
            res = opt.minimize(
                _neg_score,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            return res.x.reshape(-1, 2)
        except Exception:
            return pts

    def _refine_soft_min(pts, beta=150.0, max_iter=200):
        """Refine using soft-min objective in full space."""
        try:
            res = opt.minimize(
                lambda x: _neg_soft_score(x, beta=beta),
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            return res.x.reshape(-1, 2)
        except Exception:
            return pts

    def _targeted_worst_triangle_opt(pts, max_iter=100):
        """
        Targeted optimization: focus on improving the worst triangle.
        Identify the worst triangle and perturb its vertices.
        """
        current = pts.copy()
        best_score = _score(current)
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 10:
                break
            
            # Find worst triangle
            areas = _compute_areas(current)
            worst_idx = np.argmin(areas)
            i, j, k = tri_i[worst_idx], tri_j[worst_idx], tri_k[worst_idx]
            
            # Try to improve by moving one of the vertices
            improved = False
            for vertex in [i, j, k]:
                for direction in range(8):
                    angle = 2 * np.pi * direction / 8
                    for step in [0.01, 0.02, 0.05]:
                        candidate = current.copy()
                        candidate[vertex, 0] += step * np.cos(angle)
                        candidate[vertex, 1] += step * np.sin(angle)
                        score = _score(candidate)
                        if score > best_score + 1e-10:
                            current = candidate
                            best_score = score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try random perturbations
                rng = np.random.RandomState(hash(iteration) % 2**32)
                for _ in range(20):
                    candidate = current + rng.randn(*current.shape) * 0.01
                    score = _score(candidate)
                    if score > best_score + 1e-10:
                        current = candidate
                        best_score = score
                        break
                else:
                    break
        
        return current, best_score

    # Generate diverse initializations
    initializations = []
    
    # Two-ring configurations with various splits
    for n_outer in range(6, 13):
        n_inner = 13 - n_outer
        for r_inner in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for rot_inner in [0.0, np.pi/n_inner, 2*np.pi/n_inner]:
                params = [1.0, r_inner, 0.0, rot_inner, float(n_outer), float(n_inner)]
                initializations.append(params)
    
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    initializations.append([1.0, 0.0, 0.0, 0.0, 13.0, 0.0])
    
    # 12 outer + 1 center
    initializations.append([1.0, 0.0, 0.0, 0.0, 12.0, 1.0])
    
    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for params in initializations:
        if time.time() > start_time + time_limit - 20:
            break
        
        try:
            # First optimize in symmetric space
            opt_params, _ = _symmetric_optimize(params, max_iter=100)
            pts = _symmetric_two_ring(opt_params)
            if pts is None:
                continue
            
            # Refine in full space with soft-min
            pts = _refine_soft_min(pts, beta=100.0, max_iter=150)
            
            # Refine with hard score
            pts = _refine_full_space(pts, max_iter=200)
            
            # Targeted worst triangle optimization
            pts, score = _targeted_worst_triangle_opt(pts, max_iter=50)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
        except Exception:
            pass

    # If we have time, do a final polish on the best result
    if best_points is not None and time.time() < start_time + time_limit - 15:
        try:
            # Final soft-min refinement
            pts = _refine_soft_min(best_points, beta=200.0, max_iter=200)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
            
            # Final Nelder-Mead on hard score
            pts = _refine_full_space(best_points, max_iter=300)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
        except Exception:
            pass

    # Fallback: if no valid result, use regular 13-gon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
