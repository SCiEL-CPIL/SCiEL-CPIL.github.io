import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

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

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _concentric_config(n_outer, n_inner, r_inner, angle_offset_outer=0.0, angle_offset_inner=0.0):
        """Create a configuration with n_outer points on outer circle and n_inner on inner circle."""
        outer = _regular_polygon(n_outer, 1.0, center=(0.0, 0.0))
        outer[:, 0] += 0  # already centered
        # Rotate outer by angle_offset_outer
        if angle_offset_outer != 0:
            c, s = np.cos(angle_offset_outer), np.sin(angle_offset_outer)
            rot = np.array([[c, -s], [s, c]])
            outer = outer @ rot.T
        if n_inner > 0:
            inner = _regular_polygon(n_inner, r_inner, center=(0.0, 0.0))
            if angle_offset_inner != 0:
                c, s = np.cos(angle_offset_inner), np.sin(angle_offset_inner)
                rot = np.array([[c, -s], [s, c]])
                inner = inner @ rot.T
            return np.vstack([outer, inner])
        return outer

    def _coordinate_descent_refine(points, max_iter=500, tol=1e-10):
        """
        Coordinate-wise descent to maximize min triangle area / hull area.
        For each point, optimize its position while keeping others fixed.
        """
        pts = points.copy()
        current_score = compute_score(pts)
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > start_time + time_limit - 3:
                    break
                # Optimize point i's position
                def objective_i(x_flat):
                    trial = pts.copy()
                    trial[i] = x_flat
                    return -compute_score(trial)
                
                try:
                    res = opt.minimize(
                        objective_i,
                        pts[i],
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    new_pt = res.x
                    new_score = compute_score(np.vstack([pts[:i], new_pt.reshape(1, -1), pts[i+1:]]))
                    if new_score > current_score + 1e-12:
                        pts[i] = new_pt
                        current_score = new_score
                        improved = True
                except Exception:
                    pass
            
            if not improved:
                break
        
        return pts, current_score

    def _refine_with_soft_min(points, beta=100.0, max_iter=200):
        """
        Use soft-min objective with L-BFGS-B to refine all points simultaneously.
        """
        def soft_obj(x_flat):
            trial = x_flat.reshape(N, 2)
            score = compute_score(trial)
            if score < 1e-12:
                return 1e6
            # Soft min over triangle areas
            trial_c = trial - trial.mean(axis=0)
            try:
                hull = ConvexHull(trial_c)
                hull_area = hull.volume
            except Exception:
                return 1e6
            if hull_area < 1e-9:
                return 1e6
            x = trial_c[:, 0]
            y = trial_c[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            # Normalize by hull area
            areas_norm = areas / hull_area
            neg_beta = -beta * areas_norm
            max_val = np.max(neg_beta)
            if max_val < -1e6:
                return 1e6
            lse = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
            soft_min = -lse / beta
            return -soft_min
        
        try:
            res = opt.minimize(
                soft_obj,
                points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            candidate = res.x.reshape(N, 2)
            return candidate
        except Exception:
            return points

    # Generate initial configurations
    initializations = []
    
    # Concentric polygon configurations
    # 7 outer + 6 inner (hexagonal symmetry)
    for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for offset in [0.0, np.pi/7]:
            initializations.append(_concentric_config(7, 6, r, 0.0, offset))
    
    # 8 outer + 5 inner (pentagonal)
    for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for offset in [0.0, np.pi/8]:
            initializations.append(_concentric_config(8, 5, r, 0.0, offset))
    
    # 9 outer + 4 inner (square)
    for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for offset in [0.0, np.pi/9]:
            initializations.append(_concentric_config(9, 4, r, 0.0, offset))
    
    # 10 outer + 3 inner (triangle)
    for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for offset in [0.0, np.pi/10]:
            initializations.append(_concentric_config(10, 3, r, 0.0, offset))
    
    # 11 outer + 2 inner
    for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        initializations.append(_concentric_config(11, 2, r, 0.0, np.pi))
    
    # 12 outer + 1 center
    initializations.append(_concentric_config(12, 1, 0.0, 0.0, 0.0))
    
    # Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # Add perturbations to some configs
    rng = np.random.RandomState(123)
    for i in range(0, len(initializations), 5):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.05
        initializations.append(perturbed)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 15:
            break
            
        current = init.copy()
        
        # Step 1: Soft-min L-BFGS-B refinement
        if time.time() < start_time + time_limit - 10:
            current = _refine_with_soft_min(current, beta=50.0, max_iter=100)
        
        # Step 2: Increase beta for sharper min
        if time.time() < start_time + time_limit - 8:
            current = _refine_with_soft_min(current, beta=100.0, max_iter=100)
        
        # Step 3: Coordinate descent on individual points
        if time.time() < start_time + time_limit - 5:
            current, score_cd = _coordinate_descent_refine(current, max_iter=200)
        
        # Step 4: Final soft-min polish
        if time.time() < start_time + time_limit - 3:
            current = _refine_with_soft_min(current, beta=200.0, max_iter=50)
        
        # Evaluate final score
        score = compute_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()

    # Final check
    if best_points is None:
        best_points = initializations[0]

    return best_points
