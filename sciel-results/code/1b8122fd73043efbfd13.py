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

    def _tail_objective(points_flat, k=30):
        """
        Objective: minimize sum of k smallest triangle areas.
        This is a smooth, convex surrogate that lifts the entire lower tail,
        avoiding the "one bad triple dominates" problem of pure min-max.
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
        
        # Sum of k smallest areas
        if k >= len(areas):
            return np.sum(areas)
        # Use partition for speed, then sum the smallest k
        smallest = np.partition(areas, k)[:k]
        return np.sum(smallest)

    def _hard_objective(points_flat):
        """
        Objective: maximize min triangle area (for final polish).
        Returns negative min area / hull area.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _refine_nelder_mead(points, max_iter=200):
        """Nelder-Mead refinement on the hard objective."""
        pts = _normalize_points(points.copy())
        best_score = compute_score(pts)
        
        try:
            res = opt.minimize(
                _hard_objective,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            candidate = _normalize_points(res.x.reshape(-1, 2))
            score = compute_score(candidate)
            if score > best_score:
                return candidate, score
        except Exception:
            pass
        
        return pts, best_score

    def _refine_powell(points, max_iter=150):
        """Powell refinement on the hard objective."""
        pts = _normalize_points(points.copy())
        best_score = compute_score(pts)
        
        try:
            res = opt.minimize(
                _hard_objective,
                pts.flatten(),
                method='Powell',
                options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate = _normalize_points(res.x.reshape(-1, 2))
            score = compute_score(candidate)
            if score > best_score:
                return candidate, score
        except Exception:
            pass
        
        return pts, best_score

    # Generate diverse initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. Two-ring structures (most promising)
    for r_inner in [0.35, 0.45, 0.55, 0.65]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r_inner, offset=np.pi/6)
        initializations.append(np.vstack([outer, inner]))
    
    # 4. 6 outer + 6 inner + 1 center
    for r_inner in [0.35, 0.45, 0.55, 0.65]:
        outer = _regular_polygon(6, 1.0)
        inner = _regular_polygon(6, r_inner, offset=np.pi/6)
        center = np.array([[0.0, 0.0]])
        initializations.append(np.vstack([outer, inner, center]))
    
    # 5. 8 outer + 5 inner
    for r_inner in [0.35, 0.5, 0.65]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 6. 9 outer + 4 inner
    for r_inner in [0.4, 0.55]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 7. 10 outer + 3 inner
    for r_inner in [0.4, 0.55]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 8. 11 outer + 2 inner
    for r_inner in [0.4, 0.55]:
        outer = _regular_polygon(11, 1.0)
        inner = _regular_polygon(2, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 9. Three-ring structures
    for r_mid, r_inner in [(0.6, 0.3), (0.5, 0.25), (0.7, 0.4)]:
        outer = _regular_polygon(5, 1.0)
        middle = _regular_polygon(5, r_mid, offset=np.pi/5)
        inner = _regular_polygon(3, r_inner)
        initializations.append(np.vstack([outer, middle, inner]))
    
    # 10. Random perturbations
    rng = np.random.RandomState(42)
    for i in range(0, min(8, len(initializations)), 2):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)
    
    # 11. Additional random structured starts
    for _ in range(5):
        n_outer = rng.choice([6, 7, 8, 9])
        n_inner = N - n_outer
        r_outer = rng.uniform(0.9, 1.0)
        r_inner = rng.uniform(0.35, 0.6)
        outer = _regular_polygon(n_outer, r_outer)
        inner = _regular_polygon(n_inner, r_inner, offset=rng.uniform(0, 2*np.pi/max(n_inner,1)))
        pts = np.vstack([outer, inner])
        pts += rng.randn(*pts.shape) * 0.05
        initializations.append(pts)

    best_score = -float('inf')
    best_points = None

    n_inits = len(initializations)
    
    # Phase 1: Tail optimization (sum of k smallest) with L-BFGS-B
    # This lifts the entire lower tail, finding a better basin
    for init_idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 20:
            break
            
        current = _normalize_points(init.copy())
        
        # Optimize the tail objective
        try:
            res = opt.minimize(
                lambda x: _tail_objective(x, k=30),
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            current = _normalize_points(res.x.reshape(-1, 2))
        except Exception:
            pass
        
        # Evaluate hard score
        score = compute_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Phase 2: Polish the best candidate with Nelder-Mead on hard objective
    if best_points is not None and time.time() < start_time + time_limit - 15:
        try:
            candidate, score = _refine_nelder_mead(best_points, max_iter=150)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Final Powell polish
    if best_points is not None and time.time() < start_time + time_limit - 5:
        try:
            candidate, score = _refine_powell(best_points, max_iter=100)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Phase 4: Try a few more random restarts with tail+polish if time remains
    rng2 = np.random.RandomState(123)
    for _ in range(5):
        if time.time() > start_time + time_limit - 10:
            break
        # Random structured start
        n_outer = rng2.choice([6, 7, 8, 9])
        n_inner = N - n_outer
        r_outer = rng2.uniform(0.9, 1.0)
        r_inner = rng2.uniform(0.35, 0.6)
        outer = _regular_polygon(n_outer, r_outer)
        inner = _regular_polygon(n_inner, r_inner, offset=rng2.uniform(0, 2*np.pi/max(n_inner,1)))
        pts = np.vstack([outer, inner])
        pts += rng2.randn(*pts.shape) * 0.05
        current = _normalize_points(pts)
        
        try:
            res = opt.minimize(
                lambda x: _tail_objective(x, k=25),
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 15}
            )
            current = _normalize_points(res.x.reshape(-1, 2))
            candidate, score = _refine_nelder_mead(current, max_iter=80)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = _normalize_points(initializations[0])

    return best_points
