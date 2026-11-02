import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

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

    def _compute_areas(pts):
        """Compute all triangle areas for normalized points."""
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        return 0.5 * cross

    def _penalty_objective(x_flat, t, penalty_weight=100.0):
        """
        Penalty-based objective for max-min optimization.
        x_flat: 26 point coordinates
        t: target minimum triangle area
        Returns: -t + penalty for violations (area_i < t)
        """
        pts = x_flat[:2*N].reshape(N, 2)
        pts = _normalize_points(pts)
        areas = _compute_areas(pts)
        
        # Penalty for triangles with area < t
        violations = np.maximum(0, t - areas)
        penalty = penalty_weight * np.sum(violations ** 2)
        
        return -t + penalty

    def _feasibility_check(x_flat, t):
        """Check if all triangles have area >= t."""
        pts = x_flat[:2*N].reshape(N, 2)
        pts = _normalize_points(pts)
        areas = _compute_areas(pts)
        return np.min(areas) >= t

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

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
        
    # Add some random perturbations to top symmetric configs to break symmetry if beneficial
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 15:
            break
            
        current = init.copy()
        current_score = compute_score(current)
        
        # Step 1: Use penalty-based max-min optimization
        # Start with a conservative t, then increase it
        if current_score > 0:
            t_target = current_score * 0.9  # Start slightly below current score
        else:
            t_target = 0.05  # Conservative default
        
        # Iteratively increase t while maintaining feasibility
        penalty_weights = [50.0, 100.0, 200.0, 500.0]
        
        for pw in penalty_weights:
            if time.time() > start_time + time_limit - 10:
                break
            
            # Try to increase t in small steps
            t_step = t_target * 0.01
            max_iterations = 20
            
            for _ in range(max_iterations):
                if time.time() > start_time + time_limit - 5:
                    break
                
                # Optimize points for current t
                x0 = np.concatenate([current.flatten(), [t_target]])
                
                try:
                    res = opt.minimize(
                        lambda x: _penalty_objective(x, t_target, pw),
                        x0,
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    new_pts = res.x[:2*N].reshape(N, 2)
                    new_score = compute_score(new_pts)
                    
                    if new_score > current_score:
                        current = new_pts
                        current_score = new_score
                    
                    # Check if we can increase t
                    if _feasibility_check(new_pts.flatten(), t_target):
                        # Feasible, try increasing t
                        t_target += t_step
                    else:
                        # Not feasible, keep current t
                        break
                        
                except Exception:
                    break
        
        # Step 2: Powell refinement on hard objective
        if time.time() > start_time + time_limit - 3:
            break
        try:
            res = opt.minimize(
                lambda x: -compute_score(x.reshape(-1, 2)),
                current.flatten(),
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
