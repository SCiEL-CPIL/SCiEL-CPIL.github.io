import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 118.0 # Safety margin for 120s limit

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

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for L-BFGS."""
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
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        return -soft_min_area

    def _hard_objective(points_flat):
        """Hard-min objective for Powell/Nelder-Mead."""
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
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
        
    # Add random perturbations to break symmetry
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)

    # Add a few purely random initializations
    for _ in range(5):
        random_pts = rng.randn(N, 2)
        initializations.append(random_pts)

    best_score = -float('inf')
    best_points = None

    def _active_set_refinement(points, max_iters=50):
        """
        Refine points by focusing on the active constraints (smallest triangles).
        We perform a local search that tries to increase the area of the smallest triangles.
        """
        current_pts = points.copy()
        current_score = compute_score(current_pts)
        if current_score <= 0:
            return current_pts, current_score
            
        rng_local = np.random.RandomState(12345)
        
        for _ in range(max_iters):
            if time.time() > start_time + time_limit - 2:
                break
                
            # Get areas to find critical triangles
            pts_norm = _normalize_points(current_pts)
            x = pts_norm[:, 0]
            y = pts_norm[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            
            # Find indices of the smallest triangles
            min_area = np.min(areas)
            # Tolerance for "active" constraints
            tol = 1e-4 * min_area if min_area > 0 else 1e-4
            active_mask = areas <= (min_area + tol)
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) == 0:
                break
                
            # Get unique vertices involved in the active triangles
            critical_verts = set()
            for idx in active_indices:
                critical_verts.add(tri_i[idx])
                critical_verts.add(tri_j[idx])
                critical_verts.add(tri_k[idx])
            
            # Try to move these vertices to increase the minimum area
            # We use a simple gradient ascent step on the soft-min objective
            # restricted to these vertices, or just random perturbations weighted by activity.
            
            new_pts = current_pts.copy()
            
            # Calculate gradient of the soft-min objective with respect to the vertices
            # This is complex, so we use a heuristic: move vertices away from each other
            # if they form a small triangle.
            
            # For each active triangle, try to expand it.
            # A simple method: move each vertex slightly away from the centroid of the triangle.
            # But we must consider that vertices are shared.
            
            # Alternative: Use a small random perturbation on the critical vertices
            # and accept if it improves the score.
            step_size = 0.01 * (1.0 + 0.5 * rng_local.rand())
            
            for v in critical_verts:
                # Random direction
                dir_vec = rng_local.randn(2)
                dir_vec /= np.linalg.norm(dir_vec) + 1e-9
                new_pts[v] += dir_vec * step_size
            
            new_score = compute_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                # If no improvement, try a smaller step or different direction
                # Here we just continue
                pass
                
        return current_pts, current_score

    # Optimize each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 15:
            break
            
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective
        # Start with low beta for smoothness, then increase
        betas = [10.0, 50.0, 200.0]
        for beta in betas:
            if time.time() > start_time + time_limit - 5:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Powell with hard objective
        if time.time() > start_time + time_limit - 3:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            current = res.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Step 3: Active Set Refinement
        if time.time() > start_time + time_limit - 1:
            break
        current, score = _active_set_refinement(current, max_iters=100)
        
        if score > best_score:
            best_score = score
            best_points = current.copy()
            
        # Step 4: Final Polish with Powell on the refined point
        if time.time() > start_time + time_limit - 0.5:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            final_score = compute_score(candidate)
            if final_score > best_score:
                best_score = final_score
                best_points = candidate.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = initializations[0]

    return best_points
