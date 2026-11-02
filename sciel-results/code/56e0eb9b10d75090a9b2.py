import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

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
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _local_search(points, max_iters=200, step_init=0.01):
        """
        Local search that directly targets worst triangles.
        Iteratively moves points to improve the minimum triangle area.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        step = step_init
        
        for it in range(max_iters):
            if time.time() > start_time + time_limit - 5:
                break
                
            # Find the worst (smallest area) triangle
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            
            worst_idx = np.argmin(areas)
            worst_area = areas[worst_idx]
            
            if worst_area < 1e-10:
                break
                
            # Get the three points of the worst triangle
            i, j, k = tri_i[worst_idx], tri_j[worst_idx], tri_k[worst_idx]
            
            # Compute centroid of the worst triangle
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            
            # For each point in the worst triangle, try moving it away from the centroid
            # or in directions that increase the area
            improved = False
            for p_idx in [i, j, k]:
                # Try small random perturbations
                for _ in range(10):
                    if time.time() > start_time + time_limit - 5:
                        break
                    
                    # Random direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    
                    # Try different step sizes
                    for s in [step, step * 0.5, step * 2.0]:
                        new_pts = pts.copy()
                        new_pts[p_idx] += direction * s
                        
                        # Check if this improves the minimum area
                        new_score = compute_score(new_pts)
                        if new_score > best_score + 1e-12:
                            pts = new_pts
                            best_score = new_score
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                # No improvement found; decrease step size
                step *= 0.8
                if step < 1e-6:
                    break
            else:
                # Keep step size or slightly increase if making good progress
                step = min(step * 1.1, 0.05)
        
        return pts, best_score

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
        
    # Add more diverse perturbations
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 2):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.15
        initializations.append(perturbed)
    
    # Additional random starts
    for _ in range(10):
        random_pts = rng.randn(N, 2) * 0.5
        initializations.append(random_pts)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 15:
            break
            
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        # Start with low beta to explore, increase to refine
        betas = [15.0, 30.0, 60.0, 120.0, 250.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 10:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Local search targeting worst triangles
        if time.time() > start_time + time_limit - 8:
            break
        try:
            current, ls_score = _local_search(current, max_iters=300, step_init=0.02)
            if ls_score > best_score:
                best_score = ls_score
                best_points = current.copy()
        except Exception:
            pass
        
        # Step 3: Powell with hard objective (robust local search)
        if time.time() > start_time + time_limit - 3:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 4: Final local search polish
        if time.time() > start_time + time_limit - 2:
            break
        if best_points is not None:
            try:
                polished, ps_score = _local_search(best_points, max_iters=100, step_init=0.005)
                if ps_score > best_score:
                    best_score = ps_score
                    best_points = polished.copy()
            except Exception:
                pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
