# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13():
    """
    Maximize the ratio of the smallest triangle area to the convex hull area
    for 13 points using smooth log-sum-exp soft-min with L-BFGS-B.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s kill

    # Precompute triangle indices
    tri_indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tri = len(tri_indices)

    def _compute_ratio(points):
        """Compute the true ratio: min triangle area / hull area."""
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) * 0.5
        min_area = np.min(areas)
        if min_area < 1e-12:
            return 0.0
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
        if hull_area < 1e-12:
            return 0.0
        return min_area / hull_area

    def _soft_min_objective(flat, beta=50.0):
        """
        Smooth approximation of -min_area / hull_area using log-sum-exp.
        Minimizes this (negative smooth max of areas) to maximize min area.
        """
        points = flat.reshape(n, 2)
        if not np.all(np.isfinite(points)):
            return 1e10, None
        
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) * 0.5
        
        # Log-sum-exp for stable soft-min: soft_min ≈ -log(sum(exp(-areas/beta)))/beta
        neg_areas = -areas / beta
        max_neg = np.max(neg_areas)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
        soft_min_area = -log_sum_exp * beta
        
        # Hull area (use smooth approximation via convex hull)
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e10, None
        
        if hull_area < 1e-12:
            return 1e10, None
        
        score = soft_min_area / hull_area
        return -score, None

    def _objective_with_grad(flat, beta=50.0):
        """Objective and gradient for L-BFGS-B."""
        val, _ = _soft_min_objective(flat, beta)
        if val > 1e9:
            return val, np.zeros_like(flat)
        
        # Numerical gradient (central differences)
        eps = 1e-6
        grad = np.zeros_like(flat)
        for i in range(len(flat)):
            fp = flat.copy()
            fm = flat.copy()
            fp[i] += eps
            fm[i] -= eps
            vp, _ = _soft_min_objective(fp, beta)
            vm, _ = _soft_min_objective(fm, beta)
            if vp > 1e9 or vm > 1e9:
                grad[i] = 0.0
            else:
                grad[i] = (vp - vm) / (2 * eps)
        return val, grad

    def _make_starting_points():
        """Generate diverse starting configurations."""
        starts = []
        
        # 1. Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        reg = np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
        starts.append(reg)
        
        # 2. Hexagonal lattice subset
        s = 0.22
        hex_pts = []
        for row in range(-3, 4):
            for col in range(-3, 4):
                x = 0.5 + col * s + (row % 2) * s / 2
                y = 0.5 + row * s * np.sqrt(3) / 2
                hex_pts.append([x, y])
        hex_pts = np.array(hex_pts)
        mask = (hex_pts[:, 0] - 0.5)**2 + (hex_pts[:, 1] - 0.5)**2 <= 0.42**2
        hex_pts = hex_pts[mask]
        if len(hex_pts) >= n:
            # Pick most spread out n points
            idx = np.random.choice(len(hex_pts), n, replace=False)
            starts.append(hex_pts[idx])
        
        # 3. Circle with perturbation
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.uniform(-0.1, 0.1, n)
        r = 0.45 + np.random.uniform(-0.05, 0.05, n)
        circ = np.column_stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)])
        starts.append(circ)
        
        # 4. Random points in unit square
        rand_pts = np.random.uniform(0.05, 0.95, (n, 2))
        starts.append(rand_pts)
        
        # 5. Two concentric rings
        inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.2
        inner = np.column_stack([0.5 + 0.2 * np.cos(inner_angles), 0.5 + 0.2 * np.sin(inner_angles)])
        outer = np.column_stack([0.5 + 0.45 * np.cos(outer_angles), 0.5 + 0.45 * np.sin(outer_angles)])
        ring_pts = np.vstack([inner, outer])
        starts.append(ring_pts)
        
        # 6. Grid-like arrangement
        grid_pts = np.array([
            [0.2, 0.2], [0.5, 0.2], [0.8, 0.2],
            [0.3, 0.5], [0.5, 0.5], [0.7, 0.5],
            [0.2, 0.8], [0.4, 0.8], [0.6, 0.8], [0.8, 0.8],
            [0.35, 0.35], [0.65, 0.65], [0.5, 0.65]
        ])
        starts.append(grid_pts)
        
        return starts

    # Generate starting points
    np.random.seed(42)
    starts = _make_starting_points()
    
    best_ratio = 0.0
    best_points = None
    
    # Optimize from each starting point with increasing beta (coarser to finer)
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for start_idx, start_pts in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        current = start_pts.flatten().copy()
        
        # Progressive optimization with increasing beta
        for beta in betas:
            if time.time() > deadline - 20:
                break
            
            try:
                result = minimize(
                    _objective_with_grad,
                    current,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * n),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
                )
                if result.fun < 0:  # Valid result (negative of positive ratio)
                    current = result.x
            except Exception:
                pass
        
        # Evaluate true ratio at the optimized point
        opt_pts = current.reshape(n, 2)
        # Clip to valid range
        opt_pts = np.clip(opt_pts, 0.0, 1.0)
        ratio = _compute_ratio(opt_pts)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = opt_pts.copy()
    
    # Final local refinement with Nelder-Mead on the best found
    if best_points is not None and time.time() < deadline - 10:
        try:
            def nm_obj(flat):
                pts = flat.reshape(n, 2)
                r = _compute_ratio(pts)
                return -r
            
            result = minimize(
                nm_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if -result.fun > best_ratio:
                best_points = result.x.reshape(n, 2)
                best_ratio = -result.fun
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
