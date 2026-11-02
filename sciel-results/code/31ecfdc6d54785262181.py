"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute grids at module level to avoid re-creation overhead
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    def fibonacci_spiral(n, offset=0.0):
        i = np.arange(n) + 0.5 + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_hull(pts):
        """Compute exact covering radius via convex hull facets."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = b - a
            ac = c - a
            normal = np.cross(ab, ac)
            norms = np.linalg.norm(normal, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                cc = normal[valid] / norms[valid, np.newaxis]
                centroids = (a[valid] + b[valid] + c[valid]) / 3.0
                dots = np.sum(cc * centroids, axis=1)
                cc[dots < 0] *= -1
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            return np.inf

    def repulsion_coverage_objective(flat_pts, alpha=50.0):
        """
        Simple repulsion + coverage objective.
        Points repel each other and are attracted to cover the sphere.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Repulsion: minimize sum of 1/d^2 for close pairs
        dots = pts @ pts.T
        np.fill_diagonal(dots, 0)
        dots = np.clip(dots, -1.0, 1.0)
        dists_sq = 2.0 * (1.0 - dots)
        dists_sq = np.maximum(dists_sq, 1e-12)
        repulsion = np.sum(1.0 / dists_sq)
        
        # Coverage: use grid to measure how well the sphere is covered
        grid = GRID_COARSE
        grid_dots = grid @ pts.T  # (G, N)
        grid_dots = np.clip(grid_dots, -1.0, 1.0)
        grid_dists = np.arccos(grid_dots)
        # Soft-min over points for each grid point
        neg_beta = -alpha * grid_dists
        max_neg = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(neg_beta - max_neg), axis=1)) + max_neg.ravel()
        soft_min = -log_sum_exp / alpha
        coverage = np.max(soft_min)
        
        # Combined objective
        obj = repulsion * 0.001 + coverage
        
        # Gradient
        # Repulsion gradient
        grad_rep = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d_sq = dists_sq[i, j]
                # d/dp_i (1/d^2) where d^2 = 2(1-p_i.p_j)
                # = -1/d^4 * 2 * (p_j - p_i * (p_i.p_j))
                coeff = -2.0 / (d_sq * d_sq)
                tangent = pts[j] - dots[i, j] * pts[i]
                grad_rep[i] += coeff * tangent
        
        # Coverage gradient
        outer_weights = np.exp(alpha * soft_min - np.max(alpha * soft_min))
        outer_weights /= np.sum(outer_weights)
        
        inner_weights = np.exp(neg_beta - max_neg)
        inner_weights /= np.sum(inner_weights, axis=1, keepdims=True)
        
        one_minus_dot_sq = 1.0 - grid_dots * grid_dots
        one_minus_dot_sq = np.maximum(one_minus_dot_sq, 1e-12)
        sqrt_oms = np.sqrt(one_minus_dot_sq)
        
        weighted = outer_weights[:, np.newaxis] * inner_weights / sqrt_oms
        term1 = grid.T @ weighted
        term2 = np.sum(weighted * grid_dots, axis=0)
        grad_cov = -(term1 - term2[:, np.newaxis] * pts)
        grad_cov -= np.sum(grad_cov * pts, axis=1, keepdims=True) * pts
        
        grad = 0.001 * grad_rep + grad_cov
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        return obj, grad_tangent.ravel()

    def targeted_refinement(points, max_iters=300, step_scale=1.0):
        """Refine by targeting the worst facet."""
        pts = points.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            try:
                hull = ConvexHull(pts)
                faces = hull.simplices
                a = pts[faces[:, 0]]
                b = pts[faces[:, 1]]
                c = pts[faces[:, 2]]
                
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                norms = np.linalg.norm(normal, axis=1)
                
                radii = np.zeros(len(faces))
                valid = norms > 1e-10
                
                if np.any(valid):
                    cc = normal[valid] / norms[valid, np.newaxis]
                    centroids = (a[valid] + b[valid] + c[valid]) / 3.0
                    dots_cc = np.sum(cc * centroids, axis=1)
                    cc[dots_cc < 0] *= -1
                    cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    radii[valid] = r
                
                if np.any(~valid):
                    a_deg = a[~valid]
                    b_deg = b[~valid]
                    c_deg = c[~valid]
                    d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                    d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                    d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                    radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                
                max_idx = np.argmax(radii)
                rad = radii[max_idx]
                face = faces[max_idx]
            except Exception:
                break
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            ab = b - a
            ac = c - a
            c_dir = np.cross(ab, ac)
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    direction = p - 0.05 * mid
                    direction /= np.linalg.norm(direction)
                    pts[idx] = direction
                continue
            
            c_dir /= c_norm
            antipode = -c_dir
            
            improved = False
            steps = [0.02 * step_scale, 0.01 * step_scale, 0.005 * step_scale, 0.002 * step_scale, 0.001 * step_scale]
            for step in steps:
                new_a = a + step * antipode
                new_b = b + step * antipode
                new_c = c + step * antipode
                
                new_a /= np.linalg.norm(new_a)
                new_b /= np.linalg.norm(new_b)
                new_c /= np.linalg.norm(new_c)
                
                pts[i] = new_a
                pts[j] = new_b
                pts[k] = new_c
                
                new_rad = compute_covering_radius_hull(pts)
                if new_rad < rad:
                    improved = True
                    break
                else:
                    pts[i] = a
                    pts[j] = b
                    pts[k] = c
            
            if not improved:
                for idx, p in [(i, a), (j, b), (k, c)]:
                    for step in [0.01 * step_scale, 0.005 * step_scale, 0.002 * step_scale]:
                        new_p = p + step * antipode
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                        new_rad = compute_covering_radius_hull(pts)
                        if new_rad < rad:
                            improved = True
                            break
                        else:
                            pts[idx] = p
                    if improved:
                        break
            
            if not improved:
                rng = np.random.RandomState(it)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    for _ in range(3):
                        rand_vec = rng.randn(3)
                        tangent = rand_vec - np.dot(rand_vec, p) * p
                        t_norm = np.linalg.norm(tangent)
                        if t_norm < 1e-10:
                            continue
                        tangent /= t_norm
                        new_p = p + 0.003 * tangent
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                        new_rad = compute_covering_radius_hull(pts)
                        if new_rad < rad:
                            improved = True
                            break
                        else:
                            pts[idx] = p
                    if improved:
                        break
            
            if not improved:
                break
        
        return pts

    best_points = None
    best_rad = np.inf

    # Try multiple initializations
    for restart in range(3):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        else:
            offset = 0.3 * restart
            points = fibonacci_spiral(N, offset=offset).astype(np.float64)
        
        # L-BFGS-B with repulsion + coverage objective
        x0 = points.ravel()
        
        def objective_wrapper(x):
            val, _ = repulsion_coverage_objective(x, alpha=50.0)
            return val
        
        def gradient_wrapper(x):
            _, grad = repulsion_coverage_objective(x, alpha=50.0)
            return grad
        
        try:
            result = minimize(
                objective_wrapper,
                x0,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 150}
            )
            candidate = result.x.reshape(N, 3)
            norms = np.linalg.norm(candidate, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            candidate = candidate / norms
            rad = compute_covering_radius_hull(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).copy()
        best_rad = compute_covering_radius_hull(best_points)

    # Phase 2: Targeted refinement
    if time.time() < deadline - 15.0:
        refined = targeted_refinement(best_points.copy(), max_iters=400, step_scale=1.0)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined.copy()

    # Phase 3: Fine polish
    if time.time() < deadline - 5.0:
        refined = targeted_refinement(best_points.copy(), max_iters=200, step_scale=0.3)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    return best_points

# EVOLVE-BLOCK-END
