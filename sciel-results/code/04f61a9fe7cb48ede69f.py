"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_hull(pts):
        """Compute exact covering radius via convex hull facets."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
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

    def make_grid(n_theta=60, n_phi=120):
        """Create a grid of points on the unit sphere."""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    grid_points = make_grid(60, 120)  # ~7200 points - faster

    def soft_max_objective(flat_pts, beta=50.0, top_k=200):
        """
        Smooth surrogate for covering radius using soft-max.
        Only computes gradient for top-K grid points to save time.
        Input: flat array of 3N coordinates.
        Returns: (objective_value, gradient)
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        dots = grid_points @ pts.T  # shape (G, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # shape (G, N)
        
        # Soft-min over j (config points) for each grid point i
        neg_beta_dists = -beta * dists
        max_neg = np.max(neg_beta_dists, axis=1, keepdims=True)
        log_sum_exp_inner = np.log(np.sum(np.exp(neg_beta_dists - max_neg), axis=1)) + max_neg.ravel()
        soft_min = -log_sum_exp_inner / beta  # shape (G,)
        
        # Soft-max over grid points i
        beta_soft_min = beta * soft_min
        max_outer = np.max(beta_soft_min)
        log_sum_exp_outer = np.log(np.sum(np.exp(beta_soft_min - max_outer))) + max_outer
        objective = log_sum_exp_outer / beta
        
        # Gradient computation - only for top-K grid points
        outer_weights = np.exp(beta_soft_min - max_outer)
        outer_weights /= np.sum(outer_weights)
        
        # Select top-K grid points by outer weight
        k = min(top_k, len(outer_weights))
        top_indices = np.argsort(outer_weights)[-k:]
        outer_weights_top = outer_weights[top_indices]
        
        inner_weights_top = np.exp(neg_beta_dists[top_indices] - max_neg[top_indices])  # shape (K, N)
        inner_weights_top /= np.sum(inner_weights_top, axis=1, keepdims=True)
        
        dots_top = dots[top_indices]  # shape (K, N)
        one_minus_dot_sq = 1.0 - dots_top * dots_top
        one_minus_dot_sq = np.maximum(one_minus_dot_sq, 1e-12)
        sqrt_oms = np.sqrt(one_minus_dot_sq)
        
        coeff = inner_weights_top * dots_top / sqrt_oms  # shape (K, N)
        
        weighted_coeff = coeff * outer_weights_top[:, np.newaxis]  # shape (K, N)
        grad_pts = weighted_coeff.T @ grid_points[top_indices]  # shape (N, 3)
        
        grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
        grad_flat = grad_tangent.ravel()
        
        return objective, grad_flat

    def repulsion_force(points, exponent=1.5):
        """Compute repulsion forces between all point pairs."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        return forces

    def gradient_descent(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            forces = repulsion_force(points, exponent)
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def compute_facet_radii(pts):
        """Compute radii for all hull facets. Returns (radii_array, faces_array)."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return radii, faces
        except Exception:
            return np.array([np.inf]), np.array([[0, 1, 2]])

    def targeted_refinement(points, max_iters=300):
        """Refine by targeting the worst facet: move its 3 vertices to reduce circumradius."""
        pts = points.copy()
        step = 0.005
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            radii, faces = compute_facet_radii(pts)
            max_idx = np.argmax(radii)
            rad = radii[max_idx]
            face = faces[max_idx]
            
            i, j, k = face
            a = pts[i].copy()
            b = pts[j].copy()
            c = pts[k].copy()
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            c_norm = np.linalg.norm(c_dir)
            
            if c_norm < 1e-10:
                mid = (a + b + c) / 3.0
                mid /= np.linalg.norm(mid)
                for idx, p in [(i, a), (j, b), (k, c)]:
                    direction = p - 0.1 * mid
                    direction /= np.linalg.norm(direction)
                    pts[idx] = direction
                continue
            
            c_dir /= c_norm
            
            # Try moving vertices in the direction that reduces circumradius
            improved = False
            for trial_step in [step, step * 0.5, step * 0.25, step * 0.1]:
                for idx, p in [(i, a), (j, b), (k, c)]:
                    tangent = c_dir - np.dot(c_dir, p) * p
                    t_norm = np.linalg.norm(tangent)
                    if t_norm > 1e-10:
                        tangent /= t_norm
                        new_p = p + trial_step * tangent
                        new_p /= np.linalg.norm(new_p)
                        pts[idx] = new_p
                
                new_rad = compute_covering_radius_hull(pts)
                if new_rad < rad:
                    improved = True
                    step = trial_step
                    break
                else:
                    # Revert
                    pts[i] = a
                    pts[j] = b
                    pts[k] = c
            
            if not improved:
                no_improve_count += 1
                step *= 0.5
                if step < 1e-6 or no_improve_count > 10:
                    break
            else:
                no_improve_count = 0
        
        return pts

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion-based gradient descent
    n_restarts = 5
    exponents = [1.0, 1.5, 2.0, 2.5, 1.2]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 50.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(200)
            points = points + 0.5 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(300)
                points = best_points + 0.15 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 150, exp)
        rad = compute_covering_radius_hull(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: L-BFGS-B optimization with smooth soft-max surrogate
    if best_points is not None and time.time() < deadline - 25.0:
        for beta in [30.0, 50.0, 80.0]:
            if time.time() > deadline - 12.0:
                break
            
            x0 = best_points.ravel()
            
            def objective_wrapper(x):
                val, _ = soft_max_objective(x, beta=beta, top_k=150)
                return val
            
            def gradient_wrapper(x):
                _, grad = soft_max_objective(x, beta=beta, top_k=150)
                return grad
            
            try:
                result = minimize(
                    objective_wrapper,
                    x0,
                    jac=gradient_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 40, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 80}
                )
                
                if result.success or result.fun < objective_wrapper(x0):
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

    # Phase 3: Targeted refinement on the worst facet
    if best_points is not None and time.time() < deadline - 10.0:
        refined = targeted_refinement(best_points.copy(), max_iters=300)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Phase 4: Final L-BFGS-B polish with higher beta
    if best_points is not None and time.time() < deadline - 8.0:
        beta = 100.0
        x0 = best_points.ravel()
        
        def objective_wrapper(x):
            val, _ = soft_max_objective(x, beta=beta, top_k=100)
            return val
        
        def gradient_wrapper(x):
            _, grad = soft_max_objective(x, beta=beta, top_k=100)
            return grad
        
        try:
            result = minimize(
                objective_wrapper,
                x0,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 25, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50}
            )
            
            if result.success or result.fun < objective_wrapper(x0):
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

    # Final targeted refinement
    if best_points is not None and time.time() < deadline - 5.0:
        refined = targeted_refinement(best_points.copy(), max_iters=150)
        rad = compute_covering_radius_hull(refined)
        if rad < best_rad:
            best_rad = rad
            best_points = refined

    # Final check
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
