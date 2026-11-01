"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute a dense grid for the smooth objective
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(2000)
GRID_FINE = _make_fib_grid(8000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
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
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID_FINE, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def fast_grid_estimate(pts, grid_pts=None):
        """Fast grid-based estimate of covering radius."""
        if grid_pts is None:
            grid_pts = GRID_COARSE
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)
        min_dists = np.min(ang_dists, axis=1)
        return float(np.max(min_dists))

    def repulsion_energy_grad(pts, exponent=2.0):
        """Compute repulsion energy and gradient for uniform distribution."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        w = 1.0 / (dist_sq ** exponent)
        np.fill_diagonal(w, 0.0)
        
        w_grad = exponent * (dist_sq ** (-exponent/2 - 1))
        np.fill_diagonal(w_grad, 0.0)
        
        grad = w_grad @ pts
        
        return float(np.sum(w)), grad

    def gradient_descent_repulsion(pts, exponent, alpha_init, decay, max_iter):
        """Optimize points by gradient descent on repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = repulsion_energy_grad(pts, exponent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            pts = pts + alpha * grad
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            alpha *= decay
        
        return pts

    def smooth_covering_objective_and_grad(x_flat, grid_pts, T=30.0):
        """
        Smooth surrogate for covering radius using log-sum-exp over grid points.
        Fully vectorized - no Python loops over grid points.
        
        Objective: t + (1/T) * log(sum_j exp(T * (d_j - t)))
        where d_j = min_k arccos(dot(grid_j, pts_k))
        
        For the gradient, we use soft weights:
        w_j = exp(T * (d_j - t)) / sum_k exp(T * (d_k - t))
        
        And for each grid point j, soft weights over all N points:
        v_k(j) = exp(-T2 * d_jk) / sum_l exp(-T2 * d_jl)
        
        This gives a smooth approximation to the hard min.
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        M = len(grid_pts)

        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)  # M x N
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # M x N
        
        # Hard min distances for each grid point
        min_dists = np.min(ang_dists, axis=1)  # M
        
        # Soft-max approximation of max(min_dists)
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        exp_shifted = np.exp(shifted * T)
        sum_exp = np.sum(exp_shifted)
        val = max_d + np.log(sum_exp) / T
        
        # Weights for each grid point (soft-max)
        w_j = exp_shifted / sum_exp  # M
        
        # For gradient: use soft weights over all N points for each grid point
        # v_k(j) = exp(-T2 * ang_dists[j,k]) / sum_l exp(-T2 * ang_dists[j,l])
        T2 = 50.0  # Temperature for soft-min over points
        neg_dists = -T2 * ang_dists  # M x N
        max_neg = np.max(neg_dists, axis=1, keepdims=True)  # M x 1
        exp_neg = np.exp(neg_dists - max_neg)  # M x N
        sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)  # M x 1
        v = exp_neg / sum_exp_neg  # M x N, soft weights over points
        
        # Gradient of d_jk w.r.t pts_norm[k]:
        # d/d(pts_norm[k]) arccos(dot(grid_j, pts_norm[k]))
        # = -grid_j / sqrt(1 - dot_jk^2)
        # where dot_jk = dot(grid_j, pts_norm[k])
        
        # For each k, accumulate: sum_j w_j * v_k(j) * (-grid_j / sqrt(1 - dot_jk^2))
        # Vectorized: 
        # denom[j,k] = sqrt(1 - dot[j,k]^2)
        denom = np.sqrt(np.maximum(1.0 - dot * dot, 1e-10))  # M x N
        
        # grad_pts_norm[k] = sum_j w_j * v[j,k] * (-grid_pts[j] / denom[j,k])
        # = sum_j (w_j * v[j,k] / denom[j,k]) * (-grid_pts[j])
        
        # weights[j,k] = w_j * v[j,k] / denom[j,k]  # M x N
        weights = w_j[:, np.newaxis] * v / denom  # M x N
        
        # grad_pts_norm[k] = -sum_j weights[j,k] * grid_pts[j]
        # = -(weights.T @ grid_pts)[k]
        grad_pts_norm = -(weights.T @ grid_pts)  # N x 3
        
        # Chain rule: project gradient to tangent space and scale by 1/norm
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            # d(pts_norm[i])/d(pts[i]) = (I - p p^T)/n
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return float(val), grad

    def optimize_with_lbfgsb(points, grid_pts, maxiter=200, T=30.0):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = smooth_covering_objective_and_grad(x, grid_pts, T)
            return val
        
        def gradient(x):
            _, g = smooth_covering_objective_and_grad(x, grid_pts, T)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 400, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            rng = np.random.RandomState(200 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 4:
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(400 + restart)
            points = points + 0.1 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 5:
            # Two-scale: start from fibonacci, add structured perturbation
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(500 + restart)
            # Add low-frequency perturbation
            for i in range(N):
                angle = rng.uniform(0, 2*np.pi)
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                direction -= np.dot(direction, points[i]) * points[i]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    points[i] += 0.15 * direction / dir_norm
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 6:
            # Random with antipodal symmetry consideration
            rng = np.random.RandomState(600 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(700 + restart)
                points = best_points + 0.08 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 3.0, 0.08, 0.985, 300)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 15.0
        if time_left > 15.0:
            points, _ = optimize_with_lbfgsb(points, GRID_COARSE, maxiter=150, T=30.0)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 30.0:
        for trial in range(5):
            if time.time() > deadline - 20.0:
                break
            rng = np.random.RandomState(800 + trial)
            candidate = best_points + 0.03 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 10.0
            if time_left > 8.0:
                candidate, _ = optimize_with_lbfgsb(candidate, GRID_FINE, maxiter=100, T=40.0)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Improved Local Search using fast grid estimate for pruning
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()
        improved = True
        max_rounds = 20
        round_count = 0
        
        while improved and round_count < max_rounds and time.time() < deadline - 5.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                old_p = points[i].copy()
                
                # Try perturbing point i in random directions
                rng = np.random.RandomState(i * 1000 + round_count * 100)
                for trial in range(12):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p  # Project to tangent
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    # Adaptive step sizes
                    for step in [0.03, 0.015, 0.008, 0.004, 0.002]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        
                        # Fast grid-based estimate first
                        est_rad = fast_grid_estimate(points, GRID_COARSE)
                        
                        # Only do expensive hull check if grid estimate looks promising
                        if est_rad < best_rad + 0.003:
                            rad = compute_covering_radius_exact(points)
                            if rad < best_rad - 1e-10:
                                best_rad = rad
                                best_points = points.copy()
                                improved = True
                                old_p = new_p.copy()
                                break
                            else:
                                points[i] = old_p
                        else:
                            points[i] = old_p
                    else:
                        continue
                    break

    # Phase 4: Final polish with L-BFGS-B using fine grid + local search refinement
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 3.0
        if time_left > 5.0:
            # Use more iterations if time permits
            max_iter = 120 if time_left > 25.0 else 60
            final, _ = optimize_with_lbfgsb(final, GRID_FINE, maxiter=max_iter, T=50.0)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final.copy()
            
            # Additional local search on the polished result
            if time.time() < deadline - 5.0:
                points = best_points.copy()
                rng = np.random.RandomState(999)
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    old_p = points[i].copy()
                    for step in [0.01, 0.005, 0.002]:
                        direction = rng.randn(3)
                        direction -= np.dot(direction, old_p) * old_p
                        dir_norm = np.linalg.norm(direction)
                        if dir_norm < 1e-10:
                            continue
                        direction /= dir_norm
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        rad = compute_covering_radius_exact(points)
                        if rad < best_rad - 1e-10:
                            best_rad = rad
                            best_points = points.copy()
                            old_p = new_p.copy()
                        else:
                            points[i] = old_p

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
