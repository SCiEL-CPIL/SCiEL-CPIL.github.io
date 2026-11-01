"""Program for covering the sphere with 130 equal caps using repulsion-based optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

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
            
            # Compute circumradius of each face
            radii = np.zeros(len(faces))
            for i in range(len(faces)):
                ai, bi, ci = a[i], b[i], c[i]
                # Side lengths
                ab = np.linalg.norm(ai - bi)
                bc = np.linalg.norm(bi - ci)
                ca = np.linalg.norm(ci - ai)
                # Area using cross product
                area = 0.5 * np.linalg.norm(np.cross(ai - bi, ai - ci))
                if area < 1e-10:
                    radii[i] = 0.5 * max(ab, bc, ca)
                else:
                    radii[i] = (ab * bc * ca) / (4.0 * area)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 80)
            lon = np.linspace(0, 2*np.pi, 160, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def make_fib_grid(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    grid_pts_coarse = make_fib_grid(3000)
    grid_pts_fine = make_fib_grid(10000)

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
            if time.time() > deadline - 10.0:
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

    def facet_objective_and_grad(x_flat, grid_pts=None):
        """
        Objective: soft-max of angular distances from a fixed set of sphere points to nearest configuration point.
        Uses a smooth approximation via log-sum-exp.
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts_norm = pts / norms

        if grid_pts is None:
            grid_pts = make_fib_grid(2000)
        M = len(grid_pts)

        # Compute angular distances: M x N
        dot = np.dot(grid_pts, pts_norm.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)

        # Soft-max approximation of max(min_dists)
        temperature = 50.0
        min_dists = np.min(ang_dists, axis=1)
        
        # For the gradient, we need d/dpts of max_j min_i d(y_j, p_i)
        # We approximate this by finding for each y_j its nearest p_i, then using softmax over y_j
        argmin_k = np.argmin(ang_dists, axis=1)
        
        # Soft weights over the grid points
        max_d = np.max(min_dists)
        shifted = min_dists - max_d
        exp_shifted = np.exp(shifted * temperature)
        weights = exp_shifted / np.sum(exp_shifted)
        
        # Objective value
        obj = float(np.log(np.sum(exp_shifted)) / temperature + max_d)
        
        # Gradient computation
        grad_pts_norm = np.zeros((N, 3))
        
        # For each grid point j, the nearest configuration point is k = argmin_k[j]
        # The gradient of d(y_j, p_k) w.r.t. p_k is -y_j / sin(d_jk) projected to tangent
        for j in range(M):
            k = argmin_k[j]
            w = weights[j]
            d = ang_dists[j, k]
            sin_d = np.sin(d)
            if sin_d < 1e-10:
                sin_d = 1e-10
            # Gradient of arccos(dot(y,p)) w.r.t. p is -y/sqrt(1-dot^2)
            grad_contrib = -w * grid_pts[j] / sin_d
            # Project to tangent plane at p_k
            p_k = pts_norm[k]
            grad_tangent = grad_contrib - np.dot(grad_contrib, p_k) * p_k
            grad_pts_norm[k] += grad_tangent
        
        # Chain rule: d(pts_norm)/d(pts)
        grad = np.zeros_like(x_flat)
        for i in range(N):
            p = pts_norm[i]
            n = norms[i, 0]
            grad_i = (grad_pts_norm[i] - np.dot(grad_pts_norm[i], p) * p) / n
            grad[i*3:(i+1)*3] = grad_i
        
        return obj, grad

    def optimize_with_lbfgsb(points, max_time=30.0, grid_pts=None, maxiter=300):
        """Optimize using L-BFGS-B on the flattened coordinates."""
        
        def objective(x):
            val, _ = facet_objective_and_grad(x, grid_pts)
            return val
        
        def gradient(x):
            _, g = facet_objective_and_grad(x, grid_pts)
            return g
        
        # Normalize initial points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        x0 = (points / norms).flatten()
        
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        return x_opt, result.fun

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple starts with repulsion + L-BFGS-B optimization
    n_restarts = 6
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
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
            rng = np.random.RandomState(300 + restart)
            points = points + 0.1 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(400 + restart)
                points = best_points + 0.05 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # First, optimize with repulsion to get a good starting configuration
        points = gradient_descent_repulsion(points, 2.0, 0.05, 0.995, 150)
        
        # Then optimize with L-BFGS-B on the grid-based objective (coarse grid)
        time_left = deadline - time.time() - 15.0
        if time_left > 10.0:
            points, _ = optimize_with_lbfgsb(points, max_time=min(time_left, 35.0), grid_pts=grid_pts_coarse, maxiter=200)
        
        # Evaluate exact covering radius
        rad = compute_covering_radius_exact(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Local refinement with smaller L-BFGS-B runs using fine grid
    if best_points is not None and time.time() < deadline - 25.0:
        for trial in range(4):
            if time.time() > deadline - 12.0:
                break
            rng = np.random.RandomState(500 + trial)
            candidate = best_points + 0.03 * rng.randn(N, 3)
            candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
            
            time_left = deadline - time.time() - 10.0
            if time_left > 5.0:
                candidate, _ = optimize_with_lbfgsb(candidate, max_time=min(time_left, 20.0), grid_pts=grid_pts_fine, maxiter=150)
            
            rad = compute_covering_radius_exact(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: Coordinate-wise local search with exact radius evaluation
    if best_points is not None and time.time() < deadline - 15.0:
        points = best_points.copy()
        improved = True
        max_rounds = 5
        round_count = 0
        
        while improved and round_count < max_rounds and time.time() < deadline - 8.0:
            improved = False
            round_count += 1
            
            for i in range(N):
                if time.time() > deadline - 8.0:
                    break
                
                old_p = points[i].copy()
                
                # Try perturbing point i in random directions
                rng = np.random.RandomState(i * 1000 + round_count)
                for trial in range(5):
                    direction = rng.randn(3)
                    direction -= np.dot(direction, old_p) * old_p  # Project to tangent
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        continue
                    direction /= dir_norm
                    
                    for step in [0.015, 0.008, 0.004, 0.002]:
                        new_p = old_p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        points[i] = new_p
                        
                        rad = compute_covering_radius_exact(points)
                        if rad < best_rad - 1e-12:
                            best_rad = rad
                            best_points = points.copy()
                            improved = True
                            break
                        else:
                            points[i] = old_p
                    else:
                        continue
                    break
                
                if not improved or rad >= best_rad - 1e-12:
                    points[i] = old_p

    # Phase 4: Final polish with L-BFGS-B using fine grid
    if best_points is not None and time.time() < deadline - 5.0:
        final = best_points.copy()
        time_left = deadline - time.time() - 3.0
        if time_left > 3.0:
            final, _ = optimize_with_lbfgsb(final, max_time=min(time_left, 12.0), grid_pts=grid_pts_fine, maxiter=100)
            rad_final = compute_covering_radius_exact(final)
            if rad_final < best_rad:
                best_rad = rad_final
                best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
