"""Program for covering the sphere with 130 equal caps."""
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

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces (vectorized)."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
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
            # Fallback: grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
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

    # Precompute a grid for soft-max surrogate
    lat = np.linspace(-np.pi/2, np.pi/2, 40)
    lon = np.linspace(0, 2*np.pi, 80, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    z_grid = np.sin(LA)
    x_grid = np.cos(LA) * np.cos(LO)
    y_grid = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    def soft_max_covering_radius_and_grad(pts, beta=20.0):
        """Compute soft-max covering radius and its gradient w.r.t. pts.
        
        Uses log-sum-exp trick for numerical stability.
        Returns (radius, grads) where grads has shape (N, 3).
        """
        # Compute angular distances from each grid point to each config point
        dot = np.dot(grid_pts, pts.T)  # (G, N)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # (G, N)
        
        # For each grid point, min distance to any config point
        min_dists = np.min(ang_dists, axis=1)  # (G,)
        
        # Soft-max over grid points
        neg_beta_dists = -beta * min_dists
        max_neg = np.max(neg_beta_dists)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_dists - max_neg)))
        radius = -log_sum / beta
        
        # Gradient
        weights = np.exp(neg_beta_dists - max_neg)
        weights /= np.sum(weights)
        
        # For each grid point g, find which config point i_g gives the min distance
        min_indices = np.argmin(ang_dists, axis=1)  # (G,)
        
        # Gradient of min_dists[g] w.r.t. pts[i_g]:
        grads = np.zeros_like(pts)
        for g in range(len(grid_pts)):
            i = min_indices[g]
            dot_gi = dot[g, i]
            denom = np.sqrt(np.maximum(1.0 - dot_gi * dot_gi, 1e-12))
            grads[i] -= weights[g] * grid_pts[g] / denom
        
        grads /= beta
        return radius, grads

    def gradient_descent_smooth(points, alpha_init, decay, max_iter, beta=20.0):
        """Optimize using smooth surrogate with gradient descent."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            radius, grads = soft_max_covering_radius_and_grad(points, beta)
            points = points + alpha * grads
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def soft_max_objective_and_grad(x_flat, beta=20.0):
        """Flattened objective and gradient for scipy.optimize.minimize."""
        pts = x_flat.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        radius, grads = soft_max_covering_radius_and_grad(pts, beta)
        
        # Project gradient to tangent space (remove radial component)
        pts_flat = pts.flatten()
        grads_flat = grads.flatten()
        # Subtract radial component: grad_tangent = grad - (grad . p) * p
        radial_comp = np.sum(grads_flat * pts_flat.reshape(N, 3), axis=1)
        grads_tangent = grads_flat.reshape(N, 3) - radial_comp[:, np.newaxis] * pts.reshape(N, 3)
        
        return radius, grads_tangent.flatten()

    def soft_max_minimize(points, beta=20.0, max_iter=50):
        """Use scipy.optimize.minimize with the soft-max surrogate."""
        x0 = points.flatten()
        
        def obj(x):
            val, _ = soft_max_objective_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = soft_max_objective_and_grad(x, beta)
            return g
        
        try:
            result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 20})
            pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            return pts
        except Exception:
            return points

    def simulated_annealing(points, t_start=0.05, t_end=0.001, n_iter=200):
        """Simulated annealing to escape local minima."""
        pts = points.copy()
        best_pts = pts.copy()
        best_rad = compute_covering_radius(pts)
        current_rad = best_rad
        n = len(points)
        
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (it / n_iter)
            
            # Random perturbation
            idx = np.random.randint(0, n)
            perturbation = t * np.random.randn(3)
            new_pt = pts[idx] + perturbation
            new_pt /= np.linalg.norm(new_pt)
            
            old_pt = pts[idx].copy()
            pts[idx] = new_pt
            new_rad = compute_covering_radius(pts)
            
            # Accept or reject
            if new_rad < current_rad or np.random.rand() < np.exp(-(new_rad - current_rad) / max(t, 1e-10)):
                current_rad = new_rad
                if new_rad < best_rad:
                    best_rad = new_rad
                    best_pts = pts.copy()
            else:
                pts[idx] = old_pt
        
        return best_pts, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple random starts with repulsion-based optimization
    n_restarts = 4
    exponents = [1.0, 1.5, 2.0, 3.0]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            # Random start
            rng = np.random.RandomState(42 + restart)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Perturbed fibonacci
            points = fibonacci_spiral(N).astype(np.float64)
            rng = np.random.RandomState(100 + restart)
            points = points + 0.3 * rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Use best found so far with perturbation
            if best_points is not None:
                rng = np.random.RandomState(200 + restart)
                points = best_points + 0.15 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart % len(exponents)]
        points = gradient_descent(points, 0.05, 0.995, 200, exp)
        rad = compute_covering_radius(points)
        
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: Smooth surrogate optimization with gradient descent
    if best_points is not None and time.time() < deadline - 30.0:
        candidate = best_points.copy()
        candidate = gradient_descent_smooth(candidate, 0.02, 0.995, 150, beta=20.0)
        rad = compute_covering_radius(candidate)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()
        
        # Try with different beta
        if time.time() < deadline - 20.0:
            candidate = best_points.copy()
            candidate = gradient_descent_smooth(candidate, 0.01, 0.99, 100, beta=30.0)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 3: scipy.optimize.minimize with soft-max surrogate (structural change)
    if best_points is not None and time.time() < deadline - 25.0:
        candidate = soft_max_minimize(best_points.copy(), beta=20.0, max_iter=50)
        rad = compute_covering_radius(candidate)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()
        
        # Try with higher beta for sharper approximation
        if time.time() < deadline - 15.0:
            candidate = soft_max_minimize(best_points.copy(), beta=40.0, max_iter=50)
            rad = compute_covering_radius(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 4: Simulated annealing to escape local minima
    if best_points is not None and time.time() < deadline - 20.0:
        candidate, rad = simulated_annealing(best_points.copy(), t_start=0.05, t_end=0.001, n_iter=200)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()
        
        # Second SA run with different parameters
        if time.time() < deadline - 10.0:
            candidate, rad = simulated_annealing(best_points.copy(), t_start=0.03, t_end=0.0005, n_iter=150)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()

    # Phase 5: Final polish with smooth surrogate
    if best_points is not None and time.time() < deadline - 5.0:
        final = gradient_descent_smooth(best_points.copy(), 0.005, 0.99, 50, beta=40.0)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
