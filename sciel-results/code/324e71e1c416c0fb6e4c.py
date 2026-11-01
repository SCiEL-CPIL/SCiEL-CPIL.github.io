"""Program for covering the sphere with 130 equal caps using hybrid optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
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

    def lloyd_iteration(points, alpha=1.0):
        """
        One Lloyd/Voronoi relaxation step using convex hull adjacency.
        """
        try:
            hull = ConvexHull(points)
            faces = hull.simplices  # (F, 3)
            
            # For each point, collect its neighbors via the faces
            adj = [[] for _ in range(N)]
            for f in faces:
                for i in range(3):
                    j = (i + 1) % 3
                    a_idx = f[i]
                    b_idx = f[j]
                    adj[a_idx].append(b_idx)
                    adj[b_idx].append(a_idx)
            
            new_points = points.copy()
            for i in range(N):
                if len(adj[i]) > 0:
                    neighbors = points[adj[i]]
                    cell_points = np.vstack([points[i], neighbors])
                    centroid = np.sum(cell_points, axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-10:
                        centroid /= norm
                        new_points[i] = (1 - alpha) * points[i] + alpha * centroid
                        new_points[i] /= np.linalg.norm(new_points[i])
            return new_points
        except Exception:
            return points

    def gradient_descent_repulsion(points, alpha_init, decay, max_iter, exponent=1.5):
        """Optimize points via gradient descent on the repulsion energy."""
        alpha = alpha_init
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
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
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
        return points

    def smooth_objective(x):
        """
        Smooth objective: max nearest-neighbor angular distance over a dense spherical grid.
        Uses a soft-max approximation for differentiability.
        """
        if time.time() > deadline - 5.0:
            return 1e10
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Dense grid for covering radius estimation
        # Use a moderately dense grid for speed
        n_lat = 30
        n_lon = 60
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        xg = np.cos(LA) * np.cos(LO)
        yg = np.cos(LA) * np.sin(LO)
        grid_pts = np.column_stack([xg.ravel(), yg.ravel(), z.ravel()])
        
        # Compute distances
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # shape (n_grid, N)
        min_dists = np.min(ang_dists, axis=1)  # shape (n_grid,)
        
        # Soft max for smoothness
        beta = 50.0
        max_d = np.max(min_dists)
        exp_vals = np.exp(beta * (min_dists - max_d))
        soft_max = max_d + np.log(np.sum(exp_vals)) / beta
        return soft_max

    def smooth_gradient(x):
        """Numerical gradient of the smooth objective."""
        eps = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = smooth_objective(x_plus)
            x_minus = x.copy()
            x_minus[i] -= eps
            f_minus = smooth_objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def lbfgs_refine(points):
        """Use L-BFGS-B to refine points using the smooth objective."""
        if time.time() > deadline - 10.0:
            return points
        
        x0 = points.flatten()
        try:
            result = minimize(
                smooth_objective,
                x0,
                method='L-BFGS-B',
                jac=smooth_gradient,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < smooth_objective(x0):
                new_pts = result.x.reshape(N, 3)
                # Normalize
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                new_pts = new_pts / norms
                return new_pts
        except Exception:
            pass
        return points

    def local_search_perturb(points, max_iterations=50):
        """Local search by perturbing random points and re-optimizing."""
        best_rad = compute_covering_radius(points)
        best_points = points.copy()
        rng = np.random.RandomState(42)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for trial in range(30):
                if time.time() > deadline - 10.0:
                    break
                
                # Perturb a random subset of points
                n_perturb = rng.randint(1, 5)
                idx = rng.choice(N, size=n_perturb, replace=False)
                
                old_pts = points[idx].copy()
                noise = 0.05 * rng.randn(n_perturb, 3)
                points[idx] = old_pts + noise
                points[idx] /= np.linalg.norm(points[idx], axis=1, keepdims=True)
                
                rad = compute_covering_radius(points)
                if rad < best_rad - 1e-9:
                    best_rad = rad
                    best_points = points.copy()
                    improved = True
                    break
                else:
                    points[idx] = old_pts
            
            if not improved:
                break
        
        return best_points, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple random starts with varying exponent
    n_restarts = 3
    exponents = [1.5, 2.0, 2.5]

    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            rng = np.random.RandomState(42)
            points = rng.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is not None:
                rng = np.random.RandomState(100 + restart)
                points = best_points + 0.1 * rng.randn(N, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        exp = exponents[restart]
        points = gradient_descent_repulsion(points, 0.05, 0.995, 200, exp)
        
        # Lloyd iterations for Voronoi relaxation
        lloyd_iters = 100
        for it in range(lloyd_iters):
            if time.time() > deadline - 40.0:
                break
            points = lloyd_iteration(points, alpha=0.5)
        
        rad = compute_covering_radius(points)

        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()

    # Phase 2: L-BFGS refinement using smooth objective
    if best_points is not None and time.time() < deadline - 25.0:
        candidate = lbfgs_refine(best_points.copy())
        rad = compute_covering_radius(candidate)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()

    # Phase 3: Local search with perturbations
    if best_points is not None and time.time() < deadline - 20.0:
        best_points, best_rad = local_search_perturb(best_points, max_iterations=30)

    # Phase 4: Final Lloyd polishing
    if best_points is not None and time.time() < deadline - 15.0:
        candidate = best_points.copy()
        lloyd_iters = 100
        for it in range(lloyd_iters):
            if time.time() > deadline - 10.0:
                break
            candidate = lloyd_iteration(candidate, alpha=0.3)
        rad = compute_covering_radius(candidate)
        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()

    # Phase 5: Final L-BFGS polish
    if best_points is not None and time.time() < deadline - 5.0:
        final = lbfgs_refine(best_points.copy())
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points
