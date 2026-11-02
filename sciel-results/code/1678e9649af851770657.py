"""
Program for covering the sphere with 130 equal caps using a combination of
Lloyd's algorithm (Voronoi-based) and smooth optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_points(n, seed=0):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def lloyd_step(pts, n_iter=10):
        """
        Perform Lloyd's algorithm: assign grid points to nearest site,
        then move each site to the centroid of its assigned points (on the sphere).
        """
        # Use a moderately dense grid for Voronoi cells
        # 60x120 = 7200 points
        n_lat, n_lon = 60, 120
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        current_pts = normalize(pts)
        
        for _ in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Assign each grid point to nearest site
            # dots: (M, N)
            dots = np.dot(grid, current_pts.T)
            dots = np.clip(dots, -1.0, 1.0)
            # Nearest site is the one with max dot product
            labels = np.argmax(dots, axis=1)
            
            # Compute centroid for each site
            new_pts = np.zeros_like(current_pts)
            for i in range(N):
                mask = labels == i
                if np.any(mask):
                    # Centroid of assigned grid points
                    centroid = np.mean(grid[mask], axis=0)
                    norm_c = np.linalg.norm(centroid)
                    if norm_c > 1e-10:
                        new_pts[i] = centroid / norm_c
                    else:
                        new_pts[i] = current_pts[i]
                else:
                    new_pts[i] = current_pts[i]
            
            # Blend to avoid large jumps
            current_pts = 0.5 * current_pts + 0.5 * new_pts
            current_pts = normalize(current_pts)
        
        return current_pts

    def soft_covering_radius(pts, beta=100.0):
        """
        Compute smooth approximation of covering radius using a dense grid.
        """
        n_lat, n_lon = 80, 160
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        pts = normalize(pts)
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang_dists = np.arccos(dots)
        min_dists = np.min(ang_dists, axis=1)
        
        max_d = np.max(min_dists)
        exp_vals = np.exp(beta * (min_dists - max_d))
        soft_max = max_d + np.log(np.sum(exp_vals)) / beta
        return soft_max

    def soft_covering_radius_grad(pts, beta=100.0):
        """
        Compute smooth covering radius and its gradient.
        """
        n_lat, n_lon = 80, 160
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        pts = normalize(pts)
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang_dists = np.arccos(dots)
        min_dists = np.min(ang_dists, axis=1)
        argmins = np.argmin(ang_dists, axis=1)
        
        max_d = np.max(min_dists)
        exp_vals = np.exp(beta * (min_dists - max_d))
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient with respect to each point
        # d/dp_j arccos(dot(g_i, p_j)) = -g_i / sqrt(1 - dot^2)
        # But we need the gradient of the soft-max of min-distances.
        # This is complex. Let's use numerical gradient for now.
        value = max_d + np.log(np.sum(exp_vals)) / beta
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(pts)
        for i in range(N):
            for d in range(3):
                p_plus = pts.copy()
                p_plus[i, d] += eps
                p_plus = normalize(p_plus)
                f_plus = soft_covering_radius(p_plus, beta)
                
                p_minus = pts.copy()
                p_minus[i, d] -= eps
                p_minus = normalize(p_minus)
                f_minus = soft_covering_radius(p_minus, beta)
                
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        return value, grad

    def lbfgs_optimize(points, beta=100.0, max_iter=20):
        """
        Use L-BFGS-B to optimize points on the sphere.
        """
        if time.time() > deadline - 15.0:
            return points
        
        x0 = points.flatten()
        
        def objective(x):
            if time.time() > deadline - 5.0:
                return 1e10
            pts = x.reshape(N, 3)
            return soft_covering_radius(pts, beta)
        
        # Use numerical gradient via scipy
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            if result.success or result.fun < objective(x0):
                new_pts = result.x.reshape(N, 3)
                new_pts = normalize(new_pts)
                return new_pts
        except Exception:
            pass
        return points

    def compute_exact_covering_radius(pts):
        """
        Compute covering radius using convex hull facets.
        """
        pts = normalize(pts)
        try:
            hull = ConvexHull(pts)
            # For each facet, compute the circumradius
            # The covering radius is the max over all points on the sphere
            # of the distance to the nearest point.
            # This equals the max circumradius of the facets of the convex hull
            # (for the dual problem on the sphere).
            
            # A simpler approach: use a very dense grid
            n_lat, n_lon = 100, 200
            lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
            lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            
            dots = np.dot(grid, pts.T)
            dots = np.clip(dots, -1.0, 1.0)
            ang_dists = np.arccos(dots)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))
        except Exception:
            # Fallback to grid
            n_lat, n_lon = 100, 200
            lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
            lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            
            dots = np.dot(grid, pts.T)
            dots = np.clip(dots, -1.0, 1.0)
            ang_dists = np.arccos(dots)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple restarts with Lloyd + L-BFGS
    n_restarts = 6
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = random_points(N, seed=42)
        elif restart == 2:
            points = random_points(N, seed=123)
        elif restart == 3:
            points = random_points(N, seed=456)
        elif restart == 4:
            if best_points is not None:
                rng = np.random.RandomState(789)
                points = best_points + 0.05 * rng.randn(N, 3)
                points = normalize(points)
            else:
                points = fibonacci_spiral(N).astype(np.float64)
        else:
            if best_points is not None:
                rng = np.random.RandomState(1024)
                points = best_points + 0.02 * rng.randn(N, 3)
                points = normalize(points)
            else:
                points = fibonacci_spiral(N).astype(np.float64)

        # Apply Lloyd's algorithm to get a better starting point
        candidate = lloyd_step(points, n_iter=5)
        
        # Optimize using L-BFGS with increasing beta
        candidate = lbfgs_optimize(candidate, beta=50.0, max_iter=15)
        candidate = lbfgs_optimize(candidate, beta=100.0, max_iter=15)
        candidate = lbfgs_optimize(candidate, beta=200.0, max_iter=10)
        
        # Another Lloyd step
        candidate = lloyd_step(candidate, n_iter=3)
        
        rad = compute_exact_covering_radius(candidate)

        if rad < best_rad:
            best_rad = rad
            best_points = candidate.copy()
            print(f"Restart {restart}: Rad = {best_rad * 180 / np.pi:.6f} deg")

    # Phase 2: Final polishing
    if best_points is not None and time.time() < deadline - 5.0:
        final_candidate = lbfgs_optimize(best_points, beta=500.0, max_iter=5)
        rad_final = compute_exact_covering_radius(final_candidate)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final_candidate

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points
