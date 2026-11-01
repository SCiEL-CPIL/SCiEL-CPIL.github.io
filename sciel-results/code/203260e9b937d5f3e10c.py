"""Program for covering the sphere with 130 equal caps using soft-max optimization with analytical gradients."""
import numpy as np
import time
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s hard limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Grids: coarse for fast Lloyd, medium for intermediate, fine for final
    grid_coarse = generate_grid(50, 100)
    grid_med = generate_grid(80, 160)
    grid_fine = generate_grid(120, 240)
    grid_very_fine = generate_grid(160, 320)

    def covering_radius(pts, grid):
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def soft_covering_radius_and_grad(pts, beta, grid):
        """
        Compute soft covering radius and its gradient w.r.t. pts.
        Uses log-sum-exp over the min-distance per grid point.
        """
        # dots[j, i] = grid[j] . pts[i]
        dots = np.dot(grid, pts.T)  # shape (M, N)
        dots_clipped = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        ang = np.arccos(dots_clipped)  # shape (M, N)
        
        # min over points for each grid point
        min_d = np.min(ang, axis=1)  # shape (M,)
        min_idx = np.argmin(ang, axis=1)  # shape (M,)
        
        # Soft max via log-sum-exp
        m = np.max(min_d)
        shifted = beta * (min_d - m)
        exp_shifted = np.exp(shifted)
        log_sum_exp = np.log(np.sum(exp_shifted))
        soft_val = m + log_sum_exp / beta
        
        # Gradient: d(soft_val)/d(pts[i])
        # soft_val = m + (1/beta) * log(sum_j exp(beta * (min_d[j] - m)))
        # d(soft_val)/d(min_d[j]) = exp(beta*(min_d[j]-m)) / sum_k exp(beta*(min_d[k]-m))
        # d(min_d[j])/d(pts[i]) = -d(ang[j,i])/d(pts[i]) if pts[i] is the nearest to grid[j]
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        # d(dots[j,i])/d(pts[i]) = grid[j]
        # So d(ang[j,i])/d(pts[i]) = -grid[j] / sqrt(1 - dots[j,i]^2)
        
        weights = exp_shifted / np.sum(exp_shifted)  # shape (M,)
        
        grad = np.zeros_like(pts)
        for j in range(len(min_d)):
            w = weights[j]
            if w < 1e-15:
                continue
            i = min_idx[j]
            # d(soft_val)/d(pts[i]) = w * d(min_d[j])/d(pts[i])
            # min_d[j] = ang[j, i_nearest]
            # d(ang[j,i])/d(pts[i]) = -grid[j] / sqrt(1 - dots[j,i]^2)
            d = dots_clipped[j, i]
            denom = np.sqrt(max(1.0 - d * d, 1e-12))
            grad[i] += w * (-grid[j] / denom) * (-1.0)  # chain rule signs
        
        # Project gradient onto tangent space (since pts are constrained to sphere)
        for i in range(N):
            p = pts[i]
            g = grad[i]
            grad[i] = g - np.dot(g, p) * p
        
        return soft_val, grad

    def lloyd_step(pts, grid, alpha=0.5):
        """
        Voronoi-based Lloyd step: move each point toward the circumcenter
        of the Voronoi cell (approximated by the farthest grid points in its cell).
        """
        dots = np.dot(grid, pts.T)
        nearest = np.argmax(dots, axis=1)
        
        new_pts = pts.copy()
        for i in range(N):
            mask = (nearest == i)
            n_in_cell = np.sum(mask)
            if n_in_cell < 3:
                continue
            
            cell_pts = grid[mask]
            # Find the 3 farthest points from pts[i] in the cell
            d = np.dot(pts[i], cell_pts.T)
            d = np.clip(d, -1.0, 1.0)
            ang = np.arccos(d)
            
            # Take top 3 farthest
            top_k = min(3, n_in_cell)
            far_idx = np.argsort(-ang)[:top_k]
            far_pts = cell_pts[far_idx]
            
            # Compute circumcenter of these points on the sphere
            # Use the formula: circumcenter is the normalized sum of (far_pts - centroid) projections
            # Simpler: use the point equidistant from all far_pts
            # For 2 points: midpoint normalized
            # For 3 points: solve for point equidistant
            if top_k == 2:
                cc = far_pts[0] + far_pts[1]
            elif top_k == 3:
                # Solve for point x on sphere equidistant from 3 points
                # |x - p1|^2 = |x - p2|^2 => x.(p2-p1) = (|p2|^2 - |p1|^2)/2 = 0
                # So x is in the plane through origin perpendicular to (p2-p1) and (p3-p1)
                # x = a*(p2-p1) x (p3-p1) normalized... actually x is along the cross product
                v1 = far_pts[1] - far_pts[0]
                v2 = far_pts[2] - far_pts[0]
                cc = np.cross(v1, v2)
                # Check if cc points toward the cell (dot with pts[i] should be positive)
                if np.dot(cc, pts[i]) < 0:
                    cc = -cc
            else:
                continue
            
            nrm = np.linalg.norm(cc)
            if nrm < 1e-10:
                continue
            cc = cc / nrm
            
            # Move toward circumcenter
            new_pts[i] = (1 - alpha) * pts[i] + alpha * cc
            new_pts[i] /= np.linalg.norm(new_pts[i])
        
        return new_pts

    def optimize_smooth(pts, beta, grid, max_iter=100, tol=1e-10):
        if time.time() > deadline - 10.0:
            return pts
        
        x0 = pts.flatten()
        
        def obj_grad(x):
            if time.time() > deadline - 5.0:
                return 1e10, np.zeros_like(x)
            p = x.reshape(N, 3)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            val, grad = soft_covering_radius_and_grad(p, beta, grid)
            return val, grad.flatten()
        
        try:
            res = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50,
                                    'maxfun': max_iter * 10})
            if res.fun < obj_grad(x0)[0]:
                p = res.x.reshape(N, 3)
                p /= np.linalg.norm(p, axis=1, keepdims=True)
                return p
        except Exception:
            pass
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 12
    
    for r in range(n_restarts):
        if time.time() > deadline - 35.0:
            break
            
        # Initialization
        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 4 and best_pts is not None:
            rng = np.random.RandomState(1000 + r)
            scale = 0.03 * (1.0 - r / (n_restarts + 1.0))
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)

        # Stage 1: Coarse Lloyd's iterations
        for i in range(15):
            if time.time() > deadline - 30.0:
                break
            pts = lloyd_step(pts, grid_coarse, alpha=0.6)
            
        # Stage 2: Medium grid Lloyd's
        for i in range(10):
            if time.time() > deadline - 25.0:
                break
            pts = lloyd_step(pts, grid_med, alpha=0.5)

        # Stage 3: Smooth optimization on Medium grid with analytical gradient
        pts = optimize_smooth(pts, beta=30, grid=grid_med, max_iter=50, tol=1e-9)
        
        # Stage 4: Fine grid Lloyd's
        for i in range(8):
            if time.time() > deadline - 15.0:
                break
            pts = lloyd_step(pts, grid_fine, alpha=0.4)

        # Stage 5: Smooth optimization on Fine grid
        pts = optimize_smooth(pts, beta=60, grid=grid_fine, max_iter=40, tol=1e-10)
        
        # Stage 6: Very fine grid polish
        pts = optimize_smooth(pts, beta=100, grid=grid_very_fine, max_iter=30, tol=1e-11)
        
        rad = covering_radius(pts, grid_very_fine)
        
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()

    # Final polish if time permits
    if best_pts is not None and time.time() < deadline - 8.0:
        for i in range(3):
            if time.time() > deadline - 3.0:
                break
            best_pts = lloyd_step(best_pts, grid_very_fine, alpha=0.3)
            best_pts = optimize_smooth(best_pts, beta=150, grid=grid_very_fine, max_iter=20, tol=1e-12)
            
        rad = covering_radius(best_pts, grid_very_fine)
        if rad < best_rad:
            best_rad = rad

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
