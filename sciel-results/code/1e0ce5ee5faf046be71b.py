"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

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

    # Precompute a dense spherical grid for smooth covering radius
    n_lat = 40
    n_lon = 80
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    grid_z = np.sin(LA)
    grid_x = np.cos(LA) * np.cos(LO)
    grid_y = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    n_grid = len(grid_pts)

    def compute_covering_radius_hull(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
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
            return compute_covering_radius_grid(pts)

    def compute_covering_radius_grid(pts):
        """Compute covering radius using dense grid (smooth approximation)."""
        dot = np.dot(grid_pts, pts.T)  # (n_grid, N)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # (n_grid, N)
        min_dists = np.min(ang_dists, axis=1)
        return float(np.max(min_dists))

    def soft_max_objective(pts_flat, beta=80.0):
        """
        Smooth surrogate: soft-max of angular distances over the grid.
        pts_flat is a flat array of shape (3*N,).
        Returns (objective_value, gradient_flat).
        """
        pts = pts_flat.reshape(N, 3)
        # Re-normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms

        # Compute angular distances: (n_grid, N)
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        ang_dists = np.arccos(dot)  # (n_grid, N)

        # For each grid point, the nearest point distance
        min_dists = np.min(ang_dists, axis=1)  # (n_grid,)

        # Soft-max over grid points
        m = np.max(min_dists)
        if m < 1e-15:
            return 0.0, np.zeros(3 * N)
        shifted = min_dists - m
        exp_vals = np.exp(beta * shifted)
        sum_exp = np.sum(exp_vals)
        obj = m + np.log(sum_exp) / beta

        # Gradient: d(obj)/d(pts_flat)
        weights = exp_vals / sum_exp  # (n_grid,)

        # For each grid point j, d_j = min_i arccos(dot(grid_j, pts_i))
        nearest_idx = np.argmin(ang_dists, axis=1)  # (n_grid,)

        grad = np.zeros_like(pts)  # (N, 3)
        for j in range(n_grid):
            w = weights[j]
            if w < 1e-10:
                continue
            i = nearest_idx[j]
            g = grid_pts[j]
            p = pts[i]
            u = np.dot(g, p)
            u = np.clip(u, -1.0 + 1e-12, 1.0 - 1e-12)
            g_perp = g - u * p
            denom = np.sqrt(1.0 - u * u)
            if denom < 1e-12:
                continue
            grad[i] += w * (-g_perp / denom)

        # Project gradient to tangent space and scale for raw coordinates
        raw_norms = np.linalg.norm(pts_flat.reshape(N, 3), axis=1, keepdims=True)
        raw_norms[raw_norms < 1e-15] = 1.0

        grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_p * pts

        grad_flat = (grad_tan / raw_norms).ravel()

        return obj, grad_flat

    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps."""
        for _ in range(n_iter):
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points

    best_points = None
    best_rad = np.inf

    n_restarts = 5

    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            if best_points is None:
                points = fibonacci_spiral(N).astype(np.float64)
            else:
                points = best_points + 0.12 * np.random.randn(N, 3)
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points = points / norms

        # Phase 1: L-BFGS-B optimization on the smooth grid-based objective
        if time.time() < deadline - 15.0:
            x0 = points.ravel()
            beta = 80.0

            def obj_and_grad(x):
                if time.time() > deadline - 10.0:
                    return 1e6, np.zeros_like(x)
                val, grad = soft_max_objective(x, beta=beta)
                return val, grad

            try:
                result = minimize(
                    obj_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                             'maxfun': 200, 'disp': False}
                )
                opt_pts = result.x.reshape(N, 3)
                opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                opt_norms[opt_norms < 1e-15] = 1.0
                opt_pts = opt_pts / opt_norms

                rad_opt = compute_covering_radius_grid(opt_pts)
                if rad_opt < best_rad:
                    best_rad = rad_opt
                    best_points = opt_pts.copy()
            except Exception:
                pass

        # Phase 2: Repulsion-based refinement
        alpha = 0.02
        decay = 0.995
        max_iter = 100
        stall_count = 0

        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break

            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay

            if it % 5 == 0:
                rad = compute_covering_radius_grid(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 10:
                        alpha *= 0.9
                        stall_count = 0

            if alpha < 1e-8:
                break

        # Phase 3: Polish
        if time.time() < deadline - 8.0 and best_points is not None:
            polished = polish_step(best_points.copy(), 0.003, 30)
            rad_polished = compute_covering_radius_grid(polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = polished

    # Final global polish
    if time.time() < deadline - 5.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.001, 50)
        rad_final = compute_covering_radius_grid(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

        # One more L-BFGS-B pass with higher beta for sharper max
        if time.time() < deadline - 3.0:
            x0 = best_points.ravel()
            beta_final = 150.0

            def obj_and_grad_final(x):
                if time.time() > deadline - 1.0:
                    return 1e6, np.zeros_like(x)
                val, grad = soft_max_objective(x, beta=beta_final)
                return val, grad

            try:
                result = minimize(
                    obj_and_grad_final,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10,
                             'maxfun': 100, 'disp': False}
                )
                opt_pts = result.x.reshape(N, 3)
                opt_norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
                opt_norms[opt_norms < 1e-15] = 1.0
                opt_pts = opt_pts / opt_norms

                rad_opt = compute_covering_radius_grid(opt_pts)
                if rad_opt < best_rad:
                    best_rad = rad_opt
                    best_points = opt_pts.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
