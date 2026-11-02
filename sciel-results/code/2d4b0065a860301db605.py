import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _pressure_gradient(pts):
    """
    Compute gradient of convex hull volume w.r.t. vertices.
    For a convex polyhedron with facets, dV/dp_i = (1/3) * sum over facets
    containing i of (facet_area * facet_normal) projected to tangent plane.
    Minimizing volume = applying negative gradient (pressure relaxation).
    """
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
    except Exception:
        return None

    a = pts[facets[:, 0]]
    b = pts[facets[:, 1]]
    c = pts[facets[:, 2]]

    # Facet normal (unnormalized) = 2 * cross product
    n = np.cross(a - b, a - c)  # (F, 3), magnitude = 2*area
    # Ensure outward orientation
    center = pts.mean(axis=0)
    dots = np.sum(n * (a - center), axis=1)
    n[dots < 0] *= -1

    # Gradient contribution: for each facet, each vertex gets (1/3)*normal
    grad = np.zeros_like(pts)
    for idx in range(3):
        np.add.at(grad, facets[:, idx], n / 3.0)

    # Project to tangent plane
    dot_gp = np.sum(grad * pts, axis=1)
    grad_tan = grad - dot_gp[:, np.newaxis] * pts

    return grad_tan

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin

    # Create a dense spherical grid for smooth surrogate objective
    # Fibonacci spiral with ~2000 points
    M = 2000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_pts = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    grid_pts = grid_pts / np.linalg.norm(grid_pts, axis=1, keepdims=True)

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        # Vectorized circumcenter computation
        a = p[facets[:, 0]]  # (F, 3)
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]

        v1 = a - b  # (F, 3)
        v2 = b - c
        n = np.cross(v1, v2)  # (F, 3)
        norm_n = np.linalg.norm(n, axis=1)  # (F,)

        # Handle degenerate facets
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc

            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def grid_soft_max_objective(params, beta=50.0):
        """
        Soft-max surrogate based on grid points.
        Objective: (1/beta) * log(sum(exp(beta * min_dist)))
        where min_dist is the angular distance from grid point to nearest P point.
        """
        X = normalize(params.reshape(N, 3))
        
        # Compute angular distances from grid points to all P points
        # cos_dist = grid_pts @ X.T  -> (M, N)
        cos_dist = np.dot(grid_pts, X.T)
        cos_dist = np.clip(cos_dist, -1.0, 1.0)
        
        # Angular distance
        dists = np.arccos(cos_dist) # (M, N)
        
        # Min distance for each grid point
        min_dists = np.min(dists, axis=1) # (M,)
        
        # Soft max
        r_max = np.max(min_dists)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (min_dists - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def optimize_grid(pts_init, beta, maxiter=100):
        """Run L-BFGS-B optimization on the grid-based soft-max surrogate."""
        best_pts_local = pts_init.copy()
        best_radius_local = get_max_radius(pts_init)

        pts_init = normalize(pts_init.copy())

        try:
            res = minimize(lambda p: grid_soft_max_objective(p, beta), pts_init.flatten(),
                           method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200})

            if res.fun < grid_soft_max_objective(pts_init.flatten(), beta):
                new_pts = normalize(res.x.reshape(N, 3))
                # Validate with exact hull radius
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius_local:
                    best_pts_local = new_pts
                    best_radius_local = current_radius
        except Exception:
            pass

        return best_pts_local, best_radius_local

    def repulsion_relax(pts, n_iters=600, alpha_start=0.08):
        """
        Repulsion relaxation with adaptive step size (backtracking line search).
        Minimize sum 1/||p_i - p_j||^2 via gradient descent on sphere.
        """
        pts = pts.copy()
        alpha = alpha_start

        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)

        current_energy = repulsion_energy(pts)

        for it in range(n_iters):
            # Calculate pairwise differences
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)

            # Avoid division by zero
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)

            # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

            # Project onto tangent plane: F_tan = F - (F . p) p
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts

            # Adaptive step size with backtracking
            step = alpha
            improved = False
            for _ in range(10):
                new_pts = normalize(pts + step * forces_tan)
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5

            if not improved:
                alpha *= 0.8
            else:
                alpha *= 1.01

            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)

            if time.time() > deadline - 5.0:
                break

        return pts

    def lloyd_iteration(pts, n_iters=60):
        """
        Improved Lloyd's algorithm: move each point toward the circumcenter
        of its worst facet, with adaptive weighting.
        """
        pts = pts.copy()

        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break

            # For each point, accumulate weighted pull toward facet circumcenters
            updates = np.zeros_like(pts)
            counts = np.zeros(N)

            for f in range(len(facets)):
                facet_pts = facets[f]
                cc = circumcenters[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                # Weight by how much this facet contributes to the max
                weight = (rad / max_rad) ** 2
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight

            # Apply updates where counts > 0
            mask = counts > 1e-10
            if np.any(mask):
                step_size = 0.05 * (max_rad / (np.mean(radii) + 1e-15))
                step_size = min(step_size, 0.1)  # Cap step size
                pts[mask] += step_size * updates[mask] / counts[mask, np.newaxis]

            # Normalize
            pts = normalize(pts)

            if time.time() > deadline - 5.0:
                break

        return pts

    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = pts.copy()
    best_radius = get_max_radius(pts)

    # Phase 0: initial repulsion relaxation to spread points
    if time.time() < deadline - 20.0:
        relaxed = repulsion_relax(pts, n_iters=300)
        rad = get_max_radius(relaxed)
        if rad < best_radius:
            best_radius = rad
            best_pts = relaxed.copy()
            pts = relaxed.copy()

    # Phase 1: Grid-based soft-max optimization with annealing
    # Use grid surrogate for speed, validate with hull
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0]

    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break

        # Lloyd iteration to equalize radii (uses exact hull)
        lloyded = lloyd_iteration(pts, n_iters=40)
        rad = get_max_radius(lloyded)
        if rad < best_radius:
            best_radius = rad
            best_pts = lloyded.copy()
            pts = lloyded.copy()

        # L-BFGS polish on grid-based soft-max surrogate
        polished, pol_rad = optimize_grid(pts, beta, maxiter=100)
        if pol_rad < best_radius:
            best_radius = pol_rad
            best_pts = polished.copy()
            pts = polished.copy()

    # Phase 2: Final high-precision polish using hull-based Lloyd + Grid L-BFGS
    if time.time() < deadline - 5.0:
        # One more Lloyd pass
        lloyded = lloyd_iteration(pts, n_iters=50)
        rad = get_max_radius(lloyded)
        if rad < best_radius:
            best_radius = rad
            best_pts = lloyded.copy()
            pts = lloyded.copy()

        # Final grid polish with high beta
        polished, pol_rad = optimize_grid(pts, 800.0, maxiter=100)
        if pol_rad < best_radius:
            best_radius = pol_rad
            best_pts = polished.copy()

    # Final validation: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_pts
