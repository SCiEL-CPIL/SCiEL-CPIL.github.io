import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50


def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses a BCC lattice initialization followed by multi-start local optimization
    to maximize the minimum distance between centers and to the cube boundaries.
    """
    np.random.seed(42)

    # Generate BCC lattice points
    # BCC unit cell: corner points + body center
    # We'll generate a grid of BCC cells and select 50 points

    def generate_bcc_points(n_per_side, offset=0.0):
        """Generate BCC points in a grid within the cube."""
        a = 1.0 / n_per_side  # unit cell size
        points = []
        for i in range(n_per_side):
            for j in range(n_per_side):
                for k in range(n_per_side):
                    # Corner point
                    cx = (i + offset) * a
                    cy = (j + offset) * a
                    cz = (k + offset) * a
                    # Shift to center in [-0.5, 0.5]
                    cx = cx - 0.5
                    cy = cy - 0.5
                    cz = cz - 0.5
                    if abs(cx) <= 0.5 and abs(cy) <= 0.5 and abs(cz) <= 0.5:
                        points.append([cx, cy, cz])

                    # Body center
                    bx = (i + 0.5) * a - 0.5
                    by = (j + 0.5) * a - 0.5
                    bz = (k + 0.5) * a - 0.5
                    if abs(bx) <= 0.5 and abs(by) <= 0.5 and abs(bz) <= 0.5:
                        points.append([bx, by, bz])

        return np.array(points)

    # Try different grid sizes to get close to 50 points
    best_init = None
    best_init_radius = -np.inf

    for n_per_side in range(3, 7):
        for offset in [0.0, 0.1, 0.2, 0.3, 0.4]:
            pts = generate_bcc_points(n_per_side, offset)
            if len(pts) < N:
                # Pad with random points
                while len(pts) < N:
                    pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, 3)])
            elif len(pts) > N:
                # Select N points using farthest-point sampling
                selected = [np.random.choice(len(pts))]
                while len(selected) < N:
                    unselected = [i for i in range(len(pts)) if i not in selected]
                    best_idx = -1
                    best_min_dist = -1
                    for idx in unselected:
                        dists = np.linalg.norm(pts[idx] - pts[selected], axis=1)
                        min_d = np.min(dists)
                        if min_d > best_min_dist:
                            best_min_dist = min_d
                            best_idx = idx
                    selected.append(best_idx)
                pts = pts[selected]

            pts = pts[:N]
            pts = np.clip(pts, -0.49, 0.49)

            # Evaluate initial radius
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d)
            dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
            min_bound = np.min(dist_to_bound)
            r = 0.5 * min(min_pair, min_bound)

            if r > best_init_radius:
                best_init_radius = r
                best_init = pts.copy()

    if best_init is None:
        # Fallback to random
        best_init = np.random.uniform(-0.45, 0.45, (N, 3))

    # Objective function: maximize minimum radius
    def objective(x):
        pts = x.reshape(N, 3)

        # Boundary penalty
        if np.any(np.abs(pts) > 0.5 + 1e-9):
            return 1e10

        # Pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)

        # Boundary distances
        dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
        min_bound_dist = np.min(dist_to_bound)

        r = 0.5 * min(min_pair_dist, min_bound_dist)
        return -r

    x0 = best_init.flatten()

    # Multi-start optimization
    best_x = x0.copy()
    best_obj = objective(x0)

    n_restarts = 15
    for restart in range(n_restarts):
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb the initial solution
            perturbation_scale = 0.02 * (restart + 1)
            x_init = x0 + np.random.normal(0, perturbation_scale, N * 3)
            x_init = np.clip(x_init.reshape(N, 3), -0.495, 0.495).flatten()

        try:
            result = minimize(
                objective,
                x_init,
                method='Powell',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )

            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

        # Also try Nelder-Mead for some restarts
        if restart % 3 == 1:
            try:
                result_nm = minimize(
                    objective,
                    x_init,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )

                if result_nm.fun < best_obj:
                    best_obj = result_nm.fun
                    best_x = result_nm.x.copy()
            except Exception:
                pass

    # Final result
    result_centers = best_x.reshape(N, 3)
    result_centers = np.clip(result_centers, -0.499999, 0.499999)

    return result_centers
