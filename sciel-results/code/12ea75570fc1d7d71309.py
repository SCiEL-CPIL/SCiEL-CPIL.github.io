"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        # Radial distances
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        # Pairwise distances using upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle."""
        best_pts = None
        best_score = -1.0

        # Try a few spacings to ensure we get enough points
        for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
            spacing = 2.0 / (np.sqrt(N) * scale)

            # Generate grid points
            nx = int(2.0 / spacing) + 3
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3

            # Center the grid
            x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0

            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])

            if len(pts_list) < N:
                continue

            pts = np.array(pts_list)
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]

            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts

        if best_pts is None:
            # Fallback: random
            rng = np.random.RandomState(42)
            best_pts = rng.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_score(best_pts)

        return best_pts, best_score

    def smooth_objective_t(params, pts_init, beta, penalty_weight):
        """
        Epigraph formulation with auxiliary variable t.
        Objective: minimize -t + penalty * sum of violations.
        
        Constraints (smoothed):
          1. For all pairs i<j: |c_i - c_j| >= 2*t
          2. For all i: |c_i| + t <= 1
        
        We use smooth penalties: for each constraint g_k <= 0,
        add penalty_weight * max(0, g_k)^2 / (2*beta)
        
        This allows simultaneous movement of all constraints.
        """
        if time.time() > deadline:
            return 1e10

        # params: [x_0, y_0, ..., x_{N-1}, y_{N-1}, t]
        pts = params[:2*N].reshape(-1, 2)
        t = params[2*N]

        if not np.all(np.isfinite(pts)) or not np.isfinite(t):
            return 1e10

        # Radial constraints: |c_i| + t <= 1
        dists = np.linalg.norm(pts, axis=1)
        radial_violations = np.maximum(dists + t - 1.0, 0.0)
        radial_penalty = penalty_weight * np.sum(radial_violations ** 2) / (2.0 * beta)

        # Pairwise constraints: |c_i - c_j| >= 2*t
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        pair_violations = np.maximum(2.0 * t - pair_dists, 0.0)
        pair_penalty = penalty_weight * np.sum(pair_violations ** 2) / (2.0 * beta)

        # Also add a small penalty for t being too large (shouldn't happen but for safety)
        t_penalty = penalty_weight * max(0.0, t - 0.5) ** 2 / (2.0 * beta)

        return -t + radial_penalty + pair_penalty + t_penalty

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()
    best_points = pts.copy()

    # 2. Multi-stage optimization with increasing beta using epigraph formulation
    # Start with low beta for global movement, increase for local precision.
    betas = [10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
    penalty_weights = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]

    current_pts = pts
    current_score = best_score

    for beta, pw in zip(betas, penalty_weights):
        if time.time() > deadline - 10:
            break

        # Initialize t with current score estimate
        t_init = best_score
        x0 = np.concatenate([current_pts.flatten(), [t_init]])
        
        # Bounds: coords in [-0.999, 0.999], t in [0.001, 0.5]
        bounds = [(-0.999, 0.999)] * (2 * N) + [(0.001, 0.5)]

        try:
            # Use more iterations for lower beta (exploration), fewer for high beta (polish)
            if beta < 100:
                maxiter = 60
                maxfun = 600
            elif beta < 500:
                maxiter = 40
                maxfun = 400
            else:
                maxiter = 30
                maxfun = 300

            result = minimize(
                smooth_objective_t, x0, args=(current_pts, beta, pw),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'maxfun': maxfun,
                         'ftol': 1e-12, 'gtol': 1e-10}
            )

            if result.fun < 1e9:
                new_points = result.x[:2*N].reshape(-1, 2)
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    current_pts = new_points.copy()
                    current_score = new_score
        except Exception:
            pass

    # 3. Try ring-based initialization as an alternative starting point
    if time.time() < deadline - 60:
        best_ring_points = None
        best_ring_score = best_score

        for n_rings in range(8, 20):
            if time.time() > deadline - 15:
                break

            ring_radii = np.linspace(0.05, 0.95, n_rings)
            ring_points = []

            for r_ring in ring_radii:
                est_r = 0.05
                n_circ = max(1, int(2 * np.pi * r_ring / (2 * est_r)))
                angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
                for a in angles:
                    x = r_ring * np.cos(a)
                    y = r_ring * np.sin(a)
                    ring_points.append([x, y])

            ring_points.append([0.0, 0.0])
            ring_points = np.array(ring_points)

            if len(ring_points) > N:
                dists = np.linalg.norm(ring_points, axis=1)
                idx = np.argsort(dists)[:N]
                ring_points = ring_points[idx]
            elif len(ring_points) < N:
                continue

            score = compute_score(ring_points)
            if score > best_ring_score:
                best_ring_score = score
                best_ring_points = ring_points.copy()

        if best_ring_points is not None and best_ring_score > best_score:
            best_points = best_ring_points
            best_score = best_ring_score

            # Optimize the ring-based points too
            current_pts = best_points.copy()
            for beta, pw in [(50.0, 5.0), (200.0, 50.0), (1000.0, 500.0)]:
                if time.time() > deadline - 10:
                    break
                t_init = best_score
                x0 = np.concatenate([current_pts.flatten(), [t_init]])
                bounds = [(-0.999, 0.999)] * (2 * N) + [(0.001, 0.5)]
                try:
                    result = minimize(
                        smooth_objective_t, x0, args=(current_pts, beta, pw),
                        method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 25, 'maxfun': 250,
                                 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if result.fun < 1e9:
                        new_points = result.x[:2*N].reshape(-1, 2)
                        new_score = compute_score(new_points)
                        if new_score > best_score:
                            best_score = new_score
                            best_points = new_points.copy()
                            current_pts = new_points.copy()
                except Exception:
                    pass

    # 4. Final high-precision polish
    if time.time() < deadline - 5:
        points = best_points.copy()
        for beta, pw in [(5000.0, 2000.0), (10000.0, 5000.0)]:
            if time.time() > deadline - 3:
                break
            t_init = best_score
            x0 = np.concatenate([points.flatten(), [t_init]])
            bounds = [(-0.999, 0.999)] * (2 * N) + [(0.001, 0.5)]
            try:
                result = minimize(
                    smooth_objective_t, x0, args=(points, beta, pw),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 40, 'maxfun': 400,
                             'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < 1e9:
                    new_points = result.x[:2*N].reshape(-1, 2)
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        points = new_points.copy()
            except Exception:
                pass

    # Final validation
    final_score = compute_score(best_points)
    if final_score <= 0:
        # Fallback to hexagonal
        pts, _ = generate_hexagonal_initialization()
        return pts

    return best_points
# EVOLVE-BLOCK-END
