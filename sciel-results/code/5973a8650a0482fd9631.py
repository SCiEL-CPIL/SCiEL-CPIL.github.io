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

    def objective_with_t(params, lambda_pen):
        """
        Objective function with auxiliary variable t.
        params: [x1, y1, ..., xN, yN, t]
        lambda_pen: penalty coefficient for constraint violations.
        """
        if time.time() > deadline:
            return 1e10

        pts = params[:2*N].reshape(-1, 2)
        t = params[2*N]

        if not np.all(np.isfinite(pts)):
            return 1e10
        if t < 0 or t > 0.5:
            return 1e10

        dists = np.linalg.norm(pts, axis=1)
        
        # Radial constraint: dists <= 1 - t
        radial_viol = np.maximum(0.0, dists - (1.0 - t))
        radial_penalty = np.sum(radial_viol ** 2)

        # Pairwise constraint: |ci - cj| >= 2t
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        pair_viol = np.maximum(0.0, 2.0 * t - pair_dists)
        pair_penalty = np.sum(pair_viol ** 2)

        # We want to maximize t, so minimize -t + penalties
        return -t + lambda_pen * (pair_penalty + radial_penalty)

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()
    best_points = pts.copy()

    # 2. Multi-stage optimization with auxiliary variable t
    # Start with small lambda (weak constraints, focus on t), increase lambda (strict constraints)
    lambdas = [1.0, 10.0, 100.0, 1000.0, 5000.0]

    current_pts = pts
    current_t = best_score  # Initial guess for t
    current_score = best_score

    for lam in lambdas:
        if time.time() > deadline - 10:
            break

        # Initial x0: flatten current points and append current t
        x0 = np.concatenate([current_pts.flatten(), [current_t]])
        
        # Bounds: x, y in [-1, 1], t in [0, 0.5]
        bounds = [(-1.0, 1.0)] * (2 * N) + [(0.0, 0.5)]

        try:
            # Increase iterations as we refine
            maxiter = 100
            maxfun = 1000

            result = minimize(
                objective_with_t, x0, args=(lam,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'maxfun': maxfun,
                         'ftol': 1e-12, 'gtol': 1e-10}
            )

            if result.fun < 1e9:
                new_points = result.x[:2*N].reshape(-1, 2)
                new_t = result.x[2*N]
                
                # Compute actual score for the new points
                new_score = compute_score(new_points)
                
                # Update best if improved
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    current_pts = new_points.copy()
                    current_t = new_score  # Use actual score as next t guess
                    current_score = new_score
                else:
                    # Even if score didn't improve, we might have a better t estimate
                    # for the next iteration if the points are valid
                    if np.all(np.isfinite(new_points)):
                         current_pts = new_points.copy()
                         # Only update t if it's consistent with the points
                         if new_score > 0:
                             current_t = new_score
        except Exception:
            pass

    # 3. Final high-precision polish with high lambda
    if time.time() < deadline - 5:
        points = best_points.copy()
        t_guess = best_score
        
        # Use very high lambda to enforce constraints strictly
        for lam in [10000.0, 50000.0]:
            if time.time() > deadline - 2:
                break
            x0 = np.concatenate([points.flatten(), [t_guess]])
            bounds = [(-1.0, 1.0)] * (2 * N) + [(0.0, 0.5)]
            try:
                result = minimize(
                    objective_with_t, x0, args=(lam,),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 500,
                             'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < 1e9:
                    new_points = result.x[:2*N].reshape(-1, 2)
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        points = new_points.copy()
                        t_guess = new_score
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
