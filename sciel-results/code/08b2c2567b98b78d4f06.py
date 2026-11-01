"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    def smooth_objective_with_grad(params, beta):
        """
        Smooth objective for L-BFGS-B with analytic gradient.
        Returns (negative soft-min radius, gradient).
        
        Uses a differentiable soft-min over all pairwise distances and radial constraints.
        """
        pts = params.reshape(-1, 2)

        if not np.all(np.isfinite(pts)):
            return 1e10, np.zeros(2 * N)

        # Compute all pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.sqrt(np.sum(diffs ** 2, axis=1) + 1e-12)
        
        # Radial distances
        dists = np.sqrt(np.sum(pts ** 2, axis=1) + 1e-12)
        radial_margins = 1.0 - dists

        # Soft-min over pairwise distances (divided by 2) and radial margins
        # We want to maximize min(min_pair/2, min_radial)
        # Use log-sum-exp for soft-min
        
        # For numerical stability, shift by the minimum
        pair_vals = pair_dists / 2.0
        radial_vals = radial_margins
        
        # Combine all values into one array
        all_vals = np.concatenate([pair_vals, radial_vals])
        n_pairs = len(pair_vals)
        n_radial = len(radial_vals)
        
        # Soft-min: -1/beta * log(sum(exp(-beta * x)))
        # For stability, subtract min
        min_val = np.min(all_vals)
        shifted = -beta * (all_vals - min_val)
        
        # Clip for numerical stability
        shifted = np.clip(shifted, -700, 700)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        soft_min_val = -1.0 / beta * (np.log(sum_exp) - beta * min_val)
        
        # Objective is negative of soft-min
        obj = -soft_min_val
        
        # Gradient computation
        # d(soft_min)/d(all_vals_k) = -exp(-beta * (all_vals_k - min_val)) / sum(exp(...))
        # But we need chain rule through the soft-min
        
        weights = exp_shifted / sum_exp  # These are the softmax weights
        
        # Gradient with respect to pair distances
        grad_pair = -weights[:n_pairs] / beta
        
        # Gradient with respect to radial distances  
        grad_radial = -weights[n_pairs:] / beta
        
        # Chain rule: pair_dist_k = |c_i - c_j|
        # d(pair_dist_k)/d(c_i) = (c_i - c_j) / |c_i - c_j|
        # d(pair_dist_k)/d(c_j) = (c_j - c_i) / |c_i - c_j|
        
        grad_pts = np.zeros((N, 2))
        
        # Accumulate gradients from pairwise terms
        for k in range(n_pairs):
            i, j = i_idx[k], j_idx[k]
            d = diffs[k]
            dist = pair_dists[k]
            if dist > 1e-10:
                unit_vec = d / dist
                # grad_pair[k] is d(obj)/d(pair_dist_k)
                # d(pair_dist_k)/d(c_i) = unit_vec
                # d(pair_dist_k)/d(c_j) = -unit_vec
                # But we also have the /2 factor in pair_vals
                factor = grad_pair[k] * 0.5
                grad_pts[i] += factor * unit_vec
                grad_pts[j] -= factor * unit_vec
        
        # Chain rule: radial_margin_i = 1 - |c_i|
        # d(radial_margin_i)/d(c_i) = -c_i / |c_i|
        for i in range(N):
            dist = dists[i]
            if dist > 1e-10:
                unit_vec = pts[i] / dist
                grad_pts[i] -= grad_radial[i] * unit_vec
        
        # Objective gradient is negative of soft-min gradient
        grad_params = -grad_pts.flatten()
        
        return obj, grad_params

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()
    best_points = pts.copy()

    # 2. Multi-stage optimization with increasing beta using analytic gradients
    betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]

    current_pts = pts
    current_score = best_score

    for beta in betas:
        if time.time() > deadline - 15:
            break

        x0 = current_pts.flatten()
        bounds = [(-0.999, 0.999)] * (2 * N)

        try:
            # Use more iterations for lower beta (exploration), fewer for high beta (polish)
            if beta < 100:
                maxiter = 80
                maxfun = 800
            elif beta < 500:
                maxiter = 50
                maxfun = 500
            else:
                maxiter = 30
                maxfun = 300

            result = minimize(
                smooth_objective_with_grad, x0, args=(beta,),
                method='L-BFGS-B', bounds=bounds,
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxfun,
                         'ftol': 1e-12, 'gtol': 1e-10,
                         'maxls': 50}
            )

            if result.fun < 1e9:
                new_points = result.x.reshape(-1, 2)
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    current_pts = new_points.copy()
                    current_score = new_score
        except Exception:
            pass

    # 3. Try ring-based initialization as an alternative starting point
    if time.time() < deadline - 40:
        best_ring_points = None
        best_ring_score = best_score

        for n_rings in range(8, 20):
            if time.time() > deadline - 20:
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
            for beta in [50.0, 200.0, 1000.0]:
                if time.time() > deadline - 15:
                    break
                x0 = current_pts.flatten()
                bounds = [(-0.999, 0.999)] * (2 * N)
                try:
                    result = minimize(
                        smooth_objective_with_grad, x0, args=(beta,),
                        method='L-BFGS-B', bounds=bounds,
                        jac=True,
                        options={'maxiter': 30, 'maxfun': 300,
                                 'ftol': 1e-12, 'gtol': 1e-10,
                                 'maxls': 50}
                    )
                    if result.fun < 1e9:
                        new_points = result.x.reshape(-1, 2)
                        new_score = compute_score(new_points)
                        if new_score > best_score:
                            best_score = new_score
                            best_points = new_points.copy()
                            current_pts = new_points.copy()
                except Exception:
                    pass

    # 4. Final high-precision polish with very high beta
    if time.time() < deadline - 8:
        points = best_points.copy()
        for beta in [3000.0, 5000.0, 10000.0]:
            if time.time() > deadline - 5:
                break
            x0 = points.flatten()
            bounds = [(-0.999, 0.999)] * (2 * N)
            try:
                result = minimize(
                    smooth_objective_with_grad, x0, args=(beta,),
                    method='L-BFGS-B', bounds=bounds,
                    jac=True,
                    options={'maxiter': 50, 'maxfun': 500,
                             'ftol': 1e-14, 'gtol': 1e-12,
                             'maxls': 50}
                )
                if result.fun < 1e9:
                    new_points = result.x.reshape(-1, 2)
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        points = new_points.copy()
            except Exception:
                pass

    # 5. Local refinement: push boundary circles outward and interior circles apart
    if time.time() < deadline - 3:
        points = best_points.copy()
        # Try a simple local search: for each point, try small perturbations
        # and keep if it improves the score
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_score_local = compute_score(points)
            step = 0.001
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        test_pts = points.copy()
                        test_pts[i, 0] += dx
                        test_pts[i, 1] += dy
                        # Check bounds
                        if np.abs(test_pts[i, 0]) > 0.999 or np.abs(test_pts[i, 1]) > 0.999:
                            continue
                        test_score = compute_score(test_pts)
                        if test_score > current_score_local + 1e-12:
                            points = test_pts
                            current_score_local = test_score
                            improved = True
                            break
                    if improved:
                        break

        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()

    # Final validation
    final_score = compute_score(best_points)
    if final_score <= 0:
        # Fallback to hexagonal
        pts, _ = generate_hexagonal_initialization()
        return pts

    return best_points
# EVOLVE-BLOCK-END
