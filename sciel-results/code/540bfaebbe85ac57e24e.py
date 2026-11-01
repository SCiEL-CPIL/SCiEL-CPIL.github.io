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
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
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

        for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
            spacing = 2.0 / (np.sqrt(N) * scale)
            nx = int(2.0 / spacing) + 3
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3
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
            rng = np.random.RandomState(42)
            best_pts = rng.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_score(best_pts)

        return best_pts, best_score

    def generate_ring_initialization():
        """Generate ring-based initialization."""
        best_pts = None
        best_score = -1.0

        for n_rings in range(8, 22):
            if time.time() > deadline - 80:
                break

            ring_radii = np.linspace(0.03, 0.97, n_rings)
            ring_points = []

            for r_ring in ring_radii:
                est_r = 0.053
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
            if score > best_score:
                best_score = score
                best_pts = ring_points.copy()

        if best_pts is not None:
            return best_pts, best_score
        return None, -1.0

    def soft_min_obj_and_grad(params, beta, t_min=0.001):
        """
        Soft-min objective with auxiliary variable t.
        Objective: minimize -t + sum of soft penalty terms.
        
        Uses log-sum-exp for smooth min approximation.
        """
        if time.time() > deadline:
            return 1e10, np.zeros_like(params)

        pts = params[:2*N].reshape(-1, 2)
        t = params[2*N]

        if not np.all(np.isfinite(pts)) or not np.isfinite(t):
            return 1e10, np.zeros_like(params)

        # Radial constraints: |c_i| + t <= 1 -> violation = |c_i| + t - 1
        dists = np.linalg.norm(pts, axis=1)
        radial_violations = np.maximum(dists + t - 1.0, 0.0)

        # Pairwise constraints: |c_i - c_j| >= 2*t -> violation = 2*t - |c_i - c_j|
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        pair_violations = np.maximum(2.0 * t - pair_dists, 0.0)

        # Soft penalty using log-sum-exp for smoothness
        # sum of max(0, v)^2 / (2*beta) approximated smoothly
        def smooth_penalty(violations, beta):
            """Smooth approximation of sum(max(0,v)^2)/(2*beta)"""
            v = np.maximum(violations, 0.0)
            # Use softplus-like smoothing
            eps = 1e-8
            v_safe = np.maximum(v, eps)
            # Quadratic penalty with smooth clipping
            return np.sum(v ** 2) / (2.0 * beta)

        radial_penalty = smooth_penalty(radial_violations, beta)
        pair_penalty = smooth_penalty(pair_violations, beta)

        obj = -t + radial_penalty + pair_penalty

        # Gradient computation
        grad = np.zeros_like(params)

        # Gradient w.r.t. t
        grad[2*N] = -1.0 + np.sum(radial_violations) / beta + np.sum(pair_violations) / beta

        # Gradient w.r.t. points
        # Radial: d/dp_i (|p_i| + t - 1)^2 / (2*beta) = (|p_i| + t - 1) / beta * p_i / |p_i|
        for i in range(N):
            if dists[i] > 1e-10:
                radial_v = max(dists[i] + t - 1.0, 0.0)
                if radial_v > 0:
                    grad[2*i] += radial_v * pts[i, 0] / (beta * dists[i])
                    grad[2*i+1] += radial_v * pts[i, 1] / (beta * dists[i])

        # Pairwise: d/dp_i sum_j max(2t - |p_i-p_j|, 0)^2 / (2*beta)
        # For each pair (i,j) with violation v = 2t - d_ij > 0:
        # d/dp_i = v/beta * (p_i - p_j) / d_ij
        pair_v = np.maximum(2.0 * t - pair_dists, 0.0)
        for k in range(len(pair_v)):
            if pair_v[k] > 0 and pair_dists[k] > 1e-10:
                i = i_idx[k]
                j = j_idx[k]
                diff = pts[i] - pts[j]
                grad[2*i] += pair_v[k] * diff[0] / (beta * pair_dists[k])
                grad[2*i+1] += pair_v[k] * diff[1] / (beta * pair_dists[k])
                grad[2*j] -= pair_v[k] * diff[0] / (beta * pair_dists[k])
                grad[2*j+1] -= pair_v[k] * diff[1] / (beta * pair_dists[k])

        return obj, grad

    def optimize_with_lbfgs(pts_init, beta, maxiter=50, maxfun=500):
        """Run L-BFGS-B optimization with given beta."""
        t_init = compute_score(pts_init)
        x0 = np.concatenate([pts_init.flatten(), [t_init]])
        bounds = [(-0.999, 0.999)] * (2 * N) + [(0.001, 0.5)]

        try:
            result = minimize(
                soft_min_obj_and_grad, x0, args=(beta,),
                method='L-BFGS-B', bounds=bounds, jac=True,
                options={'maxiter': maxiter, 'maxfun': maxfun,
                         'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            if result.fun < 1e9:
                new_points = result.x[:2*N].reshape(-1, 2)
                return new_points, compute_score(new_points)
        except Exception:
            pass
        return pts_init, compute_score(pts_init)

    # 1. Generate multiple initializations
    hex_pts, hex_score = generate_hexagonal_initialization()
    ring_pts, ring_score = generate_ring_initialization()

    # Choose best initial point
    if ring_pts is not None and ring_score > hex_score:
        current_pts = ring_pts.copy()
        best_score = ring_score
    else:
        current_pts = hex_pts.copy()
        best_score = hex_score

    best_points = current_pts.copy()

    # 2. Multi-stage optimization with increasing beta
    # Stage 1: Low beta for global movement
    # Stage 2: Medium beta for refinement
    # Stage 3: High beta for precision
    stages = [
        (20.0, 40, 400),
        (100.0, 35, 350),
        (500.0, 30, 300),
        (2000.0, 25, 250),
        (10000.0, 20, 200),
    ]

    for beta, maxiter, maxfun in stages:
        if time.time() > deadline - 15:
            break

        new_pts, new_score = optimize_with_lbfgs(current_pts, beta, maxiter, maxfun)
        if new_score > best_score:
            best_score = new_score
            best_points = new_pts.copy()
            current_pts = new_pts.copy()

    # 3. Try optimizing from the other initialization if time permits
    if time.time() < deadline - 40:
        alt_pts = hex_pts if (ring_pts is not None and ring_score >= hex_score) else ring_pts
        if alt_pts is not None:
            alt_score = compute_score(alt_pts)
            if alt_score >= best_score * 0.95:  # Only if reasonably close
                for beta, maxiter, maxfun in [(50.0, 30, 300), (500.0, 25, 250)]:
                    if time.time() > deadline - 10:
                        break
                    new_pts, new_score = optimize_with_lbfgs(alt_pts, beta, maxiter, maxfun)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts.copy()
                        alt_pts = new_pts.copy()

    # 4. Final high-precision polish
    if time.time() < deadline - 5:
        for beta, maxiter, maxfun in [(5000.0, 30, 300), (20000.0, 25, 250)]:
            if time.time() > deadline - 3:
                break
            new_pts, new_score = optimize_with_lbfgs(best_points, beta, maxiter, maxfun)
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts.copy()

    # Final validation
    final_score = compute_score(best_points)
    if final_score <= 0:
        pts, _ = generate_hexagonal_initialization()
        return pts

    return best_points
# EVOLVE-BLOCK-END
