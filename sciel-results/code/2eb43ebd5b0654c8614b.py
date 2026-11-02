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
        if pts.shape != (N, 2):
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0 + 1e-9):
            return -1.0
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return max(0.0, r)

    def hex_lattice_init(scale=1.0, rotation=0.0):
        """Generate hexagonal lattice points clipped to unit circle."""
        spacing = 2.0 / (np.sqrt(N) * scale)
        nx = int(2.5 / spacing) + 3
        ny = int(2.5 / (spacing * np.sqrt(3.0) / 2.0)) + 3

        x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0

        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0:
                    pts_list.append([px, py])

        if len(pts_list) < N:
            return None

        pts = np.array(pts_list)
        # Apply rotation
        if rotation != 0.0:
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]])
            pts = pts @ R.T

        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:N]]
        return pts

    def ring_init(n_rings=12):
        """Generate ring-based initialization."""
        ring_radii = np.linspace(0.0, 0.95, n_rings)
        ring_points = []

        for r_ring in ring_radii:
            if r_ring < 0.01:
                ring_points.append([0.0, 0.0])
                continue
            est_spacing = 0.11
            n_circ = max(1, int(2 * np.pi * r_ring / est_spacing))
            angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            for a in angles:
                x = r_ring * np.cos(a)
                y = r_ring * np.sin(a)
                if np.hypot(x, y) <= 0.99:
                    ring_points.append([x, y])

        ring_points = np.array(ring_points)
        if len(ring_points) > N:
            norms = np.linalg.norm(ring_points, axis=1)
            idx = np.argsort(norms)[:N]
            ring_points = ring_points[idx]
        elif len(ring_points) < N:
            return None
        return ring_points

    def random_init(seed=42):
        """Generate random initialization."""
        rng = np.random.RandomState(seed)
        pts = rng.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        norms[norms == 0] = 1.0
        pts = pts * (0.95 / norms)[:, None]
        return pts

    def vectorized_soft_min_objective_and_grad(params, beta):
        """
        Vectorized soft-min over all critical constraints.
        Uses a smooth surrogate: -softmin(pair_dists/2, 1-norms)
        where softmin is computed over ALL pairwise distances and radial margins.
        
        This gives a much smoother landscape than tracking only the single min pair.
        """
        if time.time() > deadline:
            return 1e10, np.zeros(2 * N)

        pts = params.reshape(-1, 2)

        if not np.all(np.isfinite(pts)):
            return 1e10, np.zeros(2 * N)

        # Radial distances
        norms = np.linalg.norm(pts, axis=1)
        
        # Check bounds - penalize if outside
        if np.any(norms > 1.0):
            penalty = np.sum(np.maximum(norms - 1.0, 0.0)) ** 2
            return 1e10 + penalty, np.zeros(2 * N)

        # Pairwise distances - compute all
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        # Radial margins: 1 - norm_i
        radial_margins = 1.0 - norms
        
        # Combine all constraints into a single vector
        # Pairwise constraint values: pair_dists / 2
        # Radial constraint values: radial_margins
        all_vals = np.concatenate([pair_dists / 2.0, radial_margins])
        
        # Soft min using log-sum-exp
        m = np.min(all_vals)
        shifted = -beta * (all_vals - m)
        max_shifted = np.max(shifted)
        
        # Numerical stability
        if max_shifted > 500:
            # Essentially hard min
            weights = np.zeros_like(all_vals)
            min_idx = np.argmin(all_vals)
            weights[min_idx] = 1.0
            log_sum_exp = 0.0
        elif max_shifted < -500:
            log_sum_exp = max_shifted + np.log(len(all_vals))
            weights = np.ones_like(all_vals) / len(all_vals)
        else:
            exp_shifted = np.exp(shifted - max_shifted)
            sum_exp = np.sum(exp_shifted)
            log_sum_exp = max_shifted + np.log(sum_exp)
            weights = exp_shifted / sum_exp
        
        soft_min_val = m - log_sum_exp / beta
        fval = -soft_min_val
        
        # Gradient computation (vectorized)
        grad = np.zeros(2 * N)
        
        # n_pairs
        n_pairs = len(pair_dists)
        
        # Weights for pairwise constraints
        w_pair = weights[:n_pairs]
        # Weights for radial constraints
        w_rad = weights[n_pairs:]
        
        # Gradient of pairwise distances w.r.t. points
        # For pair (i,j) with distance d_ij:
        # d(d_ij)/d(pts[i]) = (pts[i] - pts[j]) / d_ij
        # d(d_ij)/d(pts[j]) = (pts[j] - pts[i]) / d_ij
        # Since we use pair_dists/2, multiply by 1/2
        
        # Vectorized: for each pair, add contribution to gradient
        safe_dists = np.maximum(pair_dists, 1e-12)
        unit_vecs = diffs / safe_dists[:, None]
        
        # Contribution from each pair: 0.5 * w_pair[k] * unit_vec[k] to pts[i_k]
        # and -0.5 * w_pair[k] * unit_vec[k] to pts[j_k]
        pair_contrib = 0.5 * w_pair[:, None] * unit_vecs
        
        # Scatter to gradient
        np.add.at(grad, 2 * i_idx, pair_contrib)
        np.add.at(grad, 2 * j_idx, -pair_contrib)
        
        # Gradient of radial margins: d(1 - norm_i)/d(pts[i]) = -pts[i] / norm_i
        safe_norms = np.maximum(norms, 1e-12)
        radial_unit = pts / safe_norms[:, None]
        radial_contrib = -w_rad[:, None] * radial_unit
        grad += radial_contrib
        
        # Since we're minimizing fval = -soft_min, the gradient of fval is -gradient of soft_min
        # The gradient we computed above is d(soft_min)/d(pts)
        # So d(fval)/d(pts) = -d(soft_min)/d(pts)
        grad = -grad
        
        return fval, grad

    def force_directed_refine(pts, n_iters=200, r_target=0.053):
        """
        Force-directed refinement: push overlapping circles apart.
        This is a local search that works well after gradient optimization.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute all pairwise distances
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            # Find pairs that are too close (overlapping)
            min_allowed = 2.0 * r_target
            too_close = pair_dists < min_allowed
            
            if not np.any(too_close):
                # Check radial constraint
                norms = np.linalg.norm(pts, axis=1)
                if np.max(norms) + r_target <= 1.0:
                    break
                # Otherwise push inward
                norms = np.linalg.norm(pts, axis=1)
                too_far = norms > 1.0 - r_target
                if np.any(too_far):
                    for idx in np.where(too_far)[0]:
                        norm_i = norms[idx]
                        if norm_i > 1e-12:
                            pts[idx] *= (1.0 - r_target - 0.001) / norm_i
                    continue
                break
            
            # Apply repulsive forces
            forces = np.zeros_like(pts)
            safe_dists = np.maximum(pair_dists[too_close], 1e-12)
            unit_vecs = diffs[too_close] / safe_dists[:, None]
            
            # Repulsion strength proportional to overlap
            overlaps = (min_allowed - pair_dists[too_close]) / min_allowed
            force_mags = 0.5 * overlaps[:, None] * unit_vecs
            
            np.add.at(forces, i_idx[too_close], force_mags)
            np.add.at(forces, j_idx[too_close], -force_mags)
            
            # Radial constraint: push inward if too close to boundary
            norms = np.linalg.norm(pts, axis=1)
            too_far = norms > 1.0 - r_target
            if np.any(too_far):
                for idx in np.where(too_far)[0]:
                    norm_i = norms[idx]
                    if norm_i > 1e-12:
                        forces[idx] -= pts[idx] / norm_i * 0.5 * (norms[idx] - (1.0 - r_target))
            
            # Damped update
            step_size = 0.1
            pts += step_size * forces
            
            # Clamp to unit circle
            norms = np.linalg.norm(pts, axis=1)
            too_far = norms > 1.0
            if np.any(too_far):
                for idx in np.where(too_far)[0]:
                    pts[idx] *= 1.0 / norms[idx]
            
            # Accept if improved
            new_score = compute_score(pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        
        return best_pts, best_score

    def optimize_from(pts, betas, max_iters_per_beta, use_fd_refine=True):
        """Run progressive beta annealing optimization from initial points."""
        current_pts = pts.copy()
        best_pts = current_pts.copy()
        best_score = compute_score(current_pts)

        for beta in betas:
            if time.time() > deadline - 5:
                break

            x0 = current_pts.flatten()
            bounds = [(-0.9999, 0.9999)] * (2 * N)

            try:
                result = minimize(
                    vectorized_soft_min_objective_and_grad, x0, args=(beta,),
                    method='L-BFGS-B', bounds=bounds, jac=True,
                    options={'maxiter': max_iters_per_beta, 'ftol': 1e-14,
                             'gtol': 1e-12, 'maxfun': max_iters_per_beta * 5}
                )

                if result.fun < 1e9:
                    new_points = result.x.reshape(-1, 2)
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_points.copy()
                        current_pts = new_points.copy()
            except Exception:
                pass

        # Force-directed refinement if time permits
        if use_fd_refine and time.time() < deadline - 10:
            r_est = best_score
            fd_pts, fd_score = force_directed_refine(current_pts, n_iters=100, r_target=max(r_est, 0.05))
            if fd_score > best_score:
                best_score = fd_score
                best_pts = fd_pts.copy()
                current_pts = fd_pts.copy()

        return best_pts, best_score

    # Generate multiple initializations
    inits = []

    # Hexagonal lattices with different scales and rotations
    for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
        for rot in [0.0, np.pi/6, np.pi/3]:
            pts = hex_lattice_init(scale, rot)
            if pts is not None and pts.shape == (N, 2):
                inits.append(pts)

    # Ring-based
    for nr in [10, 12, 14, 16]:
        pts = ring_init(nr)
        if pts is not None and pts.shape == (N, 2):
            inits.append(pts)

    # Random
    for seed in [42, 123, 456]:
        pts = random_init(seed)
        inits.append(pts)

    # Score all initializations
    init_scores = [(compute_score(p), i, p) for i, p in enumerate(inits)]
    init_scores.sort(key=lambda x: -x[0])

    # Take top diverse initializations
    selected = []
    for score, idx, pts in init_scores:
        if len(selected) >= 6:
            break
        # Check diversity
        is_diverse = True
        for sel_pts in selected:
            if np.allclose(pts, sel_pts, atol=0.01):
                is_diverse = False
                break
        if is_diverse:
            selected.append(pts)

    # Optimize from each selected initialization
    best_overall_pts = None
    best_overall_score = -1.0

    betas_stage1 = [20.0, 100.0, 500.0]
    betas_stage2 = [2000.0, 10000.0]

    for i, init_pts in enumerate(selected):
        if time.time() > deadline - 30:
            break

        # Stage 1: coarse to medium beta
        pts, score = optimize_from(init_pts, betas_stage1, max_iters_per_beta=40, use_fd_refine=False)

        # Stage 2: high beta polish
        pts, score = optimize_from(pts, betas_stage2, max_iters_per_beta=30, use_fd_refine=True)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts.copy()

    # Final ultra-high precision polish
    if best_overall_pts is not None and time.time() < deadline - 10:
        final_betas = [50000.0, 200000.0]
        best_overall_pts, best_overall_score = optimize_from(
            best_overall_pts, final_betas, max_iters_per_beta=50, use_fd_refine=True
        )

    # Final validation
    final_score = compute_score(best_overall_pts)
    if final_score <= 0 or best_overall_pts is None:
        # Fallback to best hexagonal
        pts = hex_lattice_init(1.0, 0.0)
        if pts is None:
            pts = random_init(42)
        return pts

    return best_overall_pts
# EVOLVE-BLOCK-END
