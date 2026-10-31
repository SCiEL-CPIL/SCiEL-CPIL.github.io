"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin for safety

    def score_points(pts):
        """Compute (max_d / min_d)^2 for a set of points."""
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    def normalize_points(pts):
        """
        Normalize points by fixing the first two points (closest pair) to (0,0) and (1,0).
        Returns normalized points array of shape (N, 2).
        """
        # Find the pair with minimum distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_mat = dists[iu]
        min_idx = np.argmin(d_mat)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0

        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return pts  # Fallback

        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        # Transform all points: translate p0 to origin, rotate, scale
        transformed = (pts - p0) @ R_mat.T / norm_v

        # Reorder so that i0 is at index 0 and i1 is at index 1
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining

        transformed = transformed[perm]
        return transformed

    def continuous_refine(pts):
        """
        Refine points using continuous optimization.
        Uses a smooth approximation of the objective: 
        f(x) = log(sum(exp(2*d_i^2))) - log(sum(exp(-2*d_i^2)))
        This is smooth and differentiable, unlike the raw max/min ratio.
        """
        if time.time() > deadline - 10:
            return pts, score_points(pts)

        normalized = normalize_points(pts)
        init_params = normalized[2:].flatten()

        # Temperature parameter for smoothness
        # Lower beta -> sharper max/min, higher beta -> smoother
        # We want to approximate max(d) and min(d)
        # max(d) approx = (1/beta) * log(sum(exp(beta * d)))
        # min(d) approx = -(1/beta) * log(sum(exp(-beta * d)))
        # Ratio^2 approx = [ (1/beta) log(sum(exp(beta d))) / (-(1/beta) log(sum(exp(-beta d))) ) ]^2
        #                = [ log(sum(exp(beta d))) / log(sum(exp(-beta d))) ]^2
        
        # However, standard log-sum-exp can be unstable.
        # Let's use a fixed beta that balances smoothness and accuracy.
        # Since we are optimizing, we can start with a smaller beta and increase it?
        # Or just use a moderately large beta.
        # Let's try beta = 10.
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts_local = np.zeros((N, 2))
            pts_local[0] = [0.0, 0.0]
            pts_local[1] = [1.0, 0.0]
            pts_local[2:] = params.reshape(-1, 2)
            
            d = pdist(pts_local, 'euclidean')
            
            if np.min(d) < 1e-8:
                return 1e10
            
            # Smooth max/min approximation
            beta = 20.0
            
            # Numerical stability for log-sum-exp
            d_max = np.max(d)
            d_min = np.min(d)
            
            # log(sum(exp(beta * d))) = beta * d_max + log(sum(exp(beta * (d - d_max))))
            log_sum_exp_max = beta * d_max + np.log(np.sum(np.exp(beta * (d - d_max))))
            
            # log(sum(exp(-beta * d))) = -beta * d_min + log(sum(exp(-beta * (d - d_min))))
            # Note: exp(-beta * (d - d_min)) = exp(-beta * d + beta * d_min)
            # So sum(exp(-beta * d)) = exp(-beta * d_min) * sum(exp(-beta * (d - d_min)))
            # log(sum(exp(-beta * d))) = -beta * d_min + log(sum(exp(-beta * (d - d_min))))
            log_sum_exp_min = -beta * d_min + np.log(np.sum(np.exp(-beta * (d - d_min))))
            
            # The ratio is (smooth_max / smooth_min)^2
            # smooth_max = log_sum_exp_max / beta
            # smooth_min = log_sum_exp_min / beta  (Wait, min approx is -1/beta log(sum(exp(-beta d))))
            # Let's re-verify:
            # min(d) = -1/beta * log(sum(exp(-beta * d)))
            # So smooth_min = -1/beta * log_sum_exp_min_val where log_sum_exp_min_val is log(sum(exp(-beta d)))
            # Let's just compute the values directly.
            
            # Actually, simpler:
            # approx_max = log_sum_exp_max / beta
            # approx_min = -log_sum_exp_min / beta  # Since log_sum_exp_min is log(sum(exp(-beta d)))
            
            approx_max = log_sum_exp_max / beta
            approx_min = -log_sum_exp_min / beta
            
            if approx_min < 1e-8:
                return 1e10
                
            return (approx_max / approx_min) ** 2

        def gradient(params):
            # We can use numerical gradient or derive analytical.
            # Numerical gradient is safer for implementation speed, but slower per step.
            # L-BFGS-B can use BFGS approximation if jac is not provided.
            # Given the smoothness, BFGS should work well.
            # If we need speed, we can implement numerical gradient.
            # Let's rely on BFGS first.
            return None

        bnds = [(-5, 5)] * 28

        try:
            # Use L-BFGS-B with numerical gradient (default) or BFGS
            # L-BFGS-B is memory efficient.
            res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            best_params = res.x
            # Evaluate the TRUE objective on the result to ensure improvement
            final_pts = np.zeros((N, 2))
            final_pts[0] = [0.0, 0.0]
            final_pts[1] = [1.0, 0.0]
            final_pts[2:] = best_params.reshape(-1, 2)
            
            true_score = score_points(final_pts)
            
            # Compare with initial score
            init_score = score_points(normalized)
            
            if true_score < init_score:
                return final_pts, true_score
            else:
                return normalized, init_score

        except Exception:
            return normalized, score_points(normalized)

    # ===== Phase 1: Generate diverse initial configurations =====
    candidates = []

    # 1. Hexagonal lattice - 16 closest points to origin
    R = 5
    hex_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)

    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N].tolist()
    candidates.append(hex_pts[initial_subset].copy())

    # 2. Concentric circles (2 rings of 8 points)
    ring1_r = 1.0
    ring2_r = 2.0
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    circ_pts = np.vstack([
        np.column_stack([ring1_r * np.cos(angles1), ring1_r * np.sin(angles1)]),
        np.column_stack([ring2_r * np.cos(angles2), ring2_r * np.sin(angles2)])
    ])
    candidates.append(circ_pts)

    # 3. Perturbed hexagonal lattice
    rng = np.random.RandomState(42)
    perturbed = hex_pts[initial_subset].copy()
    perturbed += rng.normal(0, 0.1, perturbed.shape)
    candidates.append(perturbed)

    # 4. Grid with perturbation
    grid_pts = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
    grid_pts += rng.normal(0, 0.05, grid_pts.shape)
    candidates.append(grid_pts)

    # 5. Another hexagonal configuration (slightly different subset)
    if len(sorted_indices) > N:
        alt_subset = sorted_indices[1:N+1].tolist()
        candidates.append(hex_pts[alt_subset].copy())

    # ===== Phase 2: Local search on hexagonal lattice for best candidate =====
    best_score = 1e10
    best_pts = None

    # Focus local search on the hexagonal lattice candidate (index 0)
    hex_candidate = candidates[0]
    hex_score = score_points(hex_candidate)

    if hex_score < best_score:
        best_score = hex_score
        best_pts = hex_candidate.copy()

    # Local search: swap points in/out of subset
    candidate_indices = sorted_indices[:100].tolist()
    in_subset = set(initial_subset)
    outside = [i for i in candidate_indices if i not in in_subset]
    current_subset = initial_subset.copy()
    current_score = hex_score

    time_for_local_search = 20  # seconds
    local_deadline = start_time + time_for_local_search

    improved = True
    while improved and time.time() < local_deadline:
        improved = False
        for i in range(N):
            if time.time() > local_deadline:
                break
            for j in outside:
                if time.time() > local_deadline:
                    break
                new_subset = current_subset.copy()
                new_subset[i] = j
                new_pts = hex_pts[new_subset]
                new_score = score_points(new_pts)
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    current_subset = new_subset
                    in_subset = set(current_subset)
                    outside = [k for k in candidate_indices if k not in in_subset]
                    improved = True

    if current_score < best_score:
        best_score = current_score
        best_pts = hex_pts[current_subset].copy()

    # ===== Phase 3: Refine all candidates with continuous optimization =====
    # Allocate remaining time to refinement
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining < 10:
        return best_pts

    # Refine the best candidate first
    refined_pts, refined_score = continuous_refine(best_pts)
    if refined_score < best_score:
        best_score = refined_score
        best_pts = refined_pts

    # Try refining other candidates if time permits
    for idx in range(1, len(candidates)):
        if time.time() > deadline - 15:
            break
        cand = candidates[idx]
        cand_score = score_points(cand)
        if cand_score < best_score * 1.5:  # Only refine if promising
            refined_pts, refined_score = continuous_refine(cand)
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined_pts

    # Final check
    final_score = score_points(best_pts)
    if final_score < best_score:
        best_score = final_score

    return best_pts

# EVOLVE-BLOCK-END
