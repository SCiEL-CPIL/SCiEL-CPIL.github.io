import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    def get_pairwise_dists_sq(pts):
        """Calculate squared pairwise distances for N points."""
        # pts shape (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Exclude diagonal
        idx = np.triu_indices(N, 1)
        return dists_sq[idx]

    def true_score(pts):
        """Calculate the actual score: (max/min)^2."""
        d2 = get_pairwise_dists_sq(pts)
        min_d2 = np.min(d2)
        max_d2 = np.max(d2)
        if min_d2 < 1e-10:
            return 1e10
        return (max_d2 / min_d2)

    def soft_objective(params, beta):
        """
        Smooth surrogate objective.
        params: flattened array of 28 values (14 points x 2).
        beta: sharpness parameter for log-sum-exp.
        """
        if time.time() > deadline:
            return 1e10
        
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(14, 2)
        
        d2 = get_pairwise_dists_sq(pts)
        
        # Check for invalid points (too close)
        if np.min(d2) < 1e-6:
            return 1e10 + (1e-6 - np.min(d2)) * 1e6

        # log(max d2) approx
        # sum exp(beta * d2)
        # We want log(sum(exp(beta*d2))) / beta
        # Use log-sum-exp for stability
        max_val = np.max(beta * d2)
        log_sum_exp_max = max_val + np.log(np.sum(np.exp(beta * d2 - max_val)))
        log_max_d2 = log_sum_exp_max / beta
        
        # log(min d2) approx
        # sum exp(-beta * d2)
        # -log(sum(exp(-beta*d2))) / beta
        min_val = np.min(-beta * d2) # This is -beta * max(d2)
        # Actually, just compute directly with stability
        neg_beta_d2 = -beta * d2
        max_neg = np.max(neg_beta_d2)
        log_sum_exp_min = max_neg + np.log(np.sum(np.exp(neg_beta_d2 - max_neg)))
        log_min_d2 = -log_sum_exp_min / beta
        
        # Objective: log(max) - log(min)
        # Minimizing this minimizes the ratio
        return log_max_d2 - log_min_d2

    def normalize_and_optimize(initial_pts):
        """
        Take a set of points, normalize them (fix min-dist pair to (0,0)-(1,0)),
        and optimize the remaining coordinates using L-BFGS-B with beta annealing.
        """
        if time.time() > deadline - 5:
            return initial_pts, true_score(initial_pts)

        # Find min distance pair
        d2 = get_pairwise_dists_sq(initial_pts)
        min_idx = np.argmin(d2)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = initial_pts[i0]
        p1 = initial_pts[i1]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            # Degenerate, try random perturbation
            return initial_pts, 1e10

        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform points
        transformed = (initial_pts - p0) @ R.T / norm_v
        
        # Reorder: i0 -> 0, i1 -> 1, rest -> 2..15
        perm = [i0, i1] + [i for i in range(N) if i not in (i0, i1)]
        transformed = transformed[perm]
        
        init_params = transformed[2:].flatten()
        
        best_params = init_params
        best_obj = soft_objective(init_params, 10.0)
        
        # Annealing schedule for beta
        betas = [10.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            def obj_func(p):
                return soft_objective(p, beta)
            
            try:
                res = minimize(obj_func, best_params, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        # Construct final points
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(14, 2)
        
        final_score = true_score(final_pts)
        return final_pts, final_score

    best_pts = None
    best_score = 1e10

    # 1. Regular 4x4 Grid
    grid_pts = np.array([[i % 4, i // 4] for i in range(N)], dtype=float)
    # Add some noise to break symmetry
    np.random.seed(42)
    grid_pts += np.random.normal(0, 0.1, grid_pts.shape)
    pts, score = normalize_and_optimize(grid_pts)
    if score < best_score:
        best_score = score
        best_pts = pts

    # 2. Hexagonal Lattice
    hex_pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append((x, y))
    hex_pts = np.array(hex_pts)
    # Select 16 closest to center
    dists_c = np.sum(hex_pts**2, axis=1)
    sorted_idx = np.argsort(dists_c)
    hex_subset = hex_pts[sorted_idx[:N]]
    hex_subset += np.random.normal(0, 0.1, hex_subset.shape)
    pts, score = normalize_and_optimize(hex_subset)
    if score < best_score:
        best_score = score
        best_pts = pts

    # 3. Points on a Circle (regular 16-gon)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    circle_pts += np.random.normal(0, 0.1, circle_pts.shape)
    pts, score = normalize_and_optimize(circle_pts)
    if score < best_score:
        best_score = score
        best_pts = pts

    # 4. Random Restarts
    for i in range(20):
        if time.time() > deadline - 20:
            break
        # Random points in a square
        rand_pts = np.random.uniform(-1, 1, (N, 2))
        # Ensure distinct
        pts, score = normalize_and_optimize(rand_pts)
        if score < best_score:
            best_score = score
            best_pts = pts

    # 5. "Superposition" of two 8-gons? Or just more random restarts with different seeds
    for i in range(20):
        if time.time() > deadline - 5:
            break
        np.random.seed(100 + i)
        rand_pts = np.random.uniform(-1, 1, (N, 2))
        pts, score = normalize_and_optimize(rand_pts)
        if score < best_score:
            best_score = score
            best_pts = pts

    if best_pts is None:
        best_pts = grid_pts

    return best_pts
