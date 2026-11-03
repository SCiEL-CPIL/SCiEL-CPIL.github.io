import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 300

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin (evaluator kills at 150s)

    def compute_score(pts):
        """Compute the minimum radius for a configuration of points."""
        d = pdist(pts)
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        min_pair = np.min(d) / 2.0
        min_bound = np.min(bound_dists)
        return min(min_pair, min_bound)

    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2

    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])

    points = np.array(points_list)

    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = compute_score(points)

    # Precompute indices for pairwise distances (for gradient computation)
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)

    # Phase 1: Soft-min gradient descent to find a good local optimum
    # This is kept to establish a good starting point for the QP-like refinement
    alpha = 0.05
    cooling = 0.9999
    k_soft_start = 10.0
    k_soft_end = 500.0
    max_iter = 2000
    iter_count = 0
    stall_count = 0
    stall_threshold = 50
    k_soft = k_soft_start

    while time.time() - start_time < time_limit * 0.3 and iter_count < max_iter:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_mat, np.inf)

        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad

        pair_vals = dists_mat[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])

        current_min = np.min(all_vals)

        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1

        # Soft-min weights
        shifted_vals = all_vals - current_min
        max_shift = np.max(shifted_vals)
        if max_shift > 50.0:
            mask_close = shifted_vals < 50.0
            weights = np.zeros_like(all_vals)
            if np.any(mask_close):
                close_vals = shifted_vals[mask_close]
                close_weights = np.exp(-k_soft * close_vals)
                weights[mask_close] = close_weights
                sum_weights = np.sum(weights)
                if sum_weights > 0:
                    weights /= sum_weights
                else:
                    weights = np.ones_like(all_vals) / len(all_vals)
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.exp(-k_soft * shifted_vals)
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights
            else:
                weights = np.ones_like(all_vals) / len(all_vals)

        grads = np.zeros_like(points)

        pair_weights = weights[:num_pairs]
        d_pair = dists_mat[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)

        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads

        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm

        points += alpha * grads

        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]

        alpha *= cooling
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 2)

        if stall_count > stall_threshold:
            noise_scale = alpha * 3.0
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count = 0
            alpha *= 0.8

        iter_count += 1

    # Phase 2: SLSQP Refinement
    # We formulate the problem as maximizing r.
    # Variables: x = [x_0, y_0, ..., x_{N-1}, y_{N-1}, r]
    # Objective: minimize -r
    # Constraints:
    #   1. x_i^2 + y_i^2 <= (1 - r)^2  (Inside unit circle with margin r)
    #      => (1-r)^2 - (x_i^2 + y_i^2) >= 0
    #   2. (x_i - x_j)^2 + (y_i - y_j)^2 >= (2r)^2 (Non-overlapping)
    #      => (x_i - x_j)^2 + (y_i - y_j)^2 - (2r)^2 >= 0

    x0 = np.concatenate([best_points.flatten(), [best_score]])

    def objective(x):
        return -x[-1]

    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad

    # Constraints: return value >= 0
    # We will define a function that returns the vector of constraint values.
    # For SLSQP, we pass a function that returns an array of constraint values.
    
    def constraints(x):
        pts = x[:-1].reshape(N, 2)
        r = x[-1]
        
        # Boundary constraints: (1-r)^2 - |p_i|^2 >= 0
        rad_sq = np.sum(pts**2, axis=1)
        bound_cons = (1.0 - r)**2 - rad_sq
        
        # Pairwise constraints: |p_i - p_j|^2 - (2r)^2 >= 0
        # We need to be efficient here.
        # Computing all pairs is O(N^2).
        diff = pts[i_idx, :] - pts[j_idx, :]
        dist_sq = np.sum(diff**2, axis=1)
        pair_cons = dist_sq - (2.0 * r)**2
        
        return np.concatenate([bound_cons, pair_cons])

    def constraints_jac(x):
        pts = x[:-1].reshape(N, 2)
        r = x[-1]
        
        # Jacobian shape: (N + num_pairs, 2*N + 1)
        jac = np.zeros((N + num_pairs, 2*N + 1))
        
        # Boundary constraints: c_i = (1-r)^2 - x_i^2 - y_i^2
        # dc_i/dx_i = -2*x_i, dc_i/dy_i = -2*y_i, dc_i/dr = -2*(1-r)
        for i in range(N):
            jac[i, 2*i] = -2.0 * pts[i, 0]
            jac[i, 2*i+1] = -2.0 * pts[i, 1]
            jac[i, -1] = -2.0 * (1.0 - r)
            
        # Pairwise constraints: c_k = (x_i-x_j)^2 + (y_i-y_j)^2 - (2r)^2
        # Let i = i_idx[k], j = j_idx[k]
        # dc_k/dx_i = 2*(x_i-x_j), dc_k/dy_i = 2*(y_i-y_j)
        # dc_k/dx_j = 2*(x_j-x_i), dc_k/dy_j = 2*(y_j-y_i)
        # dc_k/dr = -8*r
        
        diff = pts[i_idx, :] - pts[j_idx, :]
        
        # Update rows for pairs
        row_offset = N
        # dc/dx_i
        jac[row_offset:row_offset+num_pairs, 2*i_idx] += 2.0 * diff[:, 0]
        jac[row_offset:row_offset+num_pairs, 2*i_idx+1] += 2.0 * diff[:, 1]
        
        # dc/dx_j
        jac[row_offset:row_offset+num_pairs, 2*j_idx] -= 2.0 * diff[:, 0]
        jac[row_offset:row_offset+num_pairs, 2*j_idx+1] -= 2.0 * diff[:, 1]
        
        # dc/dr
        jac[row_offset:row_offset+num_pairs, -1] -= 8.0 * r
        
        return jac

    # Bounds for variables
    # x_i, y_i in [-1, 1]
    # r in [0, 1]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 1.0)]

    polish_start_time = time.time()
    polish_time_limit = time_limit - 5.0

    # Try SLSQP
    # Since SLSQP can be slow, we limit iterations and time.
    # We wrap the call to check time periodically? No, SLSQP is a black box.
    # We must rely on maxiter and the time limit of the whole construct.
    # To be safe, we run SLSQP with a modest maxiter.
    
    # We might run SLSQP multiple times if it fails or gets stuck, but usually once is enough for polishing.
    
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad, 
                       constraints={'type': 'ineq', 'fun': constraints, 'jac': constraints_jac},
                       bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10})
        
        if res.success or res.fun < -best_score:
            new_points = res.x[:-1].reshape(N, 2)
            # Ensure validity
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
    except Exception:
        pass

    # If time remains, try a second pass or different starting point?
    # For now, return the best found.
    
    # Final check
    final_score = compute_score(best_points)
    if final_score < best_score:
        # This shouldn't happen if we only update on improvement, but just in case
        pass

    return best_points
