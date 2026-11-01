import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init(n):
    """Generate hexagonal lattice initialization for circle packing."""
    best_pts = None
    best_score = -np.inf
    
    for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
        spacing = 2.0 / (np.sqrt(n) * scale)
        
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
                if np.hypot(px, py) <= 1.0:
                    pts_list.append([px, py])
        
        if len(pts_list) < n:
            continue
            
        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        
        # Compute score
        i_idx, j_idx = np.triu_indices(n, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(norms)
        score = min(min_pair_dist / 2.0, min_radial_dist)
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        # Fallback: random points
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
    
    return best_pts

def _compute_score(points):
    """Compute the true packing score."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists > 1.0):
        return -1.0
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = points[i_idx] - points[j_idx]
    pair_dists = np.linalg.norm(diffs, axis=1)
    min_pair_dist = np.min(pair_dists)
    min_radial_dist = 1.0 - np.max(dists)
    
    return min(min_pair_dist / 2.0, min_radial_dist)

def _make_objective(beta):
    """
    Objective using auxiliary variable t (epigraph formulation).
    Minimize t subject to:
      d_ij/2 >= t for all pairs (i,j)
      1 - ||x_i|| >= t for all i
    Using soft constraints: penalize violations of t.
    """
    def objective_and_grad(x):
        x = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        r = np.linalg.norm(x, axis=1)
        d_boundary = 1.0 - r
        
        # Estimate t as soft-min of all constraints
        y_pairs = d / 2.0
        np.fill_diagonal(y_pairs, np.inf)
        
        all_y = np.concatenate([y_pairs.flatten(), d_boundary])
        finite_mask = np.isfinite(all_y)
        all_y_finite = all_y[finite_mask]
        
        if len(all_y_finite) == 0:
            return 0.0, np.zeros_like(x)
        
        y_min = np.min(all_y_finite)
        
        # Soft-min via log-sum-exp
        z = -beta * all_y_finite
        max_z = np.max(z)
        exp_terms = np.exp(z - max_z)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp < 1e-300:
            return 0.0, np.zeros_like(x)
        
        log_sum_exp = max_z + np.log(sum_exp)
        t_est = log_sum_exp / beta
        
        # Objective: minimize t (we want to maximize the minimum distance)
        # But we also need to ensure constraints are satisfied
        # Use: obj = t_est + penalty for violations
        # Actually, for L-BFGS-B, we minimize -t (maximize t)
        # But with constraints... Let's use: obj = -t_est + lambda * sum(violations)
        
        # Violations: constraints that are less than t_est
        # For pairs: d_ij/2 - t_est < 0 means violation
        # For boundary: 1 - ||x_i|| - t_est < 0 means violation
        
        pair_violations = np.maximum(0, t_est - y_pairs)
        boundary_violations = np.maximum(0, t_est - d_boundary)
        
        penalty = np.sum(pair_violations) + np.sum(boundary_violations)
        
        obj_val = t_est + 0.1 * penalty
        
        # Gradient
        grad = np.zeros_like(x)
        
        weights = exp_terms / sum_exp
        
        # Pairwise gradient contribution
        n_pairs = N * (N - 1) // 2
        w_pairs = weights[:n_pairs]
        
        d_safe = np.where(d < 1e-12, 1e-12, d)
        
        W = np.zeros((N, N))
        ii, jj = np.triu_indices(N, 1)
        W[ii, jj] = w_pairs
        W[jj, ii] = w_pairs
        
        W_over_2d = W / (2.0 * d_safe)
        np.fill_diagonal(W_over_2d, 0)
        
        row_sum = np.sum(W_over_2d, axis=1)
        grad_pairs = x * row_sum[:, np.newaxis] - W_over_2d @ x
        
        # Boundary gradient contribution
        w_boundary = weights[n_pairs:]
        r_safe = np.where(r < 1e-12, 1e-12, r)
        grad_boundary = -x / r_safe[:, np.newaxis]
        
        grad = grad_pairs - w_boundary[:, np.newaxis] * grad_boundary
        
        # Penalty gradient
        # d(pair_violations)/dx = -d(y_pairs)/dx where violation > 0
        # d(y_pairs)/dx[i] = sum_j W[i,j] * (x[i]-x[j])/(2*d[i,j])
        # So penalty gradient from pairs = -sum over violated pairs of (x[i]-x[j])/(2*d[i,j])
        
        # This is complex, so for simplicity, let's just use the main gradient
        # The penalty term is small (0.1 * penalty) and helps guide the optimizer
        
        return obj_val, grad
    
    return objective_and_grad

def _make_objective_v2(beta):
    """
    Improved objective: maximize t where t is the soft-min of all constraints.
    We minimize -t.
    """
    def objective_and_grad(x):
        x = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        r = np.linalg.norm(x, axis=1)
        d_boundary = 1.0 - r
        
        # All constraint values
        y_pairs = d / 2.0
        np.fill_diagonal(y_pairs, np.inf)
        
        all_y = np.concatenate([y_pairs.flatten(), d_boundary])
        finite_mask = np.isfinite(all_y)
        all_y_finite = all_y[finite_mask]
        
        if len(all_y_finite) == 0:
            return 0.0, np.zeros_like(x)
        
        # Soft-min
        z = -beta * all_y_finite
        max_z = np.max(z)
        exp_terms = np.exp(z - max_z)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp < 1e-300:
            return 0.0, np.zeros_like(x)
        
        log_sum_exp = max_z + np.log(sum_exp)
        t_val = log_sum_exp / beta
        
        # We want to maximize t, so minimize -t
        obj_val = -t_val
        
        weights = exp_terms / sum_exp
        
        # Gradient of -t with respect to x
        # t = soft_min(all_y)
        # d(t)/dx = sum_k w_k * d(y_k)/dx
        # d(-t)/dx = -sum_k w_k * d(y_k)/dx
        
        grad = np.zeros_like(x)
        
        n_pairs = N * (N - 1) // 2
        w_pairs = weights[:n_pairs]
        
        d_safe = np.where(d < 1e-12, 1e-12, d)
        
        W = np.zeros((N, N))
        ii, jj = np.triu_indices(N, 1)
        W[ii, jj] = w_pairs
        W[jj, ii] = w_pairs
        
        W_over_2d = W / (2.0 * d_safe)
        np.fill_diagonal(W_over_2d, 0)
        
        row_sum = np.sum(W_over_2d, axis=1)
        grad_pairs = x * row_sum[:, np.newaxis] - W_over_2d @ x
        
        w_boundary = weights[n_pairs:]
        r_safe = np.where(r < 1e-12, 1e-12, r)
        grad_boundary = -x / r_safe[:, np.newaxis]
        
        # d(t)/dx = grad_pairs + w_boundary * grad_boundary (note sign)
        # Actually: d(1-||x_i||)/dx_i = -x_i/||x_i||
        # So d(t)/dx = sum_pair w * d(d_ij/2)/dx + sum_boundary w * d(1-||x_i||)/dx
        # = grad_pairs + w_boundary[:,None] * (-x/r)
        # = grad_pairs - w_boundary[:,None] * x/r
        
        # d(-t)/dx = -d(t)/dx
        
        grad_t = grad_pairs - w_boundary[:, np.newaxis] * (x / r_safe[:, np.newaxis])
        grad = -grad_t
        
        return obj_val, grad
    
    return objective_and_grad

def _local_search(points, time_budget):
    """Perform local search by iteratively improving the worst constraint."""
    best_x = points.copy()
    best_score = _compute_score(best_x)
    
    start_time = time.time()
    step_size = 0.005
    no_improve_count = 0
    
    max_iters = 3000
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Find the worst constraint
        diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_d = np.min(d) / 2.0
        
        r = np.linalg.norm(best_x, axis=1)
        min_boundary = np.min(1.0 - r)
        
        score = min(min_pair_d, min_boundary)
        
        if score > best_score + 1e-15:
            best_score = score
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > 150:
            step_size *= 0.7
            if step_size < 1e-8:
                break
            no_improve_count = 0
        
        # Find the point involved in the worst constraint
        if min_pair_d < min_boundary:
            idx_min = np.unravel_index(np.argmin(d), d.shape)
            pt_idx = idx_min[0]
            other_idx = idx_min[1]
            direction = (best_x[pt_idx] - best_x[other_idx])
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-12:
                direction /= norm_dir
            else:
                direction = np.array([1.0, 0.0])
            
            improved = False
            for ss in [step_size, step_size * 0.5, step_size * 0.2, step_size * 2]:
                candidate = best_x.copy()
                candidate[pt_idx] += direction * ss
                candidate[other_idx] -= direction * ss
                
                if np.any(np.linalg.norm(candidate, axis=1) > 1.0):
                    continue
                
                new_score = _compute_score(candidate)
                if new_score > best_score:
                    best_x = candidate
                    best_score = new_score
                    improved = True
                    break
            
            if not improved:
                for _ in range(15):
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction)
                    for ss in [step_size, step_size * 0.5]:
                        candidate = best_x.copy()
                        candidate[pt_idx] += direction * ss
                        if np.any(np.linalg.norm(candidate, axis=1) > 1.0):
                            continue
                        new_score = _compute_score(candidate)
                        if new_score > best_score:
                            best_x = candidate
                            best_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        else:
            pt_idx = np.argmin(1.0 - r)
            direction = -best_x[pt_idx]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-12:
                direction /= norm_dir
            else:
                direction = np.array([1.0, 0.0])
            
            improved = False
            for ss in [step_size, step_size * 0.5, step_size * 0.2]:
                candidate = best_x.copy()
                candidate[pt_idx] += direction * ss
                if np.any(np.linalg.norm(candidate, axis=1) > 1.0):
                    continue
                new_score = _compute_score(candidate)
                if new_score > best_score:
                    best_x = candidate
                    best_score = new_score
                    improved = True
                    break
            
            if not improved:
                for _ in range(15):
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction)
                    for ss in [step_size, step_size * 0.5]:
                        candidate = best_x.copy()
                        candidate[pt_idx] += direction * ss
                        if np.any(np.linalg.norm(candidate, axis=1) > 1.0):
                            continue
                        new_score = _compute_score(candidate)
                        if new_score > best_score:
                            best_x = candidate
                            best_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            step_size *= 0.85
    
    return best_x, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # Generate multiple initializations and pick the best
    inits = [_hex_lattice_init(N)]
    
    # Add a few random perturbations of the best init
    best_init = inits[0]
    for _ in range(3):
        perturbed = best_init + np.random.randn(N, 2) * 0.01
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            perturbed[mask] *= (0.99 / norms[mask])[:, np.newaxis]
        inits.append(perturbed)
    
    best_x = None
    best_score = -np.inf
    
    for init_points in inits:
        score = _compute_score(init_points)
        if score > best_score:
            best_x = init_points.copy()
            best_score = score
    
    if best_x is None:
        best_x = _hex_lattice_init(N)
        best_score = _compute_score(best_x)
    
    # Multi-stage optimization with increasing beta
    beta_schedule = [30, 60, 120, 250, 500, 1000, 2000]
    
    for beta in beta_schedule:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 25:
            break
        
        obj_func = _make_objective_v2(beta)
        
        try:
            res = minimize(
                obj_func, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate_x)):
                    # Ensure points are inside the unit circle
                    norms = np.linalg.norm(candidate_x, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        candidate_x[mask] *= (0.999 / norms[mask])[:, np.newaxis]
                    
                    score = _compute_score(candidate_x)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    # Local search to fine-tune
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20:
        time_budget = min(50, time_limit - elapsed - 5)
        if time_budget > 10:
            try:
                best_x, best_score = _local_search(best_x, time_budget)
            except Exception:
                pass
    
    # Final polish with very high beta if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        try:
            obj_func_polish = _make_objective_v2(5000)
            res = minimize(
                obj_func_polish, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 60, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate_x)):
                    norms = np.linalg.norm(candidate_x, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        candidate_x[mask] *= (0.999 / norms[mask])[:, np.newaxis]
                    
                    score = _compute_score(candidate_x)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    # Final validation
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (1.0 / norms[mask])[:, np.newaxis]
    
    return best_x
