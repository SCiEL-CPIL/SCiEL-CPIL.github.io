import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init(n):
    """Generate hexagonal lattice initialization for circle packing."""
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95
    
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y))
                
    points = np.array(points)
    
    if len(points) >= n:
        radii = np.linalg.norm(points, axis=1)
        indices = np.argsort(radii)[:n]
        return points[indices]
    else:
        r_max = np.max(np.linalg.norm(points, axis=1))
        scale = 1.0 / r_max if r_max > 0 else 1.0
        return points * scale

def _ring_init(n):
    """Initialize points in concentric rings with staggered phases."""
    points = []
    num_rings = int(np.sqrt(n / np.pi * 2)) + 1
    
    ring_counts = []
    for k in range(num_rings):
        if k == 0:
            ring_counts.append(1)
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n / np.pi)))
            count = max(1, count)
            ring_counts.append(count)
    
    diff = n - sum(ring_counts)
    if diff > 0:
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        for i in range(len(ring_counts)-1, -1, -1):
            remove = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= remove
            diff += remove
            if diff >= 0:
                break
    
    while sum(ring_counts) > n:
        for i in range(len(ring_counts)-1, -1, -1):
            if ring_counts[i] > 1:
                ring_counts[i] -= 1
                break
    while sum(ring_counts) < n:
        ring_counts[-1] += 1
    
    idx = 0
    for k, count in enumerate(ring_counts):
        if k == 0:
            if count >= 1:
                points.append([0.0, 0.0])
                idx += 1
        else:
            radius = k * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
            phase = (k % 2) * np.pi / count
            for m in range(count):
                if idx >= n:
                    break
                angle = 2 * np.pi * m / count + phase
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                if x*x + y*y <= 1.0:
                    points.append([x, y])
                    idx += 1
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x*x + y*y <= 1.0:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    
    return points

def _compute_score(points):
    """Compute the true packing score."""
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d) / 2.0
    
    # Distance to boundary
    r = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - r)
    
    return min(min_d, min_boundary)

def _make_objective(beta):
    """Create objective function using soft-min (log-sum-exp) for all constraints."""
    def objective_and_grad(x):
        x = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        r = np.linalg.norm(x, axis=1)
        d_boundary = 1.0 - r
        
        # All constraint values (distances / 2 for pairs, distances for boundary)
        y_pairs = d / 2.0
        # Remove diagonal
        y_pairs = np.where(np.eye(N, dtype=bool), np.inf, y_pairs)
        
        all_y = np.concatenate([y_pairs.flatten(), d_boundary])
        # Filter out inf values
        finite_mask = np.isfinite(all_y)
        all_y = all_y[finite_mask]
        
        if len(all_y) == 0:
            return 0.0, np.zeros_like(x)
        
        y_min = np.min(all_y)
        
        # Soft-min via log-sum-exp
        z = -beta * all_y
        max_z = np.max(z)
        exp_terms = np.exp(z - max_z)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            return 0.0, np.zeros_like(x)
        
        log_sum_exp = max_z + np.log(sum_exp)
        obj_val = log_sum_exp / beta
        
        weights = exp_terms / sum_exp
        
        grad = np.zeros_like(x)
        
        # Gradient from pairwise terms
        n_pairs = N * (N - 1) // 2
        w_pairs = weights[:n_pairs]
        
        # We need to map the flattened upper triangle back to indices
        # This is expensive, so let's use a different approach
        # Use the full matrix and handle symmetric contributions
        
        # Actually, let's restructure: compute gradient using the full distance matrix
        # For each pair (i,j) with i<j, the contribution to grad[i] and grad[j]
        
        # Let's use a more efficient approach:
        # d[i,j] = ||x[i] - x[j]||
        # d/dx[i] (d[i,j]) = (x[i] - x[j]) / d[i,j]
        # d/dx[j] (d[i,j]) = (x[j] - x[i]) / d[i,j]
        
        # The objective is soft_min of {d[i,j]/2 for i<j} and {1-||x[i]||}
        # d(obj)/dx[i] = sum_j w_ij * (x[i]-x[j])/(2*d[i,j]) - w_i_boundary * x[i]/||x[i]||
        
        # Let's compute this properly
        d_safe = np.where(d < 1e-12, 1e-12, d)
        
        # Weight matrix for pairs
        W = np.zeros((N, N))
        # Fill upper triangle with weights
        for idx in range(n_pairs):
            i = idx // (N - 1)
            j = (idx % (N - 1)) + i + 1
            W[i, j] = w_pairs[idx]
            W[j, i] = w_pairs[idx]
        
        # Gradient from pairwise terms
        # For each i: sum_j W[i,j] * (x[i] - x[j]) / (2 * d[i,j])
        grad_pairs = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                if i != j and W[i, j] > 0:
                    grad_pairs[i] += W[i, j] * (x[i] - x[j]) / (2.0 * d_safe[i, j])
        
        # Gradient from boundary terms
        w_boundary = weights[n_pairs:]
        r_safe = np.where(r < 1e-12, 1e-12, r)
        grad_boundary = -x / r_safe[:, np.newaxis]
        
        grad = grad_pairs - w_boundary[:, np.newaxis] * grad_boundary
        
        return obj_val, grad
    
    return objective_and_grad

def _make_objective_v2(beta):
    """More efficient objective using vectorized operations."""
    def objective_and_grad(x):
        x = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        r = np.linalg.norm(x, axis=1)
        d_boundary = 1.0 - r
        
        # Collect all finite constraint values
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
        obj_val = log_sum_exp / beta
        
        weights = exp_terms / sum_exp
        
        # Now compute gradient
        grad = np.zeros_like(x)
        
        # Pairwise contribution
        n_pairs = N * (N - 1) // 2
        w_pairs = weights[:n_pairs]
        
        d_safe = np.where(d < 1e-12, 1e-12, d)
        
        # Vectorized pairwise gradient
        # For pair (i,j): grad[i] += w * (x[i]-x[j])/(2*d[i,j]), grad[j] += w * (x[j]-x[i])/(2*d[i,j])
        # Use the full symmetric weight matrix
        W = np.zeros((N, N))
        # Fill using the fact that pairs are stored in upper triangle order
        # idx = i*(N-1) - i*(i+1)//2 + (j-i-1) for i < j
        # Simpler: iterate
        ii, jj = np.triu_indices(N, 1)
        W[ii, jj] = w_pairs
        W[jj, ii] = w_pairs
        
        # grad_pairs[i] = sum_j W[i,j] * (x[i] - x[j]) / (2 * d[i,j])
        # = x[i] * sum_j W[i,j]/(2*d[i,j]) - sum_j W[i,j]*x[j]/(2*d[i,j])
        
        W_over_2d = W / (2.0 * d_safe)
        np.fill_diagonal(W_over_2d, 0)
        
        row_sum = np.sum(W_over_2d, axis=1)
        grad_pairs = x * row_sum[:, np.newaxis] - W_over_2d @ x
        
        # Boundary contribution
        w_boundary = weights[n_pairs:]
        r_safe = np.where(r < 1e-12, 1e-12, r)
        grad_boundary = -x / r_safe[:, np.newaxis]
        
        grad = grad_pairs - w_boundary[:, np.newaxis] * grad_boundary
        
        return obj_val, grad
    
    return objective_and_grad

def _local_search(points, time_budget):
    """Perform local search by iteratively improving the worst constraint."""
    best_x = points.copy()
    best_score = _compute_score(best_x)
    
    start_time = time.time()
    step_size = 0.005
    no_improve_count = 0
    
    max_iters = 2000
    
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
        
        if no_improve_count > 100:
            step_size *= 0.7
            if step_size < 1e-7:
                break
            no_improve_count = 0
        
        # Find the point involved in the worst constraint
        if min_pair_d < min_boundary:
            # Find the pair with minimum distance
            idx_min = np.unravel_index(np.argmin(d), d.shape)
            pt_idx = idx_min[0]
            other_idx = idx_min[1]
            # Move both points slightly apart
            direction = (best_x[pt_idx] - best_x[other_idx])
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-12:
                direction /= norm_dir
            else:
                direction = np.array([1.0, 0.0])
            
            # Try moving pt_idx in direction and other_idx in opposite
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
                # Try random perturbations
                for _ in range(10):
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
            # Boundary is the constraint, move the point closest to boundary inward
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
                for _ in range(10):
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
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
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
    beta_schedule = [30, 60, 120, 250, 500, 1000]
    
    for beta in beta_schedule:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break
        
        obj_func = _make_objective_v2(beta)
        
        try:
            res = minimize(
                obj_func, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 80, 'gtol': 1e-9}
            )
            
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate_x)):
                    score = _compute_score(candidate_x)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    # Local search to fine-tune
    elapsed = time.time() - start_time
    if elapsed < time_limit - 15:
        time_budget = min(40, time_limit - elapsed - 5)
        if time_budget > 10:
            try:
                best_x, best_score = _local_search(best_x, time_budget)
            except Exception:
                pass
    
    # Final polish with very high beta if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        try:
            obj_func_polish = _make_objective_v2(3000)
            res = minimize(
                obj_func_polish, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 40, 'gtol': 1e-11}
            )
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate_x)):
                    score = _compute_score(candidate_x)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    return best_x
