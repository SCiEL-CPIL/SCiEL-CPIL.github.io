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
    
    # Distribute points among rings proportional to circumference
    ring_counts = []
    for k in range(num_rings):
        if k == 0:
            ring_counts.append(1)
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n / np.pi)))
            count = max(1, count)
            ring_counts.append(count)
    
    # Adjust to match total
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

def _compute_score(points, idx_i, idx_j):
    """Compute the true packing score."""
    diff = points[idx_i] - points[idx_j]
    d_ij = np.linalg.norm(diff, axis=1)
    min_d_ij = np.min(d_ij) / 2.0
    
    r_i = np.linalg.norm(points, axis=1)
    min_d_bound = np.min(1.0 - r_i)
    
    return np.min([min_d_ij, min_d_bound])

def _make_objective(idx_i, idx_j, beta):
    """Create objective function and gradient for soft-min optimization."""
    def objective_and_grad(x):
        x = x.reshape(-1, 2)
        diff = x[idx_i] - x[idx_j]
        d_ij = np.linalg.norm(diff, axis=1)
        d_ij_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
        
        r_i = np.linalg.norm(x, axis=1)
        d_bound = 1.0 - r_i
        
        y_pairs = d_ij / 2.0
        y_bound = d_bound
        all_y = np.concatenate([y_pairs, y_bound])
        y_min = np.min(all_y)
        
        # Use log-sum-exp trick for numerical stability
        z = -beta * all_y
        max_z = np.max(z)
        exp_terms = np.exp(z - max_z)
        sum_exp = np.sum(exp_terms)
        
        log_sum_exp = max_z + np.log(sum_exp)
        obj_val = log_sum_exp / beta
        
        weights = exp_terms / sum_exp
        
        grad = np.zeros_like(x)
        w_pairs = weights[:len(y_pairs)]
        terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
        np.add.at(grad, idx_i, -terms)
        np.add.at(grad, idx_j, terms)
        
        w_bound = weights[len(y_pairs):]
        r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
        grad_bound = -x / r_i_safe[:, np.newaxis]
        grad -= (w_bound[:, np.newaxis] * grad_bound)
        
        return obj_val, grad
    
    return objective_and_grad

def _local_search(points, idx_i, idx_j, time_budget):
    """Perform local search by moving the most constrained point."""
    best_x = points.copy()
    best_score = _compute_score(best_x, idx_i, idx_j)
    
    # Find the most constrained point (smallest margin)
    def get_most_constrained(x):
        diff = x[idx_i] - x[idx_j]
        d_ij = np.linalg.norm(diff, axis=1)
        min_d_ij = np.min(d_ij) / 2.0
        
        r_i = np.linalg.norm(x, axis=1)
        d_bound = 1.0 - r_i
        
        # Find which point is most constrained
        margins = np.zeros(N)
        for i in range(N):
            # Distance to all other points
            dists = np.linalg.norm(x[i] - x, axis=1)
            dists[i] = np.inf
            min_pair = np.min(dists) / 2.0
            min_margin = min(min_pair, 1.0 - np.linalg.norm(x[i]))
            margins[i] = min_margin
        return np.argmin(margins)
    
    start_time = time.time()
    step_size = 0.01
    
    for iteration in range(500):
        if time.time() - start_time > time_budget:
            break
        
        # Get the most constrained point
        pt_idx = get_most_constrained(best_x)
        
        # Try small perturbations in random directions
        improved = False
        for _ in range(20):
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
            
            # Try several step sizes
            for ss in [step_size, step_size * 0.5, step_size * 2]:
                candidate = best_x.copy()
                candidate[pt_idx] += direction * ss
                
                # Check if valid (all points inside unit circle)
                if np.any(np.linalg.norm(candidate, axis=1) > 1.0):
                    continue
                
                score = _compute_score(candidate, idx_i, idx_j)
                if score > best_score:
                    best_x = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    idx_i, idx_j = np.triu_indices(N, 1)
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
    best_x = None
    best_score = -np.inf
    
    for init_points in inits:
        score = _compute_score(init_points, idx_i, idx_j)
        if score > best_score:
            best_x = init_points.copy()
            best_score = score
    
    if best_x is None:
        best_x = _hex_lattice_init(N)
        best_score = _compute_score(best_x, idx_i, idx_j)
    
    # Multi-stage optimization with increasing beta
    beta_schedule = [50, 100, 200, 500, 1000]
    
    for beta in beta_schedule:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15:
            break
        
        obj_func = _make_objective(idx_i, idx_j, beta)
        
        try:
            res = minimize(
                obj_func, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 60, 'gtol': 1e-9}
            )
            
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate_x)):
                    score = _compute_score(candidate_x, idx_i, idx_j)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    # Local search to fine-tune
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        time_budget = min(30, time_limit - elapsed - 5)
        if time_budget > 5:
            try:
                best_x, best_score = _local_search(best_x, idx_i, idx_j, time_budget)
            except Exception:
                pass
    
    # Final polish with very high beta if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        try:
            obj_func_polish = _make_objective(idx_i, idx_j, 5000)
            res = minimize(
                obj_func_polish, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 30, 'gtol': 1e-11}
            )
            if res.x is not None:
                candidate_x = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate_x)):
                    score = _compute_score(candidate_x, idx_i, idx_j)
                    if score > best_score:
                        best_x = candidate_x
                        best_score = score
        except Exception:
            pass
    
    return best_x
