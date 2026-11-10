import numpy as np
from scipy.spatial.distance import pdist
import time
import scipy.optimize as so

def _compute_score(points):
    """Compute R = (d_min / d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _soft_min_logsumexp(vals, beta=100.0):
    """Approximate min via log-sum-exp (soft min)."""
    # For soft min: -log(sum(exp(-beta*x)))/beta
    scaled = -beta * vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def _soft_max_logsumexp(vals, beta=100.0):
    """Approximate max via log-sum-exp (soft max)."""
    scaled = beta * vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum / beta

def _objective_smooth(points, beta=100.0):
    """Smooth objective: maximize soft_min(d) / soft_max(d)."""
    d = pdist(points)
    soft_min = _soft_min_logsumexp(d, beta)
    soft_max = _soft_max_logsumexp(d, beta)
    if soft_max < 1e-12:
        return 0.0
    return soft_min / soft_max

def _gradient_smooth(points, beta=100.0):
    """Gradient of smooth objective w.r.t. points."""
    n = len(points)
    d = pdist(points)
    
    # Soft min and max
    soft_min = _soft_min_logsumexp(d, beta)
    soft_max = _soft_max_logsumexp(d, beta)
    
    if soft_max < 1e-12:
        return np.zeros(2*n)
    
    obj = soft_min / soft_max
    
    # Gradient of soft_min w.r.t. d
    scaled_min = -beta * d
    max_scaled_min = np.max(scaled_min)
    exp_min = np.exp(scaled_min - max_scaled_min)
    sum_exp_min = np.sum(exp_min)
    grad_soft_min_d = exp_min / (sum_exp_min * beta)  # d(soft_min)/d(d_i)
    
    # Gradient of soft_max w.r.t. d
    scaled_max = beta * d
    max_scaled_max = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled_max)
    sum_exp_max = np.sum(exp_max)
    grad_soft_max_d = exp_max / (sum_exp_max * beta)  # d(soft_max)/d(d_i)
    
    # d(obj)/d(d_i) = (grad_soft_min_d * soft_max - soft_min * grad_soft_max_d) / soft_max^2
    grad_obj_d = (grad_soft_min_d * soft_max - soft_min * grad_soft_max_d) / (soft_max ** 2)
    
    # Now chain rule: d(d_ij)/d(x_k)
    # d_ij = ||x_i - x_j||
    # d(d_ij)/d(x_k) = (x_k - x_j)/d_ij if k==i, (x_i - x_k)/d_ij if k==j, 0 otherwise
    
    # Build pair index mapping
    pair_idx = []
    for i in range(n):
        for j in range(i+1, n):
            pair_idx.append((i, j))
    
    grad_points = np.zeros(2*n)
    for k, (i, j) in enumerate(pair_idx):
        if d[k] < 1e-12:
            continue
        grad_d_k = grad_obj_d[k]
        # d(d_k)/d(x_i) = (x_i - x_j) / d_k
        diff_ij = points[i] - points[j]
        grad_points[2*i] += grad_d_k * diff_ij[0] / d[k]
        grad_points[2*i+1] += grad_d_k * diff_ij[1] / d[k]
        # d(d_k)/d(x_j) = (x_j - x_i) / d_k
        grad_points[2*j] -= grad_d_k * diff_ij[0] / d[k]
        grad_points[2*j+1] -= grad_d_k * diff_ij[1] / d[k]
    
    return grad_points

def _optimize_smooth(points, max_iter=200, tol=1e-10, beta_schedule=None):
    """
    Optimize using smooth surrogate with annealing of beta.
    """
    n = len(points)
    x0 = points.flatten()
    
    if beta_schedule is None:
        beta_schedule = [50, 100, 200, 500, 1000, 2000]
    
    best_x = x0.copy()
    best_obj = _objective_smooth(points, beta_schedule[0])
    
    for beta in beta_schedule:
        def obj(v):
            pts = v.reshape(n, 2)
            return -_objective_smooth(pts, beta)
        
        def jac(v):
            pts = v.reshape(n, 2)
            return -_gradient_smooth(pts, beta)
        
        try:
            res = so.minimize(obj, x0, method='L-BFGS-B', jac=jac, 
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
            if res.success or res.fun < best_obj:
                x0 = res.x
                obj_val = -res.fun
                if obj_val > best_obj:
                    best_obj = obj_val
                    best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(n, 2), best_obj

def _fast_local_search(points, time_budget=5.0, seed=42):
    """
    Fast local search using coordinate descent with adaptive step sizes.
    """
    n = len(points)
    rng = np.random.RandomState(seed)
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    step_size = 0.05
    
    while time.time() - start_time < time_budget:
        # Try moving each point in random direction
        improved = False
        for idx in range(n):
            for _ in range(3):  # Try 3 random directions per point
                delta = rng.randn(2) * step_size
                new_points = current_points.copy()
                new_points[idx] += delta
                new_score = _compute_score(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                    break
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
        else:
            step_size = min(step_size * 1.1, 0.1)
    
    return best_points, best_score

def _polish(points, max_iter=500, tol=1e-12):
    """
    High-precision polish using SLSQP with epigraph formulation.
    """
    n = len(points)
    x0 = points.flatten()
    
    def objective(v):
        # v[0:2n] are points, v[2n] is r
        return -v[2*n]
        
    def jac(v):
        grad = np.zeros(2*n + 1)
        grad[2*n] = -1.0
        return grad

    constraints = []
    
    # Constraint: all pairwise distances >= r
    def min_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        r = v[2*n]
        d = pdist(pts)
        return d**2 - r**2

    cons_min_dist = {'type': 'ineq', 'fun': min_dist_constraint}
    constraints.append(cons_min_dist)
    
    # Constraint: all pairwise distances <= 1 (normalize)
    def max_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        d = pdist(pts)
        return 1.0 - d**2

    cons_max_dist = {'type': 'ineq', 'fun': max_dist_constraint}
    constraints.append(cons_max_dist)
    
    bounds = [(-5, 5)] * (2*n) + [(1e-8, 1.0)]
    
    # Initial r estimate
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        normalized_points = points / d_max
        d_norm = pdist(normalized_points)
        r0 = d_norm.min() * 0.9
    else:
        normalized_points = points.copy()
        r0 = 1e-4
    
    v0 = np.append(normalized_points.flatten(), r0)
    
    try:
        res = so.minimize(objective, v0, method='SLSQP', jac=jac, bounds=bounds, 
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x[:2*n].reshape(n, 2), res.x[2*n]
        else:
            return normalized_points, d_norm.min()
    except Exception:
        return normalized_points, d_norm.min()

def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)
    
    # 7. Random with larger spread
    candidates.append(rng.randn(n, 2) * 2.0)

    # 8. Fibonacci sphere projection to 2D (hexagonal packing)
    golden = (1 + np.sqrt(5)) / 2
    angles = np.arange(n) * 2 * np.pi / golden
    radii = np.sqrt(np.arange(n) / n)
    fib = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(fib)

    # 9. Two interleaved squares
    sq1_angles = 2 * np.pi * np.arange(8) / 8
    sq1 = np.column_stack([np.cos(sq1_angles), np.sin(sq1_angles)])
    sq2_angles = 2 * np.pi * np.arange(8) / 8 + np.pi / 4
    sq2 = 0.7 * np.column_stack([np.cos(sq2_angles), np.sin(sq2_angles)])
    candidates.append(np.vstack([sq1, sq2]))

    return candidates

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin before 630s kill
    
    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)
    
    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)
    
    # Phase 1: Smooth optimization from each initial configuration
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - time.time() - 20.0
        if remaining < 10.0:
            break
        
        try:
            # Smooth optimization with annealing
            smooth_pts, smooth_obj = _optimize_smooth(init_pts, max_iter=100, tol=1e-8)
            smooth_score = _compute_score(smooth_pts)
            
            if smooth_score > best_score:
                best_points = smooth_pts
                best_score = smooth_score
            
            # Fast local search refinement
            if time.time() < deadline - 25.0:
                ls_pts, ls_score = _fast_local_search(best_points, time_budget=min(5.0, deadline - time.time() - 20.0), seed=i)
                if ls_score > best_score:
                    best_points = ls_pts
                    best_score = ls_score
        except Exception:
            pass

    # Phase 2: High-precision polish
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        try:
            polish_pts, polish_r = _polish(best_points, max_iter=300, tol=1e-10)
            polish_score = _compute_score(polish_pts)
            if polish_score > best_score:
                best_points = polish_pts
                best_score = polish_score
        except Exception:
            pass

    # Phase 3: Final smooth refinement if time permits
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        try:
            final_pts, final_obj = _optimize_smooth(best_points, max_iter=50, tol=1e-10, beta_schedule=[1000, 2000, 5000])
            final_score = _compute_score(final_pts)
            if final_score > best_score:
                best_points = final_pts
                best_score = final_score
        except Exception:
            pass

    return best_points
