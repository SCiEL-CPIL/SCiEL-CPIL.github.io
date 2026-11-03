"""
Pack 200 equal circles in a unit square using gradient-based optimization
with a soft-min objective, combined with targeted perturbation to escape
local minima.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_HARD_LIMIT = 145  # Hard wall-clock limit with safety margin (evaluator kills at 150s)

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    """
    if len(pts) == 0:
        return 0.0
    
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _objective_and_grad(x, beta):
    """
    Objective function for minimization.
    We want to MAXIMIZE the soft-min of constraints.
    So we MINIMIZE -soft_min.
    """
    pts = x.reshape((N, 2))
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    C = np.empty(4*N + num_pairs)
    
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0
    
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    weights = exp_shifted / sum_exp
    
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
    
    # Pair gradients
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _generate_hex_initial(spacing=None):
    """
    Generate a hexagonal lattice packing.
    """
    if spacing is None:
        spacing = 0.0732
    
    s = spacing
    points = []
    row_h = s * np.sqrt(3) / 2
    
    # Generate a grid larger than the square
    y = -0.6
    row_idx = 0
    while y < 0.6:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.6 + shift
        while x < 0.6:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    
    # Filter points inside the square
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        # Try smaller spacing
        s = 0.070
        points = []
        row_h = s * np.sqrt(3) / 2
        y = -0.6
        row_idx = 0
        while y < 0.6:
            shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
            x = -0.6 + shift
            while x < 0.6:
                points.append([x, y])
                x += s
            y += row_h
            row_idx += 1
        pts = np.array(points)
        mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
               (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
        pts = pts[mask]
        
    if len(pts) >= N:
        # Select N points closest to the center
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts = pts[indices[:N]]
        return pts
    else:
        return np.random.uniform(-0.4, 0.4, (N, 2))

def _force_directed_phase(pts, time_limit, max_iter=2000):
    """
    Force-directed repulsion to spread out overlapping circles.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    
    step = 0.002
    lr_decay = 0.9995
    
    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break
        
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        threshold = 0.08
        close_mask = D < threshold
        
        for i in range(N):
            close_j = np.where(close_mask[i] & (np.arange(N) > i))[0]
            for j in close_j:
                d = D[i, j]
                if d < 1e-10:
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                else:
                    diff = pts[i] - pts[j]
                    force_mag = 1.0 / (d * d + 1e-6)
                    forces[i] += (diff / d) * force_mag
                    forces[j] -= (diff / d) * force_mag
        
        # Boundary repulsion
        margin = 0.05
        close_left = pts[:, 0] < -0.5 + margin
        if np.any(close_left):
            forces[close_left, 0] += (margin - (pts[close_left, 0] + 0.5)) * 2.0
        close_right = pts[:, 0] > 0.5 - margin
        if np.any(close_right):
            forces[close_right, 0] -= (pts[close_right, 0] - (0.5 - margin)) * 2.0
        close_bottom = pts[:, 1] < -0.5 + margin
        if np.any(close_bottom):
            forces[close_bottom, 1] += (margin - (pts[close_bottom, 1] + 0.5)) * 2.0
        close_top = pts[:, 1] > 0.5 - margin
        if np.any(close_top):
            forces[close_top, 1] -= (pts[close_top, 1] - (0.5 - margin)) * 2.0
        
        pts += step * forces
        pts = np.clip(pts, -0.5, 0.5)
        
        step *= lr_decay
        
        r = _get_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_r, best_pts

def _perturb_and_optimize(pts, t_start, time_budget, rng):
    """
    Perturb the configuration by moving the most constrained points,
    then run gradient optimization.
    """
    r_curr = _get_radius(pts)
    
    # Find the most constrained pairs
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    
    # Find pairs that are limiting the radius
    limiting_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if D[i, j] / 2.0 <= r_curr + 1e-9:
                limiting_pairs.append((i, j, D[i, j]))
    
    # Also check walls
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if len(limiting_pairs) == 0:
        # No limiting pairs, try random perturbation
        pts_new = pts.copy()
        idx = rng.integers(0, N, size=5)
        pts_new[idx] += rng.normal(0, 0.005, (5, 2))
        pts_new = np.clip(pts_new, -0.5, 0.5)
    else:
        # Perturb points involved in limiting pairs
        pts_new = pts.copy()
        involved = set()
        for i, j, d in limiting_pairs[:10]:  # Top 10 most limiting
            involved.add(i)
            involved.add(j)
        
        if len(involved) == 0:
            involved = {0}
        
        for idx in involved:
            pts_new[idx] += rng.normal(0, 0.003, 2)
        
        pts_new = np.clip(pts_new, -0.5, 0.5)
    
    # Run gradient optimization on the perturbed configuration
    r_opt, pts_opt = _run_gradient_opt(pts_new, t_start, time_budget)
    
    if r_opt > r_curr:
        return r_opt, pts_opt
    else:
        return r_curr, pts

def _run_gradient_opt(pts, t_start, time_budget):
    """
    Run gradient-based optimization with increasing beta.
    """
    if time.time() - t_start > time_budget:
        return _get_radius(pts), pts
    
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    # Force-directed phase to fix initial overlaps
    fd_time = min(3.0, time_budget * 0.1)
    if fd_time > 0.5:
        r_fd, pts_fd = _force_directed_phase(pts, time_limit=fd_time)
        if r_fd > best_r:
            best_r = r_fd
            best_x = pts_fd.flatten()
            x0 = pts_fd.flatten()
    
    # Gradient optimization with increasing beta
    betas = [20, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        elapsed = time.time() - t_start
        if elapsed > time_budget - 2.0:
            break
        
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
        
        try:
            maxiter = 100 if beta < 100 else 200
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
    
    return best_r, best_x.reshape((N, 2))

def construct():
    t0 = time.time()
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_r = 0.0
    best_pts = None
    
    # Generate initial hexagonal configuration
    pts_hex = _generate_hex_initial(spacing=0.0732)
    
    # Initial force-directed phase
    fd_time = min(5.0, _HARD_LIMIT * 0.05)
    r_fd, pts_fd = _force_directed_phase(pts_hex, time_limit=fd_time)
    
    if r_fd > best_r:
        best_r = r_fd
        best_pts = pts_fd
    
    # Main optimization loop with perturbations
    # We'll run several rounds of: optimize -> perturb -> optimize
    num_rounds = 8
    time_per_round = (_HARD_LIMIT - 10) / num_rounds
    
    for round_idx in range(num_rounds):
        elapsed = time.time() - t0
        remaining = _HARD_LIMIT - 5 - elapsed
        if remaining < time_per_round:
            break
        
        # Run gradient optimization
        r_opt, pts_opt = _run_gradient_opt(best_pts, t0, time_per_round * 0.6)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt
        
        # Perturb and try to escape local minimum
        elapsed = time.time() - t0
        remaining = _HARD_LIMIT - 5 - elapsed
        if remaining > time_per_round * 0.4:
            r_pert, pts_pert = _perturb_and_optimize(
                best_pts, t0, remaining * 0.8, rng
            )
            if r_pert > best_r:
                best_r = r_pert
                best_pts = pts_pert
    
    # Final high-precision polish
    elapsed = time.time() - t0
    remaining = _HARD_LIMIT - 3 - elapsed
    if remaining > 5:
        betas_polish = [500, 1000, 2000]
        x0 = best_pts.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        for beta in betas_polish:
            if time.time() - t0 > _HARD_LIMIT - 3:
                break
            
            def fun_and_grad(x):
                val, g = _objective_and_grad(x, beta)
                return val, g
            
            try:
                res = minimize(
                    fun_and_grad, 
                    x0, 
                    jac=True, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 100}
                )
                x0 = res.x
                pts_curr = x0.reshape((N, 2))
                pts_curr = np.clip(pts_curr, -0.5, 0.5)
                
                r = _get_radius(pts_curr)
                if r > best_r:
                    best_r = r
                    best_pts = pts_curr
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
