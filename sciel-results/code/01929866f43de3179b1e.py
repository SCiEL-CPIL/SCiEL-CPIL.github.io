"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import random


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Vectorized distance check
    # Create distance matrix
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    rad_sum = radii[:, None] + radii[None, :]
    
    # Check upper triangle
    iu = np.triu_indices(n, k=1)
    if np.any(dist_mat[iu] < rad_sum[iu] - tol):
        return False
    return True


def _objective_and_grad(x, n, lam_ov=1000.0, lam_bd=1000.0):
    """
    Objective function and gradient for L-BFGS-B.
    x is a flat vector of length 3*n: [x0, y0, r0, x1, y1, r1, ...]
    """
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    loss = 0.0
    grad = np.zeros_like(x)
    
    # 1. Boundary constraints (soft penalty)
    # x < r -> penalty (r-x)^2
    # x > 1-r -> penalty (x-(1-r))^2
    # Same for y
    
    # Lower bound violations
    v_x_low = np.maximum(0, radii - pos[:, 0])
    v_y_low = np.maximum(0, radii - pos[:, 1])
    # Upper bound violations
    v_x_high = np.maximum(0, pos[:, 0] - (1 - radii))
    v_y_high = np.maximum(0, pos[:, 1] - (1 - radii))
    
    loss += lam_bd * (np.sum(v_x_low**2) + np.sum(v_y_low**2) + np.sum(v_x_high**2) + np.sum(v_y_high**2))
    
    # Gradients for boundary
    # d/dx (r-x)^2 = -2(r-x) if x<r else 0
    # d/dx (x-(1-r))^2 = 2(x-(1-r)) if x>1-r else 0
    grad[:n, 0] += lam_bd * (-2 * v_x_low + 2 * v_x_high)
    grad[n:2*n, 0] += lam_bd * (-2 * v_y_low + 2 * v_y_high)
    # d/dr (r-x)^2 = 2(r-x) if x<r else 0
    # d/dr (x-(1-r))^2 = -2(x-(1-r)) if x>1-r else 0
    grad[2*n:] += lam_bd * (2 * v_x_low + 2 * v_y_low - 2 * v_x_high - 2 * v_y_high)
    
    # 2. Overlap constraints (soft penalty)
    # For each pair i < j
    # if dist < r_i + r_j, penalty (r_i + r_j - dist)^2
    # We can vectorize this partially or use a loop for clarity given N=26 is small
    # N=26 -> 325 pairs. Loop is fine.
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                viol = min_dist - dist
                loss += lam_ov * viol * viol
                
                if dist > 1e-12:
                    nx = dx / dist
                    ny = dy / dist
                    # Gradient w.r.t pos_i: -2 * lam_ov * viol * (-nx, -ny) ? No.
                    # viol = (r_i + r_j) - dist
                    # d(viol)/d(pos_i) = -d(dist)/d(pos_i) = -(-nx, -ny) = (nx, ny) ??
                    # dist = ||p_j - p_i||. d(dist)/d(p_i) = -(p_j - p_i)/dist = -n_vec.
                    # d(viol)/d(p_i) = -d(dist)/d(p_i) = n_vec.
                    # d(loss)/d(p_i) = 2 * lam_ov * viol * n_vec.
                    # d(viol)/d(p_j) = -d(dist)/d(p_j) = -n_vec.
                    # d(loss)/d(p_j) = 2 * lam_ov * viol * (-n_vec).
                    
                    grad[i, 0] += 2 * lam_ov * viol * nx
                    grad[i, 1] += 2 * lam_ov * viol * ny
                    grad[j, 0] -= 2 * lam_ov * viol * nx
                    grad[j, 1] -= 2 * lam_ov * viol * ny
                    
                # Gradient w.r.t radii
                # d(viol)/d(r_i) = 1
                # d(loss)/d(r_i) = 2 * lam_ov * viol
                grad[2*n + i] += 2 * lam_ov * viol
                grad[2*n + j] += 2 * lam_ov * viol
    
    # 3. Objective: Minimize Loss - Sum(Radii)
    # We want to MAXIMIZE sum(radii), so we MINIMIZE -sum(radii).
    loss -= np.sum(radii)
    grad[2*n:] -= 1.0
    
    return loss, grad


def _run_lbfgs(pos, radii, time_budget=10.0, start_time=None):
    """Run L-BFGS-B optimization."""
    if start_time is None:
        start_time = time.time()
        
    n = len(pos)
    x0 = np.concatenate([pos.flatten(), radii.flatten()])
    
    def fun(x):
        loss, _ = _objective_and_grad(x, n)
        return loss
        
    def grad(x):
        _, g = _objective_and_grad(x, n)
        return g
        
    # Bounds:
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in unit square is 0.5 if single circle, but with packing smaller)
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-6, 0.49)) # r
    
    res = scipy.optimize.minimize(fun, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000, 'disp': False})
    
    x_opt = res.x
    pos_opt = x_opt[:2*n].reshape(n, 2)
    radii_opt = x_opt[2*n:]
    
    # Ensure validity and clamp
    if _check_validity(pos_opt, radii_opt):
        return pos_opt, radii_opt
    else:
        # If invalid, try to repair or return original if better
        return pos, radii


def _sa_search(pos, radii, time_budget=20.0, start_time=None):
    """
    Simulated Annealing to explore configuration space.
    """
    if start_time is None:
        start_time = time.time()
        
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    best_score = -np.inf
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    T = 0.05 # Initial temperature
    cooling = 0.99
    min_T = 0.001
    
    rng = random.Random(42)
    
    def _current_score(p, r):
        # Validity check is expensive, use penalty for SA acceptance
        # But we want to track best VALID score
        # Let's use a quick penalty score for acceptance, and exact sum for best tracking
        loss, _ = _objective_and_grad(np.concatenate([p.flatten(), r.flatten()]), n)
        return loss
        
    # Initial penalty
    current_penalty = _current_score(pos, radii)
    
    # We want to minimize penalty.
    # SA accepts worse (higher penalty) with prob exp(-delta/T)
    
    while time.time() - start_time < time_budget:
        if T < min_T:
            break
            
        # Pick a random circle to perturb
        i = rng.randint(0, n - 1)
        
        # Perturb position
        dx = rng.gauss(0, 0.05)
        dy = rng.gauss(0, 0.05)
        
        # Perturb radius
        dr = rng.gauss(0, 0.01)
        
        new_pos = pos.copy()
        new_radii = radii.copy()
        
        new_pos[i, 0] += dx
        new_pos[i, 1] += dy
        new_radii[i] += dr
        
        # Clamp to bounds
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0, 1)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0, 1)
        new_radii[i] = np.clip(new_radii[i], 1e-6, 0.49)
        
        new_penalty = _current_score(new_pos, new_radii)
        
        delta = new_penalty - current_penalty
        
        # Accept if better (delta < 0) or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            pos = new_pos
            radii = new_radii
            current_penalty = new_penalty
            
            # Check validity and update best
            if _check_validity(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    
        T *= cooling
        
    return best_pos, best_radii


def _build_hex(n):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.17
            y = (row + 0.5) * 0.149
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    return pos, radii


def _build_grid(n):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=0):
    """Build a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Multiple initial configurations
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy()))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy()))
    
    for seed in [42, 123, 456, 789, 101, 202]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy()))
        
    # Process each configuration
    for ci, (pos, radii) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        remaining = deadline - elapsed - 5
        
        # Phase 1: Simulated Annealing to explore
        if remaining > 15:
            time_alloc_sa = min(15.0, remaining * 0.6)
            pos, radii = _sa_search(pos, radii, time_budget=time_alloc_sa, start_time=start_time)
        
        # Phase 2: L-BFGS-B to polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 5:
            time_alloc_lbfgs = min(10.0, remaining * 0.8)
            pos, radii = _run_lbfgs(pos, radii, time_budget=time_alloc_lbfgs, start_time=start_time)
            
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
        # If we have time, do a second pass on the best found so far
        if best_pos is not None and ci > 0:
             elapsed = time.time() - start_time
             remaining = deadline - elapsed - 2
             if remaining > 10:
                 # Restart SA from best
                 p2, r2 = _sa_search(best_pos, best_radii, time_budget=min(10.0, remaining*0.5), start_time=start_time)
                 p2, r2 = _run_lbfgs(p2, r2, time_budget=min(5.0, remaining*0.4), start_time=start_time)
                 if _check_validity(p2, r2):
                     score2 = np.sum(r2)
                     if score2 > best_score:
                         best_score = score2
                         best_pos = p2.copy()
                         best_radii = r2.copy()
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
