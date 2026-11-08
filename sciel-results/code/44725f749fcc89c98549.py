"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    # Boundary checks
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Pairwise overlap check
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _build_grid(n, jitter=0.0):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.45 / side
    radii = np.full(n, r)
    return pos, radii


def _build_hex(n, jitter=0.0):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.55 / cols
            y = (row + 0.5) * 0.5 / (int(np.ceil(n / cols)) + 1)
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.22
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=None):
    """Build a random initial configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.08)
    return pos, radii


def _softplus(x, beta=100.0):
    """Smooth approximation of max(0, x)."""
    # Avoid overflow in exp
    x = np.clip(x, -50, 50)
    return np.log1p(np.exp(beta * x)) / beta

def _objective_and_grad(x, beta=100.0):
    """
    x: array of shape (3N,) containing [x1, y1, r1, x2, y2, r2, ...]
    Returns: (objective_value, gradient_array)
    Objective: -sum(r) + lambda * sum(softplus(violations))
    """
    N = len(x) // 3
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # 1. Main term: -sum(radii)
    obj = -np.sum(radii)
    grad = np.zeros_like(x)
    grad[2*N:] = -1.0  # d(-sum(r))/dr = -1
    
    # 2. Penalty terms
    # Boundary violations:
    # left: r - x < 0  -> violation = r - x
    # right: x - (1-r) < 0 -> violation = x - 1 + r
    # bottom: r - y < 0 -> violation = r - y
    # top: y - (1-r) < 0 -> violation = y - 1 + r
    
    # Left
    v_left = radii - pos[:, 0]
    p_left = _softplus(v_left, beta)
    obj += np.sum(p_left)
    # Grad for left: d/dx (softplus(r-x)) = -softplus'(r-x) = -exp(beta(r-x))/(1+exp(beta(r-x)))
    # Note: softplus'(x) = 1 / (1 + exp(-beta*x)) = sigmoid(beta*x)
    # So d/dx is -sigmoid(beta * v)
    # But wait, my _softplus returns log(1+exp(beta*x))/beta. 
    # Derivative is exp(beta*x)/(1+exp(beta*x)) = sigmoid(beta*x).
    # So d/dx (softplus(r-x)) = -sigmoid(beta*(r-x))
    # d/dr (softplus(r-x)) = sigmoid(beta*(r-x))
    
    # Let's compute sigmoids efficiently
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        
    # Left
    sig_left = sigmoid(beta * v_left)
    grad[:N] -= sig_left  # d/dx_i
    grad[2*N:] += sig_left # d/dr_i
    
    # Right
    v_right = pos[:, 0] - (1.0 - radii)
    p_right = _softplus(v_right, beta)
    obj += np.sum(p_right)
    sig_right = sigmoid(beta * v_right)
    grad[:N] += sig_right # d/dx_i
    grad[2*N:] -= sig_right # d/dr_i
    
    # Bottom
    v_bottom = radii - pos[:, 1]
    p_bottom = _softplus(v_bottom, beta)
    obj += np.sum(p_bottom)
    sig_bottom = sigmoid(beta * v_bottom)
    grad[N:2*N] -= sig_bottom # d/dy_i
    grad[2*N:] += sig_bottom # d/dr_i
    
    # Top
    v_top = pos[:, 1] - (1.0 - radii)
    p_top = _softplus(v_top, beta)
    obj += np.sum(p_top)
    sig_top = sigmoid(beta * v_top)
    grad[N:2*N] += sig_top # d/dy_i
    grad[2*N:] -= sig_top # d/dr_i
    
    # 3. Pairwise overlap penalties
    # violation = r_i + r_j - dist_ij
    # We need to sum softplus(v_ij) for all i < j
    
    # Vectorized pairwise distance calculation
    # pos shape (N, 2)
    # diff shape (N, N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # dist is symmetric, diagonal is 0.
    # We only want i < j.
    # Create a mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Calculate violations for all pairs
    # v_ij = r_i + r_j - dist_ij
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    violations = r_sum - dist
    
    # Zero out non-upper-triangle and diagonal
    violations = violations * mask
    
    # Sum softplus of violations
    p_pairs = np.sum(_softplus(violations, beta))
    obj += p_pairs
    
    # Gradients for pairs
    # d/dx_i softplus(v_ij) = softplus'(v_ij) * d(v_ij)/dx_i
    # v_ij = r_i + r_j - sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    # d(v_ij)/dx_i = -(x_i - x_j) / dist_ij
    # d(v_ij)/dr_i = 1
    
    sig_pairs = sigmoid(beta * violations) # Shape (N, N)
    
    # Gradient w.r.t positions
    # For each i, sum over j != i
    # grad_x_i = sum_j sig_ij * (-(x_i - x_j) / dist_ij)
    # Note: dist_ij is 0 on diagonal, but mask is 0 there, so we must avoid div by 0.
    safe_dist = np.where(dist < 1e-8, 1e-8, dist)
    
    # Term for x: sig * (-(x_i - x_j) / dist)
    # diff_x = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    # grad_x_i = sum_j sig_ij * (-diff_x_ij / dist_ij)
    
    diff_x = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    diff_y = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    
    # Weighted sum
    wx = sig_pairs * (-diff_x / safe_dist)
    wy = sig_pairs * (-diff_y / safe_dist)
    
    # Sum over j for each i
    grad_x_pairs = np.sum(wx, axis=1)
    grad_y_pairs = np.sum(wy, axis=1)
    
    # Add to gradient
    grad[:N] += grad_x_pairs
    grad[N:2*N] += grad_y_pairs
    
    # Gradient w.r.t radii
    # d(v_ij)/dr_i = 1
    # So grad_r_i = sum_j sig_ij
    grad_r_pairs = np.sum(sig_pairs, axis=1)
    grad[2*N:] += grad_r_pairs
    
    return obj, grad

def _scipy_optimize(pos, radii, time_budget=10.0, start_time=None):
    """Optimize using L-BFGS-B."""
    if start_time is None:
        start_time = time.time()
    
    N = len(pos)
    x0 = np.concatenate([pos.flatten(), radii.flatten()])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-4, 0.5)) # r
    
    best_obj = np.inf
    best_x = x0
    
    # Multi-stage beta optimization
    # Start with low beta (smooth) to find global structure, increase to sharpen
    betas = [10, 50, 200, 500]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        def func_and_grad(x):
            if time.time() - start_time > time_budget:
                return 0, np.zeros_like(x) # Abort if time exceeded, though scipy might not handle this gracefully
            obj, grad = _objective_and_grad(x, beta)
            return obj, grad
        
        try:
            res = scipy.optimize.minimize(
                func_and_grad, 
                current_x, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
                
            current_x = res.x
        except Exception as e:
            pass
    
    # Extract results
    best_pos = best_x[:2*N].reshape(N, 2)
    best_radii = best_x[2*N:]
    
    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Multiple initial configurations
    configs = []
    
    # Grid
    pos, radii = _build_grid(N, jitter=0.0)
    configs.append((pos, radii))
    
    # Grid with jitter
    pos, radii = _build_grid(N, jitter=0.02)
    configs.append((pos, radii))
    
    # Hexagonal
    pos, radii = _build_hex(N, jitter=0.0)
    configs.append((pos, radii))
    
    # Hexagonal with jitter
    pos, radii = _build_hex(N, jitter=0.02)
    configs.append((pos, radii))
    
    # Random starts
    for seed in [42, 123, 456, 789, 1011]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos, radii))
    
    # Process each configuration
    for ci, (pos, radii) in enumerate(configs):
        if time.time() - start_time > deadline:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        # Optimize
        pos, radii = _scipy_optimize(pos, radii, time_budget=min(25.0, remaining - 2.0),
                                      start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final polish if time remains and we have a best solution
    if best_pos is not None and time.time() - start_time < deadline - 5:
        remaining = deadline - (time.time() - start_time) - 2
        if remaining > 5.0:
             best_pos, best_radii = _scipy_optimize(best_pos, best_radii, 
                                               time_budget=min(10.0, remaining),
                                               start_time=start_time)
             if _check_validity(best_pos, best_radii):
                 best_score = np.sum(best_radii)
    
    # Return best result
    if best_pos is None:
        # Fallback: simple grid
        pos, radii = _build_grid(N, jitter=0.0)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
