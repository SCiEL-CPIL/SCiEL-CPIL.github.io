"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
# EVOLVE-BLOCK-START

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import minimize

def _hex_init(n, offset=0.0):
    """Generate hexagonal packing initialization for n circles."""
    points = []
    for q in range(-12, 13):
        for r in range(-12, 13):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            points.append([x + offset, y + offset])
    points = np.array(points)
    
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    radii = np.full(n, 0.08)
    return init_points, radii

def _random_init(n, seed=0):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    points = rng.rand(n, 2)
    radii = rng.uniform(0.05, 0.1, n)
    return points, radii

def _grid_init(n, spacing=0.18):
    """Generate a regular grid initialization with some jitter."""
    rng = np.random.RandomState(42)
    points = []
    max_coord = 1.0
    for i in range(20):
        for j in range(20):
            x = 0.1 + i * spacing
            y = 0.1 + j * spacing
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    points = np.array(points)
    # Add small jitter
    points += rng.uniform(-0.02, 0.02, points.shape)
    points = np.clip(points, 0.05, 0.95)
    
    # Select n closest to center
    center = np.array([0.5, 0.5])
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    radii = np.full(n, 0.07)
    return init_points, radii

def _compute_penalty_and_grad(vars, C=1000.0):
    """Compute penalty value and gradient (vectorized)."""
    grad = np.zeros(3 * N)
    penalty = 0.0
    
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Objective: maximize sum(r), so gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Pairwise overlap penalties (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx**2 + dy**2
    d = np.sqrt(d_sq + 1e-16)  # avoid division by zero
    r_sum = r[:, None] + r[None, :]
    
    # Only consider upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_upper = d[mask]
    r_sum_upper = r_sum[mask]
    dx_upper = dx[mask]
    dy_upper = dy[mask]
    
    # Overlap amount (positive when overlapping)
    V = r_sum_upper - d_upper
    V_pos = np.maximum(V, 0.0)
    
    if np.any(V_pos > 0):
        penalty += C * np.sum(V_pos**2)
        coeff = 2.0 * C * V_pos
        inv_d = 1.0 / d_upper
        
        # Get indices
        ii, jj = np.where(mask)
        
        # Gradient contributions
        # d/dx_i = -dx * inv_d, d/dx_j = dx * inv_d
        gx_i = coeff * (-dx_upper * inv_d)
        gy_i = coeff * (-dy_upper * inv_d)
        gr_i = coeff
        
        gx_j = coeff * (dx_upper * inv_d)
        gy_j = coeff * (dy_upper * inv_d)
        gr_j = coeff
        
        # Accumulate using np.add.at for duplicate indices
        np.add.at(grad, 3*ii, gx_i)
        np.add.at(grad, 3*ii+1, gy_i)
        np.add.at(grad, 3*ii+2, gr_i)
        np.add.at(grad, 3*jj, gx_j)
        np.add.at(grad, 3*jj+1, gy_j)
        np.add.at(grad, 3*jj+2, gr_j)
    
    # Boundary penalties
    # Left boundary: x < r
    V_left = np.maximum(r - x, 0.0)
    if np.any(V_left > 0):
        penalty += C * np.sum(V_left**2)
        coeff = 2.0 * C * V_left
        grad[0::3] -= coeff
        grad[2::3] += coeff
    
    # Right boundary: x + r > 1
    V_right = np.maximum(x + r - 1.0, 0.0)
    if np.any(V_right > 0):
        penalty += C * np.sum(V_right**2)
        coeff = 2.0 * C * V_right
        grad[0::3] += coeff
        grad[2::3] += coeff
    
    # Bottom boundary: y < r
    V_bottom = np.maximum(r - y, 0.0)
    if np.any(V_bottom > 0):
        penalty += C * np.sum(V_bottom**2)
        coeff = 2.0 * C * V_bottom
        grad[1::3] -= coeff
        grad[2::3] += coeff
    
    # Top boundary: y + r > 1
    V_top = np.maximum(y + r - 1.0, 0.0)
    if np.any(V_top > 0):
        penalty += C * np.sum(V_top**2)
        coeff = 2.0 * C * V_top
        grad[1::3] += coeff
        grad[2::3] += coeff
    
    return penalty, grad

def _is_valid(x, y, r, tol=1e-6):
    """Check if configuration is valid."""
    if np.any(x < r - tol) or np.any(x + r > 1.0 + tol):
        return False
    if np.any(y < r - tol) or np.any(y + r > 1.0 + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[mask] < r_sum[mask] - tol):
        return False
    return True

def _optimize(vars0, start_time, deadline, C=1000.0):
    """Run gradient descent with momentum and adaptive learning rate."""
    vars = vars0.copy()
    velocity = np.zeros_like(vars)
    best_val = np.sum(vars0[2::3])
    best_vars = vars0.copy()
    
    lr = 0.005
    momentum = 0.9
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() > deadline - 20:  # Leave time for polish
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        # Update with momentum
        velocity = momentum * velocity - lr * g
        new_vars = vars + velocity
        
        # Projection
        new_vars[2::3] = np.maximum(new_vars[2::3], 0.01)
        new_vars[0::3] = np.clip(new_vars[0::3], 0.0, 1.0)
        new_vars[1::3] = np.clip(new_vars[1::3], 0.0, 1.0)
        
        current_sum = np.sum(new_vars[2::3])
        
        # Accept if sum improved or penalty decreased
        if current_sum > best_val:
            best_val = current_sum
            best_vars = new_vars.copy()
            vars = new_vars.copy()
            # Slightly increase LR if making good progress
            if step % 100 == 0 and step > 0:
                lr = min(lr * 1.05, 0.01)
        else:
            # Reduce LR if not improving
            lr *= 0.95
            # Keep old vars
            velocity *= 0.5
        
        # Decay learning rate gradually
        lr *= 0.9995
    
    return best_vars, best_val

def _lbfgs_polish(vars, start_time, deadline, C=200.0):
    """L-BFGS-B polish for high-precision local optimization."""
    if time.time() > deadline - 5:
        return vars
    
    def objective(v):
        p, _ = _compute_penalty_and_grad(v, C)
        # We want to maximize sum(r), minimize penalty
        # So objective = penalty - sum(r)
        return p - np.sum(v[2::3])
    
    def gradient(v):
        _, g = _compute_penalty_and_grad(v, C)
        # Gradient of objective = gradient of penalty - gradient of sum(r)
        # But our _compute_penalty_and_grad already includes -1 for radius
        # So g is the gradient of (penalty - sum(r))
        return g
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5))  # r
    
    try:
        result = minimize(
            objective, vars, method='L-BFGS-B', jac=gradient,
            bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success and np.sum(result.x[2::3]) > np.sum(vars[2::3]):
            return result.x
    except Exception:
        pass
    
    return vars

def _polish(vars, start_time, deadline):
    """High-precision polish pass with gradient descent then L-BFGS-B."""
    if time.time() > deadline:
        return vars
    
    # First: fine gradient descent
    C = 5000.0
    lr = 0.001
    momentum = 0.9
    steps = 500
    
    vars = vars.copy()
    velocity = np.zeros_like(vars)
    
    for step in range(steps):
        if time.time() > deadline - 10:  # Leave time for L-BFGS-B
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        velocity = momentum * velocity - lr * g
        vars += velocity
        
        vars[2::3] = np.maximum(vars[2::3], 0.01)
        vars[0::3] = np.clip(vars[0::3], 0.0, 1.0)
        vars[1::3] = np.clip(vars[1::3], 0.0, 1.0)
        
        lr *= 0.999
    
    # Then: L-BFGS-B polish
    vars = _lbfgs_polish(vars, start_time, deadline)
    
    return vars

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    best_vars = None
    best_sum = 0.0
    
    # Try multiple initializations
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 25:  # Leave time for polish
            break
        
        if restart == 0:
            points, radii = _hex_init(N)
        elif restart == 1:
            points, radii = _hex_init(N, offset=0.05)
        elif restart == 2:
            points, radii = _grid_init(N)
        else:
            points, radii = _random_init(N, seed=restart * 42 + 13)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        
        # Use adaptive C: start moderate, increase if overlaps persist
        C = 1000.0
        if restart > 5:
            C = 2000.0
        
        vars_opt, val = _optimize(x0, start_time, deadline, C)
        
        if val > best_sum:
            best_sum = val
            best_vars = vars_opt.copy()
    
    if best_vars is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Polish the best result
    best_vars = _polish(best_vars, start_time, deadline)
    
    # Final validation and repair
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    r_best = best_vars[2::3]
    
    if not _is_valid(x_best, y_best, r_best):
        # Try to fix by slightly reducing radii
        for factor in [0.98, 0.95, 0.90, 0.85]:
            r_test = r_best * factor
            if _is_valid(x_best, y_best, r_test):
                r_best = r_test
                break
        else:
            # Fallback
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    
    result = np.column_stack((x_best, y_best, r_best))
    return result

# EVOLVE-BLOCK-END
