"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_init(n, offset=0.0):
    """Generate hexagonal packing initialization for n circles."""
    points = []
    # Create hex grid
    for q in range(-12, 13):
        for r in range(-12, 13):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            points.append([x + offset, y + offset])
    points = np.array(points)
    
    # Center
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    
    # Select n closest to center
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    # Initial radii based on local spacing
    radii = np.full(n, 0.08)
    return init_points, radii

def _random_init(n, seed=0):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    points = rng.rand(n, 2)
    radii = rng.uniform(0.05, 0.1, n)
    return points, radii

def _compute_penalty_and_grad(vars, C=1000.0):
    """Compute penalty value and gradient."""
    grad = np.zeros(3 * N)
    penalty = 0.0
    
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq)
            r_sum = r[i] + r[j]
            
            if d < 1e-9:
                dx = 1e-9
                d = 1e-9
            
            if d < r_sum:
                V = r_sum - d
                penalty += C * V * V
                coeff = 2.0 * C * V
                inv_d = 1.0 / d
                
                grad[3*i + 0] += coeff * (-dx * inv_d)
                grad[3*i + 1] += coeff * (-dy * inv_d)
                grad[3*i + 2] += coeff * 1.0
                
                grad[3*j + 0] += coeff * (dx * inv_d)
                grad[3*j + 1] += coeff * (dy * inv_d)
                grad[3*j + 2] += coeff * 1.0
    
    # Boundary penalties
    for i in range(N):
        if x[i] < r[i]:
            V = r[i] - x[i]
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
            
        if x[i] + r[i] > 1.0:
            V = x[i] + r[i] - 1.0
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
            
        if y[i] < r[i]:
            V = r[i] - y[i]
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
            
        if y[i] + r[i] > 1.0:
            V = y[i] + r[i] - 1.0
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
    
    return penalty, grad

def _is_valid(x, y, r, tol=1e-6):
    """Check if configuration is valid."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _optimize(vars0, start_time, deadline, C=1000.0):
    """Run gradient descent with momentum."""
    vars = vars0.copy()
    velocity = np.zeros_like(vars)
    best_val = -np.sum(vars0[2::3])
    best_vars = vars0.copy()
    
    lr = 0.005
    momentum = 0.9
    steps = 3000
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        # Update with momentum
        velocity = momentum * velocity - lr * g
        vars += velocity
        
        # Projection
        vars[2::3] = np.maximum(vars[2::3], 0.01)
        vars[0::3] = np.clip(vars[0::3], 0.0, 1.0)
        vars[1::3] = np.clip(vars[1::3], 0.0, 1.0)
        
        # Track best
        current_sum = np.sum(vars[2::3])
        if current_sum > best_val:
            best_val = current_sum
            best_vars = vars.copy()
        
        # Decay learning rate
        lr *= 0.998
    
    return best_vars, best_val

def _polish(vars, start_time, deadline):
    """High-precision polish pass."""
    if time.time() > deadline:
        return vars
    
    C = 5000.0  # Higher penalty for tight constraints
    lr = 0.001
    momentum = 0.9
    steps = 1000
    
    vars = vars.copy()
    velocity = np.zeros_like(vars)
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        velocity = momentum * velocity - lr * g
        vars += velocity
        
        vars[2::3] = np.maximum(vars[2::3], 0.01)
        vars[0::3] = np.clip(vars[0::3], 0.0, 1.0)
        vars[1::3] = np.clip(vars[1::3], 0.0, 1.0)
        
        lr *= 0.999
    
    return vars

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_vars = None
    best_sum = 0.0
    
    # Try multiple initializations
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Hexagonal initialization
            points, radii = _hex_init(N)
        elif restart == 1:
            # Slightly offset hex
            points, radii = _hex_init(N, offset=0.05)
        else:
            # Random initialization
            points, radii = _random_init(N, seed=restart * 42 + 13)
        
        # Build initial variables
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        
        # Optimize
        vars_opt, val = _optimize(x0, start_time, deadline)
        
        if val > best_sum:
            best_sum = val
            best_vars = vars_opt.copy()
    
    if best_vars is None:
        # Fallback
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
        r_best = r_best * 0.95
        if not _is_valid(x_best, y_best, r_best):
            # Fallback
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    
    result = np.column_stack((x_best, y_best, r_best))
    return result

# EVOLVE-BLOCK-END
