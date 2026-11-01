"""Combined circle packing solver for n=32 in unit square."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hex_init(n, offset=0.0, rng_seed=0):
    """Generate hexagonal packing initialization for n circles."""
    rng = np.random.RandomState(rng_seed)
    points = []
    # Create hex grid with some randomness in selection
    for q in range(-12, 13):
        for r in range(-12, 13):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            # Add small jitter
            x += rng.uniform(-0.1, 0.1)
            y += rng.uniform(-0.1, 0.1)
            points.append([x + offset, y + offset])
    points = np.array(points)
    
    # Center it
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    
    # Select n points closest to center
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    # Initial radii: estimate based on local spacing
    radii = np.full(n, 0.08)
    return init_points, radii

def _random_init(n, seed=0):
    """Generate random initialization with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    points = []
    radii = []
    
    # Try to place circles randomly without overlap
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = rng.uniform(0.05, 0.12)
        
        # Check against existing circles
        ok = True
        for i in range(len(points)):
            dx = x - points[i][0]
            dy = y - points[i][1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r + radii[i] + 0.01:
                ok = False
                break
        
        if ok:
            points.append([x, y])
            radii.append(r)
    
    # If we couldn't place all, use grid fallback for remaining
    while len(points) < n:
        side = int(np.ceil(np.sqrt(len(points) + 1)))
        i = len(points)
        px = (i % side + 0.5) / side
        py = (i // side + 0.5) / side
        points.append([px, py])
        radii.append(0.08)
    
    return np.array(points), np.array(radii)

def _compute_penalty_and_grad(vars, C=1000.0):
    """Compute penalty value and gradient for circle packing."""
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
            
            if d < 1e-12:
                dx = 1e-12
                dy = 0.0
                d = 1e-12
            
            if d < r_sum:
                V = r_sum - d
                penalty += C * V * V
                coeff = 2.0 * C * V
                inv_d = 1.0 / d
                
                # Gradients for circle i
                grad[3*i + 0] += coeff * (-dx * inv_d)
                grad[3*i + 1] += coeff * (-dy * inv_d)
                grad[3*i + 2] += coeff * 1.0
                
                # Gradients for circle j
                grad[3*j + 0] += coeff * (dx * inv_d)
                grad[3*j + 1] += coeff * (dy * inv_d)
                grad[3*j + 2] += coeff * 1.0
    
    # Boundary penalties
    for i in range(N):
        # Left boundary
        if x[i] < r[i]:
            V = r[i] - x[i]
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
        
        # Right boundary
        if x[i] + r[i] > 1.0:
            V = x[i] + r[i] - 1.0
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 0] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
        
        # Bottom boundary
        if y[i] < r[i]:
            V = r[i] - y[i]
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * (-1.0)
            grad[3*i + 2] += coeff * 1.0
        
        # Top boundary
        if y[i] + r[i] > 1.0:
            V = y[i] + r[i] - 1.0
            penalty += C * V * V
            coeff = 2.0 * C * V
            grad[3*i + 1] += coeff * 1.0
            grad[3*i + 2] += coeff * 1.0
    
    return penalty, grad

def _is_valid(x, y, r, tol=1e-7):
    """Check if configuration is valid (no overlaps, inside box)."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _optimize(vars0, start_time, deadline, C=1000.0, lr_init=0.005, momentum=0.9):
    """Run gradient descent with momentum and adaptive learning rate."""
    vars = vars0.copy()
    velocity = np.zeros_like(vars)
    best_val = np.sum(vars0[2::3])
    best_vars = vars0.copy()
    best_penalty = _compute_penalty_and_grad(vars0, C)[0]
    
    lr = lr_init
    steps = 5000
    patience = 200
    no_improve = 0
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        # Update with momentum
        velocity = momentum * velocity - lr * g
        vars = vars + velocity
        
        # Projection: ensure radii positive and positions in [0,1]
        vars[2::3] = np.maximum(vars[2::3], 0.005)
        vars[0::3] = np.clip(vars[0::3], 0.0, 1.0)
        vars[1::3] = np.clip(vars[1::3], 0.0, 1.0)
        
        # Evaluate
        current_sum = np.sum(vars[2::3])
        current_penalty = penalty
        
        # Track best (prioritize low penalty, then high sum)
        if current_penalty < best_penalty - 1e-6 or (abs(current_penalty - best_penalty) < 1e-6 and current_sum > best_val):
            best_val = current_sum
            best_vars = vars.copy()
            best_penalty = current_penalty
            no_improve = 0
        else:
            no_improve += 1
        
        # Decay learning rate
        lr *= 0.999
        
        # Early stopping if stuck
        if no_improve > patience and step > 500:
            break
    
    return best_vars, best_val, best_penalty

def _polish(vars, start_time, deadline):
    """High-precision polish pass with very small steps."""
    if time.time() > deadline:
        return vars
    
    C = 10000.0  # Higher penalty for tight constraints
    lr = 0.0005
    momentum = 0.8
    steps = 2000
    
    vars = vars.copy()
    velocity = np.zeros_like(vars)
    best_vars = vars.copy()
    best_sum = np.sum(vars[2::3])
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        penalty, g = _compute_penalty_and_grad(vars, C)
        
        velocity = momentum * velocity - lr * g
        vars = vars + velocity
        
        vars[2::3] = np.maximum(vars[2::3], 0.005)
        vars[0::3] = np.clip(vars[0::3], 0.0, 1.0)
        vars[1::3] = np.clip(vars[1::3], 0.0, 1.0)
        
        current_sum = np.sum(vars[2::3])
        if current_sum > best_sum:
            best_sum = current_sum
            best_vars = vars.copy()
        
        lr *= 0.9995
    
    return best_vars

def _local_search(vars, start_time, deadline, num_trials=50):
    """Simple local search: try small random perturbations and keep if better."""
    vars = vars.copy()
    best_vars = vars.copy()
    best_sum = np.sum(vars[2::3])
    best_penalty = _compute_penalty_and_grad(vars, 1000.0)[0]
    
    rng = np.random.RandomState(42)
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Random perturbation
        perturbation = rng.normal(0, 0.01, size=3*N)
        # Only perturb positions, not radii
        perturbation[2::3] *= 0.5
        
        trial_vars = vars + perturbation
        trial_vars[2::3] = np.maximum(trial_vars[2::3], 0.005)
        trial_vars[0::3] = np.clip(trial_vars[0::3], 0.0, 1.0)
        trial_vars[1::3] = np.clip(trial_vars[1::3], 0.0, 1.0)
        
        trial_penalty, _ = _compute_penalty_and_grad(trial_vars, 1000.0)
        trial_sum = np.sum(trial_vars[2::3])
        
        # Accept if penalty is lower or same penalty with higher sum
        if trial_penalty < best_penalty - 1e-6 or (abs(trial_penalty - best_penalty) < 1e-6 and trial_sum > best_sum):
            vars = trial_vars
            best_vars = trial_vars.copy()
            best_sum = trial_sum
            best_penalty = trial_penalty
    
    return best_vars

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    best_vars = None
    best_sum = 0.0
    best_penalty = float('inf')
    
    # Multiple restarts with different initializations
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Hexagonal grid
            points, radii = _hex_init(N, offset=0.0, rng_seed=0)
        elif restart == 1:
            # Hexagonal with offset
            points, radii = _hex_init(N, offset=0.03, rng_seed=1)
        elif restart == 2:
            # Random initialization
            points, radii = _random_init(N, seed=100)
        elif restart == 3:
            # Another random
            points, radii = _random_init(N, seed=200)
        elif restart == 4:
            # Hex with different seed
            points, radii = _hex_init(N, offset=-0.02, rng_seed=2)
        else:
            # Random with different seed
            points, radii = _random_init(N, seed=300)
        
        # Build initial variables
        x0 = np.zeros(3 * N)
        x0[0::3] = points[:, 0]
        x0[1::3] = points[:, 1]
        x0[2::3] = radii
        
        # Optimize
        vars_opt, val, penalty = _optimize(x0, start_time, deadline)
        
        if penalty < best_penalty or (abs(penalty - best_penalty) < 1e-6 and val > best_sum):
            best_sum = val
            best_vars = vars_opt.copy()
            best_penalty = penalty
    
    if best_vars is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Polish the best result
    if time.time() < deadline - 10:
        best_vars = _polish(best_vars, start_time, deadline)
    
    # Local search if time permits
    if time.time() < deadline - 5:
        best_vars = _local_search(best_vars, start_time, deadline)
    
    # Final validation and repair
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    r_best = best_vars[2::3]
    
    # Try to fix any slight violations
    if not _is_valid(x_best, y_best, r_best, tol=1e-6):
        # Slightly reduce radii
        r_best = r_best * 0.98
        if not _is_valid(x_best, y_best, r_best, tol=1e-6):
            r_best = r_best * 0.95
            if not _is_valid(x_best, y_best, r_best, tol=1e-6):
                # Fallback to grid
                side = int(np.ceil(np.sqrt(N)))
                cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
                return np.array(cells[:N])
    
    result = np.column_stack((x_best, y_best, r_best))
    return result

# EVOLVE-BLOCK-END
