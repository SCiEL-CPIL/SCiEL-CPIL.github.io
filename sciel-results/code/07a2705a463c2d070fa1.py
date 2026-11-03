"""Circle packing in unit square using penalty-annealed L-BFGS-B optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 135.0  # safety margin before 150s kill
_START_TIME = 0.0

def _elapsed():
    return time.time() - _START_TIME

def _objective(params, penalty_weight):
    """
    Objective function: minimize -sum(r) + penalty * (boundary_violations + overlap_violations).
    params: array of shape (3*N,) where each triple is (x, y, r)
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: maximize sum of radii => minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for out-of-bounds
    # Distance to boundary for circle i is min(x, 1-x, y, 1-y)
    # Violation is max(0, r - dist_to_boundary)
    dist_to_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    boundary_violation = np.maximum(0.0, r - dist_to_boundary)
    obj += penalty_weight * np.sum(boundary_violation**2)
    
    # Penalty for overlaps
    # Vectorized distance calculation is faster for N=26
    # Create pairwise difference matrices
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Sum of radii for each pair
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Overlap violation: max(0, r_i + r_j - d_ij)
    # We only need to look at upper triangle to avoid double counting, 
    # but summing all and dividing by 2 is equivalent and vectorized.
    overlap_violation = np.maximum(0.0, r_sum - dist)
    # Zero out the diagonal (distance 0, r_i+r_i - 0 > 0, but self-overlap is not a constraint)
    np.fill_diagonal(overlap_violation, 0.0)
    
    obj += penalty_weight * 0.5 * np.sum(overlap_violation**2)
    
    return obj

def _gradient(params, penalty_weight):
    """
    Numerical gradient of the penalized objective.
    """
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, penalty_weight)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus, penalty_weight)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def _is_valid(params, tol=1e-5):
    """Check if the configuration is valid."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Check radii positive
    if np.any(r < -tol):
        return False
    
    # Check boundaries
    dist_to_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r > dist_to_boundary + tol):
        return False
    
    # Check overlaps
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    if np.any(r_sum > dist + tol):
        # Exclude self
        np.fill_diagonal(r_sum, 0)
        np.fill_diagonal(dist, np.inf)
        if np.any(r_sum > dist + tol):
            return False
            
    return True

def _grid_init(seed=42):
    """Create a grid initialization with small random perturbations."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    params = np.zeros(3 * N)
    
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
            y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
            r = 0.45 / side + rng.uniform(-0.01, 0.01)
            
            # Clamp to valid range
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = np.clip(r, 0.01, 0.25)
            
            params[3*count] = x
            params[3*count+1] = y
            params[3*count+2] = r
            count += 1
    
    return params

def _fix_to_boundary(params):
    """Fix any circles that are out of bounds by projecting them back."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    for i in range(N):
        # Ensure radius is positive
        r[i] = max(r[i], 0.001)
        
        # Ensure circle is within bounds
        if x[i] - r[i] < 0:
            x[i] = r[i]
        if x[i] + r[i] > 1:
            x[i] = 1 - r[i]
        if y[i] - r[i] < 0:
            y[i] = r[i]
        if y[i] + r[i] > 1:
            y[i] = 1 - r[i]
        
        # Ensure radius doesn't exceed bounds
        max_r = min(x[i], 1-x[i], y[i], 1-y[i])
        if r[i] > max_r:
            r[i] = max_r
        r[i] = max(r[i], 0.001)
    
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    return params

def _optimize_single(params_init, max_iter=300, time_limit=20.0):
    """
    Run L-BFGS-B optimization from a single initial point with penalty annealing.
    """
    start = time.time()
    
    # Bounds for each parameter
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.45))   # r
    
    current_params = params_init.copy()
    
    # Penalty annealing schedule
    # Start with low penalty to allow movement, end with high penalty for strictness
    penalty_weights = [1.0, 10.0, 100.0, 1000.0, 5000.0]
    
    for pw in penalty_weights:
        if time.time() - start > time_limit:
            break
            
        # If we are close to the deadline for this single restart, stop
        # The outer loop will handle overall deadline
        
        try:
            result = minimize(
                _objective,
                current_params,
                args=(pw,),
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={
                    'maxiter': max_iter // len(penalty_weights), # Spread iterations
                    'ftol': 1e-8,
                    'gtol': 1e-6,
                    'maxls': 20
                }
            )
            current_params = result.x
        except Exception:
            pass
            
    return current_params

def _construct_optimized():
    """
    Main construction using penalty-annealed L-BFGS-B optimization.
    Multiple random restarts.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple random initializations
    n_restarts = 8
    rng = np.random.default_rng(2024)
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Grid initialization
            params_init = _grid_init(seed=42 + restart)
        elif restart == 1:
            # Corner-heavy initialization
            params_init = np.zeros(3 * N)
            corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
            for i, (cx, cy) in enumerate(corners):
                params_init[3*i] = cx
                params_init[3*i+1] = cy
                params_init[3*i+2] = 0.15
            # Fill rest with random
            for i in range(4, N):
                params_init[3*i] = rng.uniform(0.1, 0.9)
                params_init[3*i+1] = rng.uniform(0.1, 0.9)
                params_init[3*i+2] = rng.uniform(0.02, 0.1)
        elif restart == 2:
            # Hexagonal-ish packing
            params_init = np.zeros(3 * N)
            row = 0
            col = 0
            spacing = 0.25
            for i in range(N):
                if col == 4:
                    col = 0
                    row += 1
                x = 0.15 + col * spacing + (0.05 if row % 2 == 1 else 0)
                y = 0.15 + row * spacing * 0.866
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                params_init[3*i] = x
                params_init[3*i+1] = y
                params_init[3*i+2] = 0.1
                col += 1
        else:
            # Pure random
            params_init = np.zeros(3 * N)
            for i in range(N):
                params_init[3*i] = rng.uniform(0.1, 0.9)
                params_init[3*i+1] = rng.uniform(0.1, 0.9)
                params_init[3*i+2] = rng.uniform(0.02, 0.12)
        
        # Fix to boundary
        params_init = _fix_to_boundary(params_init)
        
        # Run optimization with time limit
        time_remaining = _DEADLINE - _elapsed() - 5
        # Allocate time based on remaining restarts
        time_per_restart = max(5.0, time_remaining / (n_restarts - restart))
        
        if time_per_restart < 2:
            break
        
        params_opt = _optimize_single(params_init, max_iter=200, time_limit=time_per_restart)
        params_opt = _fix_to_boundary(params_opt)
        
        # Evaluate: We want to maximize sum of radii, but only if valid.
        # The objective minimizes -sum(r) + penalty.
        # If penalty is high, a valid solution will have low penalty.
        # We check validity explicitly.
        
        if _is_valid(params_opt, tol=1e-5):
            score = np.sum(params_opt[2::3])
            if score > best_score:
                best_score = score
                best_params = params_opt.copy()
        
        # Even if invalid, we might use it as a starting point for next iteration?
        # No, we keep best valid.
        
    return best_params

def _construct_grid_fallback():
    """Fallback grid construction."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N], dtype=np.float64)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    try:
        params = _construct_optimized()
        
        if params is None or not _is_valid(params, tol=1e-4):
            # Try to fix
            if params is not None:
                params = _fix_to_boundary(params)
            if params is None or not _is_valid(params, tol=1e-4):
                # Reduce radii slightly
                if params is not None:
                    r = params[2::3]
                    r = r * 0.9
                    params[2::3] = r
                    params = _fix_to_boundary(params)
                if params is None or not _is_valid(params, tol=1e-4):
                    raise ValueError("Invalid after fixing")
        
        result = params.reshape(N, 3)
        
        # Final check
        if not _is_valid(result.flatten(), tol=1e-5):
            raise ValueError("Final validation failed")
            
    except Exception:
        result = _construct_grid_fallback()
    
    return result
# EVOLVE-BLOCK-END
