"""Improved sphere packing: 100 spheres in unit cube."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 100
DEADLINE = 145.0  # safety margin before 150s kill
_start_time = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _compute_r(centres):
    """Compute the valid radius for a configuration."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check inside cube
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Wall distances
    wall_dists = np.min(np.abs(centres), axis=1)  # min over k of |c_ik|
    wall_r = 0.5 - np.max(np.abs(centres), axis=1)  # min over k of (0.5 - |c_ik|)
    # Pairwise distances
    d = pdist(centres)
    min_pair = np.min(d) if len(d) > 0 else np.inf
    r_pair = min_pair / 2.0
    r_wall = np.min(wall_r)
    return min(r_pair, r_wall)

def _soft_min_logsumexp(values, beta=50.0):
    """Approximate min via -log(sum(exp(-beta*x)))/beta, large beta -> min."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    # Numerically stable
    shifted = -beta * values
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(log_sum + np.max(shifted)) / beta

def _objective_soft(centres, beta=50.0):
    """Smooth surrogate for -r (to minimize). Returns negative of soft-min radius."""
    centres = centres.reshape(N, 3)
    # Pairwise distances
    d = pdist(centres)
    r_pairs = d / 2.0
    # Wall distances
    wall_r = 0.5 - np.max(np.abs(centres), axis=1)
    all_vals = np.concatenate([r_pairs, wall_r])
    soft_min = _soft_min_logsumexp(all_vals, beta)
    return -soft_min

def _objective_exact(centres):
    """Exact objective: -r (to minimize)."""
    centres = centres.reshape(N, 3)
    r = _compute_r(centres)
    return -r

def _initial_grid_554():
    """5x5x4 grid initialization."""
    nx, ny, nz = 5, 5, 4
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    return grid[:N]

def _initial_bcc_like():
    """Body-centered cubic inspired initialization."""
    # Use a 4x4x4 grid with some body-center points
    # Actually, let's use a perturbed 5x5x4 grid
    base = _initial_grid_554()
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=base.shape)
    return base + noise

def _initial_random():
    """Random initialization inside cube."""
    rng = np.random.default_rng(123)
    return rng.uniform(-0.4, 0.4, size=(N, 3))

def _optimize_from(centres, beta_start=20.0, beta_end=100.0, n_iters=500):
    """Optimize starting from centres, gradually increasing beta."""
    current = centres.copy()
    best_r = _compute_r(current)
    
    # Gradually increase beta for smoother optimization
    for it in range(n_iters):
        if _time_left() < 5.0:
            break
        # Interpolate beta
        t = it / max(n_iters - 1, 1)
        beta = beta_start + (beta_end - beta_start) * t
        
        # Use soft objective for optimization
        res = minimize(
            _objective_soft,
            current,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * 3 * N,
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        new_centres = res.x.reshape(N, 3)
        new_r = _compute_r(new_centres)
        
        if new_r > best_r:
            best_r = new_r
            current = new_centres
    
    return current, best_r

def _polish(centres, n_iters=100):
    """Final high-precision polish using exact objective with small steps."""
    current = centres.copy()
    best_r = _compute_r(current)
    
    # Use very high beta soft-min for near-exact behavior
    for it in range(n_iters):
        if _time_left() < 2.0:
            break
        res = minimize(
            _objective_soft,
            current,
            args=(200.0,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * 3 * N,
            options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        new_centres = res.x.reshape(N, 3)
        new_r = _compute_r(new_centres)
        
        if new_r > best_r:
            best_r = new_r
            current = new_centres
        else:
            # Try smaller perturbation
            break
    
    return current, best_r

def construct():
    global _start_time
    _start_time = time.time()
    
    # Try multiple initializations and keep the best
    initializers = [
        _initial_grid_554,
        _initial_bcc_like,
    ]
    
    best_centres = None
    best_r = 0.0
    
    for init_func in initializers:
        if _time_left() < 10.0:
            break
        
        centres = init_func()
        r0 = _compute_r(centres)
        
        # Optimize
        centres_opt, r_opt = _optimize_from(centres)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centres = centres_opt
    
    # Polish the best result
    if best_centres is not None and _time_left() > 5.0:
        best_centres, best_r = _polish(best_centres)
    
    if best_centres is None:
        # Fallback to simple grid
        best_centres = _initial_grid_554()
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_centres
