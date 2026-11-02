"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in [-0.5,0.5]^3."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Wall distances
    wall = 0.5 - np.abs(c).min(axis=0)
    min_wall = wall.min()
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask out self
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    return min(min_wall, min_pair)

def _objective(c):
    """Negative radius (to minimize)."""
    return -_compute_radius(c)

def _initial_grid():
    """Create an initial grid of N points in the cube."""
    # Use a 4x4x4 grid = 64 points, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(float)

def _jittered_start(seed):
    """Add small jitter to grid for diversity."""
    rng = np.random.RandomState(seed)
    c = _initial_grid().copy()
    c += rng.uniform(-0.01, 0.01, size=c.shape)
    # Clip to stay inside cube with margin
    c = np.clip(c, -0.49, 0.49)
    return c

def _anneal(c, max_iter=500, T0=0.05, T_end=0.001, alpha=0.999):
    """Simple simulated annealing to improve radius."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    T = T0
    for it in range(max_iter):
        if T < T_end:
            break
        # Random perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, T, size=3)
        c_new = c.copy()
        c_new[idx] += delta
        # Keep inside cube
        c_new[idx] = np.clip(c_new[idx], -0.499, 0.499)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        # Accept with probability
        if r_new >= _compute_radius(c) or np.random.rand() < np.exp((r_new - _compute_radius(c)) / T):
            c = c_new
        T *= alpha
    return best_c, best_r

def _polish(c):
    """Use scipy minimize to polish the configuration."""
    # Objective: maximize min radius = minimize -min radius
    # Use Nelder-Mead or L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    c_flat = c.flatten()
    
    result = minimize(
        lambda x: _objective(x.reshape(N, 3)),
        c_flat,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    c_opt = result.x.reshape(N, 3)
    # Ensure validity
    c_opt = np.clip(c_opt, -0.4999, 0.4999)
    return c_opt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_c = None
    best_r = 0.0
    
    # Start with grid
    c0 = _initial_grid()
    r0 = _compute_radius(c0)
    best_c = c0.copy()
    best_r = r0
    
    # Try several annealing runs with different seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        c_jit = _jittered_start(seed)
        c_annealed, r_annealed = _anneal(c_jit, max_iter=300, T0=0.03, T_end=0.0005, alpha=0.998)
        if r_annealed > best_r:
            best_r = r_annealed
            best_c = c_annealed.copy()
    
    # Polish the best configuration
    if time.time() < deadline - 30:
        c_polished = _polish(best_c)
        r_polished = _compute_radius(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Final check
    r_final = _compute_radius(best_c)
    if r_final <= 0:
        # Fallback to grid
        best_c = _initial_grid()
    
    return best_c
# EVOLVE-BLOCK-END
