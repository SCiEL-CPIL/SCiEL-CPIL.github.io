"""Packing 50 equal circles in a unit square using SLSQP with epigraph formulation."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
SEED_RADIUS = 0.071377103865  # known best (Boll et al. 2000)
DEADLINE_FRACTION = 0.85  # safety margin


def _radius(c):
    """Compute the packing radius for centre array c of shape (N, 2)."""
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    # Distance to walls: min over i,k of (1/2 - |c_ik|)
    wall = 0.5 - np.max(np.abs(c), axis=1)
    r_wall = np.min(wall)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    return float(min(r_wall, r_pair))


def _initial_grid():
    """Generate a reasonable initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_jittered(seed):
    """Generate jittered initial positions."""
    rng = np.random.RandomState(seed)
    c = _initial_grid()
    # Add small random perturbations
    c += rng.uniform(-0.01, 0.01, size=c.shape)
    # Clip to stay inside
    c = np.clip(c, -0.49, 0.49)
    return c


def _slsqp_solve(c0, time_limit):
    """
    Solve the max-radius problem using SLSQP with an epigraph formulation.
    
    Variables: x = [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, t]  (2N+1 variables)
    Maximize t  =>  Minimize -t
    
    Constraints:
      - For each pair (i,j): |c_i - c_j| >= 2t  (nonlinear, 2 constraints per pair)
        We use: (c_i - c_j)^2 - (2t)^2 >= 0  =>  (2t)^2 - (c_i - c_j)^2 <= 0
      - For each point and each wall: 1/2 - |c_ik| >= t
        => t + |c_ik| - 1/2 <= 0  (4 constraints per point, but |.| is non-smooth)
        Better: use c_ik + t - 1/2 <= 0 and -c_ik + t - 1/2 <= 0
    
    To handle the pairwise constraints smoothly, we can use:
      g_ij(x) = (2t)^2 - ||c_i - c_j||^2 <= 0
    
    For wall constraints (smooth):
      h_ik(x) = c_ik + t - 0.5 <= 0
      h_ik(x) = -c_ik + t - 0.5 <= 0
    """
    n_vars = 2 * N + 1
    start_time = time.time()
    
    def unpack(x):
        c = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        return c, t
    
    def objective(x):
        return -x[2 * N]  # maximize t
    
    def jac_objective(x):
        g = np.zeros(n_vars)
        g[2 * N] = -1.0
        return g
    
    # Precompute pair indices
    pair_idx = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_idx.append((i, j))
    
    def constraints(x):
        c, t = unpack(x)
        cons = []
        # Pairwise: (2t)^2 - ||c_i - c_j||^2 <= 0
        for i, j in pair_idx:
            diff = c[i] - c[j]
            val = (2.0 * t) ** 2 - np.dot(diff, diff)
            cons.append(val)
        # Wall constraints: c_ik + t - 0.5 <= 0 and -c_ik + t - 0.5 <= 0
        for i in range(N):
            for k in range(2):
                cons.append(c[i, k] + t - 0.5)
                cons.append(-c[i, k] + t - 0.5)
        return np.array(cons)
    
    # Initial point
    c0_arr, t0 = unpack(c0.flatten())
    t_init = _radius(c0)
    x0 = np.concatenate([c0.flatten(), [t_init]])
    
    # Bounds: positions in [-0.5, 0.5], t in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    # Use SLSQP
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=jac_objective,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success or hasattr(res, 'x'):
            c_res = res.x[:2 * N].reshape(N, 2)
            # Clip to ensure validity
            c_res = np.clip(c_res, -0.5, 0.5)
            r_res = _radius(c_res)
            if r_res > 0:
                return c_res, r_res
    except Exception:
        pass
    
    return c0, _radius(c0)


def construct():
    deadline = time.time() + 120.0 * DEADLINE_FRACTION
    
    best_c = _initial_grid()
    best_r = _radius(best_c)
    
    # Try a few different initializations and SLSQP runs
    for attempt in range(8):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            c0 = _initial_grid()
        else:
            c0 = _initial_jittered(seed=42 + attempt)
        
        # Give more time to earlier attempts
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        time_limit = min(remaining * 0.9, 15.0)
        
        c_new, r_new = _slsqp_solve(c0, time_limit)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
    
    # Final polish: one more SLSQP run from the best solution with tighter tolerance
    if time.time() < deadline:
        c_polish, r_polish = _slsqp_solve(best_c, deadline - time.time())
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish.copy()
    
    # Ensure final validity
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
