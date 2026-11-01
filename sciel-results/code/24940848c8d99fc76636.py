"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
_DEADLINE = 140.0  # seconds wall-clock (safety margin below 150s)
_start_time = None

def _elapsed():
    return time.time() - _start_time

def _compute_radius(centers):
    """Compute the common radius for a set of centers inside [-0.5,0.5]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    # Wall constraint: min distance from center to any face
    wall = 0.5 - np.abs(centers).max(axis=0)
    r_wall = wall.min()
    # Pair constraint
    # For N=50, compute all pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    return min(r_wall, r_pair)

def _soft_objective(centers, target_r, beta):
    """
    Smooth surrogate: sum of exp(beta * violation) for each constraint.
    violation = max(0, target_r - distance/2) for pairs,
               max(0, target_r - wall_dist) for walls.
    Lower is better; 0 means all constraints satisfied.
    """
    centers = np.asarray(centers, dtype=np.float64)
    # Wall violations
    wall_dist = 0.5 - np.abs(centers).max(axis=0)
    wall_viol = np.maximum(0.0, target_r - wall_dist)
    # Pair violations
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, target_r - dist / 2.0)
    
    # Soft sum via log-sum-exp for numerical stability
    all_viol = np.concatenate([wall_viol, pair_viol.ravel()])
    if all_viol.max() == 0.0:
        return 0.0
    # Use log-sum-exp
    m = all_viol.max()
    log_sum = m * beta + np.log(np.sum(np.exp(beta * (all_viol - m))))
    # Return the actual sum of exps (not log) for gradient-friendly use
    # Actually for SA we just need a scalar cost; use sum of exps directly
    # But for large beta, exps can overflow. Use: sum(exp(beta*v)) 
    # For SA acceptance we compare costs, so relative values matter.
    # Use a safer formulation: if max violation is small, return sum; else use log
    if m * beta > 500:
        # Very infeasible; return a large number
        return 1e10
    cost = np.sum(np.exp(beta * all_viol))
    return cost

def _sa_step(centers, target_r, beta, T, rng):
    """
    One SA step: perturb one center, accept/reject based on cost.
    Returns new centers and new cost.
    """
    n = len(centers)
    i = rng.integers(n)
    new_centers = centers.copy()
    # Perturb
    step_size = 0.01
    new_centers[i] += rng.normal(0, step_size, 3)
    # Clamp to cube (soft: allow slight overshoot, penalty handles it)
    # Don't hard-clamp; let the penalty handle it
    
    old_cost = _soft_objective(centers, target_r, beta)
    new_cost = _soft_objective(new_centers, target_r, beta)
    
    if new_cost <= old_cost or rng.random() < np.exp(-(new_cost - old_cost) / T):
        return new_centers, new_cost
    return centers, old_cost

def _sa_optimize(initial_centers, target_r, max_iter, rng, beta_init=50.0, beta_final=200.0, T_init=1.0, T_final=0.01):
    """
    Simulated Annealing to find a feasible packing for target radius target_r.
    Returns best centers found and best cost.
    """
    global _start_time
    centers = initial_centers.copy()
    best_centers = centers.copy()
    best_cost = _soft_objective(centers, target_r, beta_init)
    
    beta = beta_init
    T = T_init
    
    # Precompute for speed: use a simpler cost function for SA
    # Actually, let's use a faster cost: just the max violation (nonsmooth but fast)
    # For SA, nonsmooth is fine since we use random perturbations
    
    def _fast_cost(c, tr):
        wall_dist = 0.5 - np.abs(c).max(axis=0)
        wall_viol = np.maximum(0.0, tr - wall_dist).max()
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0.0, tr - dist / 2.0).max()
        return max(wall_viol, pair_viol)
    
    current_cost = _fast_cost(centers, target_r)
    best_cost = current_cost
    best_centers = centers.copy()
    
    step_size = 0.02
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 5.0:
            break
        
        i = rng.integers(N)
        new_centers = centers.copy()
        new_centers[i] += rng.normal(0, step_size, 3)
        
        new_cost = _fast_cost(new_centers, target_r)
        
        if new_cost <= current_cost or rng.random() < np.exp(-(new_cost - current_cost) / max(T, 1e-10)):
            centers = new_centers
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_centers = centers.copy()
        
        # Cool
        T = T_init * (T_final / T_init) ** (it / max(max_iter, 1))
        # Anneal step size
        step_size = 0.02 * (0.01 / 0.02) ** (it / max(max_iter, 1))
    
    return best_centers, best_cost

def _grid_init(rng, jitter=0.0):
    """Create initial grid with optional jitter."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    if jitter > 0:
        centers += rng.normal(0, jitter, centers.shape)
    return centers

def _random_init(rng):
    """Random initialization inside the cube."""
    return rng.uniform(-0.45, 0.45, (N, 3))

def _polish(centers, target_r, rng, max_iter=5000):
    """
    High-precision polish: gradient-based local optimization.
    Maximize the minimum radius using a smooth surrogate.
    """
    global _start_time
    
    def _neg_radius_smooth(c, beta=100.0):
        """Negative of smooth min radius."""
        wall_dist = 0.5 - np.abs(c).max(axis=0)
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_r = dist / 2.0
        
        # Smooth min of all radii (wall and pair)
        all_r = np.concatenate([wall_dist, pair_r.ravel()])
        # -log-sum-exp(-beta * all_r) / beta approximates min(all_r)
        m = all_r.min()
        if m <= 0:
            return 1.0  # infeasible
        log_sum = -m * beta + np.log(np.sum(np.exp(-beta * (all_r - m))))
        return log_sum / beta
    
    # Use scipy minimize with Nelder-Mead or BFGS
    # Nelder-Mead is derivative-free and robust
    try:
        result = minimize(
            lambda c: _neg_radius_smooth(c.reshape(N, 3), beta=50.0),
            centers.reshape(-1),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        new_centers = result.x.reshape(N, 3)
        # Validate
        r = _compute_radius(new_centers)
        if r >= target_r - 1e-9:
            return new_centers
    except Exception:
        pass
    return centers

def construct():
    global _start_time
    _start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    # Known best radius
    r_best = 0.1359545126417
    
    # Start with a slightly lower target and work up
    # First, get a feasible starting point
    initial = _grid_init(rng, jitter=0.005)
    r_initial = _compute_radius(initial)
    
    # Binary search for the maximum feasible radius
    r_low = r_initial  # feasible
    r_high = r_best * 1.05  # upper bound
    
    best_centers = initial.copy()
    best_r = r_initial
    
    max_sa_iters = 8000
    n_restarts = 3
    
    # Try to improve from r_initial upward
    # First, do SA at a target slightly above current
    for target_r in [r_initial + 0.001, r_initial + 0.003, r_initial + 0.005, r_best * 0.99, r_best * 0.995, r_best]:
        if _elapsed() > _DEADLINE - 10.0:
            break
        
        found = False
        for restart in range(n_restarts):
            if _elapsed() > _DEADLINE - 10.0:
                break
            
            if restart == 0:
                init = _grid_init(rng, jitter=0.005)
            else:
                init = _random_init(rng)
            
            centers, cost = _sa_optimize(init, target_r, max_sa_iters, rng)
            
            r_found = _compute_radius(centers)
            if r_found >= target_r - 1e-9:
                found = True
                if r_found > best_r:
                    best_r = r_found
                    best_centers = centers.copy()
                break
            elif r_found > best_r:
                best_r = r_found
                best_centers = centers.copy()
        
        if found:
            r_low = target_r
        else:
            # Try a lower target
            r_high = target_r
            if r_high - r_low < 0.0005:
                break
    
    # Final polish
    if _elapsed() < _DEADLINE - 15.0:
        polished = _polish(best_centers, best_r, rng, max_iter=3000)
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best_centers = polished
    
    # Validate
    best_centers = np.asarray(best_centers, dtype=np.float64)
    if best_centers.shape != (N, 3) or not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
