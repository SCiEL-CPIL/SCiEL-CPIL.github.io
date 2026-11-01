"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def _score(c):
    """Compute the largest common radius for given centers."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n == 0:
        return 0.0
    
    # Check bounds strictly
    if np.any(c < -0.5) or np.any(c > 0.5):
        return -1.0

    # Vectorized pairwise distances
    # diff is (n, n, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    # min distance from center to any of the 4 walls
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    # dist to left: c[:,0] - (-0.5) = c[:,0] + 0.5
    # dist to right: 0.5 - c[:,0]
    # dist to bottom: c[:,1] + 0.5
    # dist to top: 0.5 - c[:,1]
    wall_dists = np.min(np.stack([
        c[:, 0] + 0.5, 0.5 - c[:, 0],
        c[:, 1] + 0.5, 0.5 - c[:, 1]
    ]), axis=0)
    min_wall = np.min(wall_dists)
    
    r = min(min_pair / 2.0, min_wall)
    return r


def _penalized_objective(x, r_target, penalty=100.0):
    """Penalized objective: minimize sum of violations."""
    c = x.reshape(-1, 2)
    n = len(c)
    
    # Wall violations (vectorized)
    # We want c to be in [-0.5, 0.5]
    # Violation is max(0, -0.5 - c)^2 + max(0, c - 0.5)^2
    # Using np.where or clipping
    viol_x_neg = np.clip(-0.5 - c[:, 0], 0, None)
    viol_x_pos = np.clip(c[:, 0] - 0.5, 0, None)
    viol_y_neg = np.clip(-0.5 - c[:, 1], 0, None)
    viol_y_pos = np.clip(c[:, 1] - 0.5, 0, None)
    wall_viol = np.sum(viol_x_neg**2) + np.sum(viol_x_pos**2) + np.sum(viol_y_neg**2) + np.sum(viol_y_pos**2)
    
    # Pairwise violations (vectorized)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only care about i < j to avoid double counting, but summing all is fine if we adjust penalty or just sum upper triangle
    # Sum of squared violations for pairs where dist < 2*r_target
    # v = max(0, 2*r_target - dist)^2
    violations = np.clip(2 * r_target - dists, 0, None)**2
    # Sum only upper triangle to avoid double counting
    iu = np.triu_indices(n, 1)
    pair_viol = np.sum(violations[iu])
    
    return penalty * (wall_viol + pair_viol)


def _hexagonal_init():
    """Generate initial positions in a hexagonal pattern."""
    # Simple grid initialization
    m = int(np.ceil(np.sqrt(N)))
    # Distribute points evenly
    x_coords = np.linspace(-0.48, 0.48, m)
    y_coords = np.linspace(-0.48, 0.48, m)
    grid = np.array([(x, y) for y in y_coords for x in x_coords])
    positions = grid[:N].copy()
    # Add small random perturbations to break symmetry
    positions += np.random.randn(N, 2) * 0.01
    # Clip to valid range
    positions = np.clip(positions, -0.49, 0.49)
    return positions


def _local_optimize(c, start_time, deadline):
    """Local optimization using gradient-based method with penalty."""
    c = c.copy()
    best_r = _score(c)
    if best_r <= 0:
        return c, best_r
    
    # Try to increase radius iteratively
    # We use a fixed number of small steps to keep it fast
    r_target = best_r
    for it in range(50): # Reduced from 5000
        if time.time() > deadline - 2.0:
            break
            
        # Increase target radius slightly
        r_target = best_r * 1.001 + 1e-6
        
        # Optimize positions to achieve this radius
        x0 = c.flatten()
        try:
            res = minimize(
                _penalized_objective,
                x0,
                args=(r_target, 100.0),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12} # Reduced maxiter
            )
        except Exception:
            break
            
        c_new = res.x.reshape(-1, 2)
        r_new = _score(c_new)
        
        if r_new > best_r:
            best_r = r_new
            c = c_new
        else:
            # If no improvement, try a smaller step or break
            r_target = best_r * 1.0005 + 1e-7
            try:
                res = minimize(
                    _penalized_objective,
                    x0,
                    args=(r_target, 100.0),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 50, 'ftol': 1e-12}
                )
            except Exception:
                break
            c_new = res.x.reshape(-1, 2)
            r_new = _score(c_new)
            if r_new > best_r:
                best_r = r_new
                c = c_new
            else:
                break
                
    return c, best_r


def _simulated_annealing(c, start_time, deadline):
    """Simulated annealing to escape local minima."""
    c = c.copy()
    best_c = c.copy()
    best_r = _score(c)
    
    # Run until deadline - 1s
    while time.time() < deadline - 1.0:
        # Random perturbation
        c_new = c.copy()
        idx = np.random.randint(0, N)
        # Scale step size based on current best_r to be more effective
        step_size = best_r * 0.1
        c_new[idx] += np.random.randn(2) * step_size
        
        # Keep within bounds
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _score(c_new)
        r_curr = _score(c)
        
        # Accept if better or with probability exp((r_new - r_curr) / T)
        # T is temperature, let's use a simple cooling schedule based on time
        elapsed = time.time() - start_time
        total_time = 60.0 # Assume 60s budget for SA
        T = 0.01 * np.exp(-elapsed / 20.0)
        
        if r_new >= r_curr or (T > 0 and np.random.random() < np.exp((r_new - r_curr) / T)):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
    
    return best_c, best_r


def construct():
    """Construct optimal packing of 50 equal circles in unit square."""
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin for 150s kill, 120s eval limit
    
    random.seed(42)
    np.random.seed(42)
    
    best_c = None
    best_r = -1.0
    
    # Multiple restarts with different initializations
    # Limit restarts to ensure we finish in time
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            c = _hexagonal_init()
        elif restart % 3 == 1:
            # Random initialization
            c = np.random.uniform(-0.45, 0.45, (N, 2))
        else:
            # Grid with larger perturbations
            m = int(np.ceil(np.sqrt(N)))
            x_coords = np.linspace(-0.48, 0.48, m)
            y_coords = np.linspace(-0.48, 0.48, m)
            grid = np.array([(x, y) for y in y_coords for x in x_coords])
            c = grid[:N].copy()
            c += np.random.randn(N, 2) * 0.05
            c = np.clip(c, -0.49, 0.49)
        
        # Local optimization
        c, r = _local_optimize(c, start_time, deadline)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Simulated annealing for further improvement
        if time.time() < deadline - 5.0:
            c, r = _simulated_annealing(c, start_time, deadline)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 2.0 and best_c is not None:
        c, r = _local_optimize(best_c, start_time, deadline)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        x_coords = np.linspace(-0.48, 0.48, m)
        y_coords = np.linspace(-0.48, 0.48, m)
        grid = np.array([(x, y) for y in y_coords for x in x_coords])
        best_c = grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
