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
    # Check bounds
    if np.any(c < -0.5 + 1e-12) or np.any(c > 0.5 - 1e-12):
        return -1.0
    # Pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((c[i, 0] - c[j, 0])**2 + (c[i, 1] - c[j, 1])**2)
            dists[i, j] = d
            dists[j, i] = d
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5
    ]), axis=0)
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return r


def _penalized_objective(x, r_target, penalty=100.0):
    """Penalized objective: minimize sum of violations."""
    c = x.reshape(-1, 2)
    n = len(c)
    # Wall violations
    wall_viol = 0.0
    for i in range(n):
        for k in range(2):
            if c[i, k] < -0.5:
                wall_viol += (-0.5 - c[i, k])**2
            elif c[i, k] > 0.5:
                wall_viol += (c[i, k] - 0.5)**2
    # Pairwise violations
    pair_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((c[i, 0] - c[j, 0])**2 + (c[i, 1] - c[j, 1])**2)
            if d < 2 * r_target:
                pair_viol += (2 * r_target - d)**2
    return penalty * (wall_viol + pair_viol)


def _hexagonal_init():
    """Generate initial positions in a hexagonal pattern."""
    # Start with a grid and add some randomness
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    positions = grid[:N].copy()
    # Add small random perturbations
    positions += np.random.randn(N, 2) * 0.01
    # Clip to valid range
    positions = np.clip(positions, -0.49, 0.49)
    return positions


def _local_optimize(c, max_iter=5000, tol=1e-10):
    """Local optimization using gradient-based method with penalty."""
    c = c.copy()
    best_r = _score(c)
    
    # Try to increase radius iteratively
    r_target = best_r
    for it in range(max_iter):
        # Increase target radius slightly
        r_target = best_r * 1.001 + 1e-6
        
        # Optimize positions to achieve this radius
        x0 = c.flatten()
        res = minimize(
            _penalized_objective,
            x0,
            args=(r_target, 100.0),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        c_new = res.x.reshape(-1, 2)
        r_new = _score(c_new)
        
        if r_new > best_r:
            best_r = r_new
            c = c_new
        else:
            # Try with a smaller increase
            r_target = best_r * 1.0005 + 1e-7
            res = minimize(
                _penalized_objective,
                x0,
                args=(r_target, 100.0),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            c_new = res.x.reshape(-1, 2)
            r_new = _score(c_new)
            if r_new > best_r:
                best_r = r_new
                c = c_new
            else:
                break
                
    return c, best_r


def _simulated_annealing(c, start_temp=0.05, end_temp=1e-6, iterations=20000):
    """Simulated annealing to escape local minima."""
    c = c.copy()
    best_c = c.copy()
    best_r = _score(c)
    
    for it in range(iterations):
        # Annealing schedule
        t = start_temp * (end_temp / start_temp) ** (it / iterations)
        
        # Random perturbation
        c_new = c.copy()
        idx = np.random.randint(0, N)
        c_new[idx] += np.random.randn(2) * t * 0.5
        
        # Keep within bounds
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _score(c_new)
        
        # Accept if better or with probability exp((r_new - r) / t)
        if r_new >= _score(c) or np.random.random() < np.exp((r_new - _score(c)) / max(t, 1e-10)):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
    
    return best_c, best_r


def construct():
    """Construct optimal packing of 50 equal circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_c = None
    best_r = -1.0
    
    # Multiple restarts with different initializations
    for restart in range(50):
        if time.time() > deadline - 20.0:
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
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            c = grid[:N].copy()
            c += np.random.randn(N, 2) * 0.05
            c = np.clip(c, -0.49, 0.49)
        
        # Local optimization
        c, r = _local_optimize(c, max_iter=2000)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Simulated annealing for further improvement
        if time.time() < deadline - 10.0:
            c, r = _simulated_annealing(c, start_temp=0.03, end_temp=1e-6, iterations=10000)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        
        # Final polish
        if time.time() < deadline - 5.0:
            c, r = _local_optimize(best_c, max_iter=3000, tol=1e-12)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 2.0 and best_c is not None:
        c, r = _local_optimize(best_c, max_iter=5000, tol=1e-14)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
