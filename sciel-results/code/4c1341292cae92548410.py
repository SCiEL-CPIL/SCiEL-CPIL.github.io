"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 100
TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _radius(x):
    """Compute the radius (score) for a given set of centers."""
    x = x.reshape(-1, 3)
    # Distance to walls
    wall_dist = 0.5 - np.max(np.abs(x), axis=1)
    min_wall = np.min(wall_dist)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_wall, min_pair / 2.0)


def _neg_radius(x):
    return -_radius(x)


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))


def _init_lattice_perturbed():
    """Start from a body-centered cubic-like arrangement."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # Simple cubic with jitter
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    pts += np.random.normal(0, 0.01, size=pts.shape)
    return pts


def _soft_repulsion(x, alpha=50.0):
    """Soft surrogate: sum of exp(-alpha * dist) for pairs + wall penalty."""
    x = x.reshape(-1, 3)
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    # Soft min via log-sum-exp approximation
    # Use sum of exp(-alpha*d) as a smooth surrogate for 1/min_dist
    pair_cost = np.sum(np.exp(-alpha * d))
    # Wall penalty
    wall_dist = 0.5 - np.abs(x)
    min_wall = np.min(wall_dist)
    wall_penalty = np.exp(-alpha * min_wall) * 10.0 if min_wall < 0.05 else 0.0
    return pair_cost + wall_penalty


def _optimize_from(start, time_budget):
    """Run L-BFGS-B from a given start point."""
    start = start.reshape(-1).copy()
    best_x = start.copy()
    best_r = _radius(start)
    
    deadline = time.time() + time_budget
    max_iter = 300
    
    try:
        res = minimize(
            _neg_radius,
            start,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            r = _radius(res.x)
            if r > best_r:
                best_x = res.x.copy()
                best_r = r
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_x, best_r
    
    # Try a second round with tighter tolerance if time allows
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                _neg_radius,
                best_x,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'maxls': 30, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                r2 = _radius(res2.x)
                if r2 > best_r:
                    best_x = res2.x.copy()
                    best_r = r2
        except Exception:
            pass
    
    return best_x, best_r


def _local_polish(x, time_budget):
    """High-precision polish: coordinate descent + L-BFGS with very tight tolerances."""
    x = x.reshape(-1, 3).copy()
    best_r = _radius(x)
    deadline = time.time() + time_budget
    
    # Coordinate descent: move each point in each direction to maximize min distance
    for iteration in range(5):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for dim in range(3):
                # Try small perturbations
                for delta in [0.001, 0.0005, 0.0001, -0.001, -0.0005, -0.0001]:
                    x_new = x.copy()
                    x_new[i, dim] += delta
                    # Check bounds
                    if np.any(x_new < -0.5) or np.any(x_new > 0.5):
                        continue
                    r_new = _radius(x_new)
                    if r_new > best_r + 1e-15:
                        x = x_new
                        best_r = r_new
                        improved = True
                        break
        if not improved:
            break
    
    # Final L-BFGS polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _neg_radius,
                x.reshape(-1),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                r = _radius(res.x)
                if r > best_r:
                    x = res.x.reshape(-1, 3)
                    best_r = r
        except Exception:
            pass
    
    return x, best_r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    np.random.seed(42)
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Simple cubic grid
    initializations.append(_init_grid())
    
    # 2. Random initializations
    for _ in range(4):
        initializations.append(_init_random())
    
    # 3. Perturbed lattice
    initializations.append(_init_lattice_perturbed())
    
    # 4. FCC-like structure perturbed
    m = 5
    fcc_pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                fcc_pts.append([x, y, z])
                # FCC extra points
                if i < m-1 and j < m-1:
                    fcc_pts.append([(i+1)/m - 0.5, (j+1)/m - 0.5, z])
                if i < m-1 and k < m-1:
                    fcc_pts.append([(i+1)/m - 0.5, y, (k+1)/m - 0.5])
                if j < m-1 and k < m-1:
                    fcc_pts.append([x, (j+1)/m - 0.5, (k+1)/m - 0.5])
    fcc_pts = np.array(fcc_pts[:N])
    fcc_pts += np.random.normal(0, 0.005, size=fcc_pts.shape)
    initializations.append(fcc_pts)
    
    # Optimize each initialization
    best_x = initializations[0].copy()
    best_r = _radius(initializations[0])
    
    for idx, init in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Allocate time per initialization
        time_per = min(remaining / max(1, len(initializations) - idx), 20.0)
        
        x_opt, r_opt = _optimize_from(init, time_per)
        if r_opt > best_r:
            best_x = x_opt.copy()
            best_r = r_opt
    
    # Polish the best result
    remaining = deadline - time.time()
    if remaining > 5:
        best_x, best_r = _local_polish(best_x, min(remaining - 2, 30.0))
    
    # Final validation and cleanup
    best_x = best_x.reshape(-1, 3)
    # Clamp to valid range
    best_x = np.clip(best_x, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_x
# EVOLVE-BLOCK-END
