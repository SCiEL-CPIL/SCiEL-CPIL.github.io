"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint


def _compute_radius(cents):
    """Compute the maximum common radius for given centers."""
    # Distance between all pairs
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Centers must be in [-0.5, 0.5]^2
    wall_dists = np.minimum(
        np.minimum(0.5 - cents[:, 0], cents[:, 0] + 0.5),
        np.minimum(0.5 - cents[:, 1], cents[:, 1] + 0.5)
    )
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _negative_radius(cents):
    """Objective function to minimize (negative of radius)."""
    return -_compute_radius(cents)


def _constraints(cents):
    """Constraints: centers must be inside the square."""
    # For each center, x and y must be in [-0.5, 0.5]
    # We use inequality constraints: g(x) >= 0
    # 0.5 - x_i >= 0  =>  x_i <= 0.5
    # x_i + 0.5 >= 0  =>  x_i >= -0.5
    
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 0.5 - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] + 0.5})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 0.5 - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] + 0.5})
    return cons


def _grid_init(m=None):
    """Generate initial grid configuration."""
    if m is None:
        m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].flatten()


def _random_init(seed=None):
    """Generate random initial configuration."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-0.4, 0.4, size=(N, 2)).flatten()


def _local_optimize(x0, method='SLSQP', maxiter=1000, tol=1e-8):
    """Run local optimization from initial point x0."""
    cons = _constraints(x0)
    
    result = minimize(
        _negative_radius,
        x0,
        method=method,
        constraints=cons,
        options={'maxiter': maxiter, 'ftol': tol, 'disp': False}
    )
    
    if result.success or result.fun < 0:
        return result.x
    return x0


def construct():
    """
    Construct 50 circle centers in the unit square to maximize minimum radius.
    Uses multiple random restarts with SLSQP local optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    best_cents = None
    best_radius = 0.0
    
    # Try grid initialization first
    x_grid = _grid_init()
    r_grid = _compute_radius(x_grid.reshape(N, 2))
    best_cents = x_grid.copy()
    best_radius = r_grid
    
    # Try SLSQP from grid
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        x_opt = _local_optimize(x_grid, method='SLSQP', maxiter=500, tol=1e-6)
        r_opt = _compute_radius(x_opt.reshape(N, 2))
        if r_opt > best_radius:
            best_cents = x_opt.copy()
            best_radius = r_opt
    
    # Multiple random restarts
    np.random.seed(42)
    n_restarts = 20
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Generate random initial configuration
        x0 = _random_init(seed=42 + i)
        
        # First do a quick optimization
        x_opt = _local_optimize(x0, method='SLSQP', maxiter=300, tol=1e-5)
        r_opt = _compute_radius(x_opt.reshape(N, 2))
        
        if r_opt > best_radius:
            best_cents = x_opt.copy()
            best_radius = r_opt
            
            # If we found a good solution, do a more precise optimization
            elapsed = time.time() - start_time
            if elapsed < deadline - 15 and r_opt > 0.065:
                x_fine = _local_optimize(x_opt, method='SLSQP', maxiter=1000, tol=1e-8)
                r_fine = _compute_radius(x_fine.reshape(N, 2))
                if r_fine > best_radius:
                    best_cents = x_fine.copy()
                    best_radius = r_fine
    
    # Final polish if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5 and best_cents is not None:
        x_final = _local_optimize(best_cents, method='SLSQP', maxiter=500, tol=1e-9)
        r_final = _compute_radius(x_final.reshape(N, 2))
        if r_final > best_radius:
            best_cents = x_final.copy()
            best_radius = r_final
    
    return best_cents.reshape(N, 2)
# EVOLVE-BLOCK-END
