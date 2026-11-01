"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140  # safety margin before 150s limit


def _min_dist_and_wall(c):
    """Compute the minimum distance between any pair of centres and to walls.
    Returns (min_pair_dist, min_wall_dist)."""
    n = len(c)
    # Pairwise distances
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances: for each coordinate, distance to nearest wall
    # Cube is [-0.5, 0.5]^3, so wall distance for coordinate k is 0.5 - |c_ik|
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min_pair, min_wall


def _radius(c):
    """Compute the common radius for a given set of centres."""
    min_pair, min_wall = _min_dist_and_wall(c)
    return min(min_pair / 2.0, min_wall)


def _generate_initial_grid():
    """Generate a 5x5x2 grid of 50 points inside the unit cube."""
    # 5 points in x, 5 in y, 2 in z
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.array([-0.25, 0.25])
    
    points = []
    for zi in z:
        for yi in y:
            for xi in x:
                points.append([xi, yi, zi])
    return np.array(points)


def _generate_fcc_like():
    """Generate an FCC-like arrangement for 50 spheres."""
    # FCC structure: use a 4x4x4 grid with FCC offsets
    # Actually, let's use a body-centered cubic or FCC unit cell approach
    # For simplicity, start with a slightly perturbed grid
    
    # Use a 5x5x2 grid but with slight offsets to break symmetry
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.array([-0.25, 0.25])
    
    points = []
    for zi in z:
        for yi in y:
            for xi in x:
                points.append([xi, yi, zi])
    return np.array(points)


def _random_initial():
    """Generate random initial positions inside the cube."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))


def construct():
    start_time = time.time()
    
    # Start with a good initial configuration
    best_c = _generate_initial_grid()
    best_r = _radius(best_c)
    
    # Try a few different initializations
    inits = [
        _generate_initial_grid(),
        _generate_fcc_like(),
    ]
    # Add a couple random starts
    for _ in range(3):
        inits.append(_random_initial())
    
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        r0 = _radius(c0)
        if r0 > best_r:
            best_c = c0.copy()
            best_r = r0
        
        # Try to optimize from this starting point
        try:
            # We want to maximize the minimum distance, which is equivalent to
            # maximizing t subject to all pairwise distances >= 2t and wall distances >= t
            # Use a penalty-based approach or sequential optimization
            
            # Approach: iteratively push apart the closest pairs
            # Simple local optimization: gradient-free Nelder-Mead on negative radius
            res = minimize(
                lambda x: -_radius(x.reshape(N, 3)),
                c0.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.success or res.fun < -best_r:
                c_opt = res.x.reshape(N, 3)
                r_opt = _radius(c_opt)
                if r_opt > best_r:
                    best_c = c_opt.copy()
                    best_r = r_opt
        except Exception:
            pass
    
    # Final polish: if we have time, do a more careful local optimization
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Use a different optimizer for polish
            res = minimize(
                lambda x: -_radius(x.reshape(N, 3)),
                best_c.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            c_opt = res.x.reshape(N, 3)
            r_opt = _radius(c_opt)
            if r_opt > best_r:
                best_c = c_opt.copy()
                best_r = r_opt
        except Exception:
            pass
    
    # Ensure the result is valid
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
