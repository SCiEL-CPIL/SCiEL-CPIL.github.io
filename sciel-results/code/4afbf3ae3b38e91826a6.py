"""Fifty equal spheres in a unit cube: local optimization from a good initial guess."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50
CUBE_HALF = 0.5
DEADLINE = 150.0  # seconds wall-clock


def _score(c):
    """Compute the packing radius for centres c (N,3)."""
    c = np.asarray(c, dtype=np.float64)
    # Wall distances
    wall = np.min(CUBE_HALF - np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return 0.5 * min(min_wall, min_pair)


def _gradient_ascent(c, lr, steps, deadline):
    """
    Simple gradient ascent on the radius.
    We maximize r = min(min_pair/2, min_wall).
    Use a subgradient method: for the binding constraint, push spheres apart or away from walls.
    """
    c = np.array(c, dtype=np.float64, copy=True)
    t0 = time.time()
    
    for step in range(steps):
        if time.time() - t0 > deadline:
            break
        
        # Compute current radius
        wall = CUBE_HALF - np.abs(c)
        min_wall_idx = np.argmin(wall.min(axis=1))
        min_wall_val = wall[min_wall_idx].min()
        
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_pair_idx = np.argmin(d)
        i, j = divmod(min_pair_idx, N)
        min_pair_val = d[i, j]
        
        r = 0.5 * min(min_wall_val, min_pair_val)
        
        # Determine which constraint is binding
        if min_pair_val < min_wall_val:
            # Binding: pair (i, j)
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = np.random.randn(3)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Move i and j apart
            c[i] += lr * direction
            c[j] -= lr * direction
        else:
            # Binding: wall constraint for sphere min_wall_idx
            # Push away from the closest wall
            sphere = min_wall_idx
            # Find which wall is closest
            closest_axis = np.argmin(wall[sphere])
            sign = np.sign(c[sphere, closest_axis])
            if sign == 0:
                sign = 1.0
            
            move = np.zeros(3)
            move[closest_axis] = sign
            
            c[sphere] += lr * move
        
        # Clamp to cube
        c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return c


def _local_search(c, deadline):
    """
    Iterative local search with decreasing learning rate.
    """
    t0 = time.time()
    c = np.array(c, dtype=np.float64, copy=True)
    best_r = _score(c)
    best_c = c.copy()
    
    lr = 0.01
    patience = 0
    max_patience = 500
    
    while time.time() - t0 < deadline and patience < max_patience:
        # Try a step
        c_new = _gradient_ascent(c, lr, steps=10, deadline=deadline)
        r_new = _score(c_new)
        
        if r_new > best_r + 1e-10:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
            patience = 0
        else:
            patience += 1
            lr *= 0.95
            # Small random perturbation to escape local minima
            c = c + 0.001 * np.random.randn(*c.shape)
            c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_c, best_r


def _initial_guess():
    """
    Create an initial configuration: 50 points in a body-centered cubic-like arrangement
    or a jittered grid.
    """
    # Use a 4x4x4 grid = 64 positions, pick 50, then jitter
    # Actually, let's use a more structured approach.
    # Best known packing has r ≈ 0.13595, so diameter ≈ 0.2719
    # In a unit cube, if we had a 5x5x5 grid, spacing = 1/4 = 0.25, radius = 0.125
    # With 4x4x4, spacing = 1/3 ≈ 0.333, radius ≈ 0.1667 but we only use 50 of 64.
    
    # Let's start with a 4x4x4 grid and remove 14 points, then optimize.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Take first 50
    c = grid[:50].copy()
    
    # Add small jitter to break symmetry
    c += 0.005 * np.random.randn(*c.shape)
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return c


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - 5.0  # 5s safety margin
    
    np.random.seed(42)
    
    best_c = None
    best_r = -1.0
    
    # Multiple restarts with different initial configurations
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initial guesses
        if restart == 0:
            c = _initial_guess()
        elif restart == 1:
            # Body-centered cubic inspired
            c = _bcc_initial()
        elif restart == 2:
            # Random
            c = np.random.uniform(-CUBE_HALF + 0.1, CUBE_HALF - 0.1, size=(N, 3))
        elif restart == 3:
            # Grid with different selection
            c = _grid_initial_alt()
        else:
            # Jittered 5x5x4 grid
            c = _grid_initial_554()
        
        # Local search
        c_opt, r = _local_search(c, deadline)
        
        if r > best_r:
            best_r = r
            best_c = c_opt
    
    # Final polish with very small steps
    if best_c is not None and time.time() < deadline:
        best_c = _gradient_ascent(best_c, lr=0.001, steps=100, deadline=deadline)
        best_c = np.clip(best_c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    if best_c is None:
        # Fallback
        best_c = _initial_guess()
    
    return best_c


def _bcc_initial():
    """Body-centered cubic initial guess."""
    # FCC/BCC inspired
    # Simple: two interpenetrating grids
    m = 3
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid1 = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add offset points
    offset = 0.5 / m
    grid2 = grid1 + offset
    
    # Combine and select
    all_pts = np.vstack([grid1, grid2])
    # Shuffle and take N
    idx = np.random.permutation(len(all_pts))
    c = all_pts[idx[:N]]
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    return c


def _grid_initial_alt():
    """Alternative grid initial guess."""
    # 5x5x2 grid = 50 points exactly
    nx, ny, nz = 5, 5, 2
    tx = (np.arange(nx) + 0.5) / nx - 0.5
    ty = (np.arange(ny) + 0.5) / ny - 0.5
    tz = (np.arange(nz) + 0.5) / nz - 0.5
    c = np.array([(x, y, z) for z in tz for y in ty for x in tx])
    c += 0.003 * np.random.randn(*c.shape)
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    return c


def _grid_initial_554():
    """5x5x4 grid, select 50."""
    nx, ny, nz = 5, 5, 4
    tx = (np.arange(nx) + 0.5) / nx - 0.5
    ty = (np.arange(ny) + 0.5) / ny - 0.5
    tz = (np.arange(nz) + 0.5) / nz - 0.5
    c = np.array([(x, y, z) for z in tz for y in ty for x in tx])
    # Take first 50
    c = c[:50].copy()
    c += 0.003 * np.random.randn(*c.shape)
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    return c
