"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score(positions):
    """Compute the packing radius for given positions."""
    n = len(positions)
    # Distance to walls
    wall_dists = np.min(np.abs(positions), axis=1)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)


def _objective(pos_flat):
    """Negative of the packing radius (we minimize the negative)."""
    positions = pos_flat.reshape(N, 2)
    return -_score(positions)


def _grad(pos_flat):
    """Gradient of the objective using finite differences (approximate)."""
    # For speed, use a sparse finite difference approach
    positions = pos_flat.reshape(N, 2)
    n = len(positions)
    
    # Compute current score components
    wall_dists = np.min(np.abs(positions), axis=1)
    min_wall = np.min(wall_dists)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_half = np.min(dists) / 2.0
    
    r = min(min_wall, min_pair_half)
    
    grad = np.zeros_like(pos_flat)
    eps = 1e-7
    
    # Only compute gradient for the active constraints
    if min_wall <= min_pair_half:
        # Active constraint is wall
        for i in range(n):
            for k in range(2):
                if np.abs(positions[i, k]) == min_wall + 1e-10 or abs(abs(positions[i, k]) - min_wall) < 1e-8:
                    # Which side is it on?
                    if positions[i, k] > 0:
                        grad[i*2+k] = -1.0
                    else:
                        grad[i*2+k] = 1.0
    else:
        # Active constraint is pairwise
        # Find the pair(s) achieving min distance
        min_dist = np.min(dists)
        for i in range(n):
            for j in range(i+1, n):
                if abs(dists[i, j] - min_dist) < 1e-8:
                    d = positions[i] - positions[j]
                    dist = np.linalg.norm(d)
                    if dist > 1e-12:
                        grad[i*2] += d[0] / dist
                        grad[i*2+1] += d[1] / dist
                        grad[j*2] -= d[0] / dist
                        grad[j*2+1] -= d[1] / dist
    
    return grad


def _initial_guess_grid():
    """Create an initial grid-like guess."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _initial_guess_honeycomb():
    """Create a honeycomb-like initial guess for better packing."""
    positions = []
    # Use a hexagonal lattice pattern
    dx = 1.0 / 7
    dy = dx * np.sqrt(3) / 2
    x_start = -0.5 + dx/2
    y_start = -0.5 + dy/2
    
    for row in range(10):
        for col in range(10):
            x = x_start + col * dx + (0 if row % 2 == 0 else dx/2)
            y = y_start + row * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                positions.append([x, y])
    
    positions = np.array(positions[:N])
    return positions


def _run_optimization(x0, time_budget):
    """Run the optimization from initial guess x0 with a time budget."""
    start_time = time.time()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = None
    best_pos = x0.copy()
    best_score = _score(x0)
    
    try:
        # First, do a quick Nelder-Mead for local improvement
        res = minimize(
            _objective,
            x0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        elapsed = time.time() - start_time
        if elapsed < time_budget * 0.5:
            pos = res.x.reshape(N, 2)
            sc = _score(pos)
            if sc > best_score:
                best_score = sc
                best_pos = pos.copy()
        
        # Then refine with L-BFGS-B
        elapsed = time.time() - start_time
        if elapsed < time_budget * 0.7:
            res2 = minimize(
                _objective,
                best_pos.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            pos = res2.x.reshape(N, 2)
            sc = _score(pos)
            if sc > best_score:
                best_score = sc
                best_pos = pos.copy()
        
    except Exception:
        pass
    
    return best_pos, best_score


def _perturb(positions, scale=0.05):
    """Add random perturbation to positions."""
    perturbed = positions + np.random.randn(*positions.shape) * scale
    # Clip to valid range
    perturbed = np.clip(perturbed, -0.49, 0.49)
    return perturbed


def construct():
    """Main entry point. Returns array of shape (N, 2) with circle centers."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    # Generate multiple initial guesses
    initial_guesses = []
    
    # 1. Grid
    grid = _initial_guess_grid()
    initial_guesses.append(grid)
    
    # 2. Honeycomb
    honeycomb = _initial_guess_honeycomb()
    initial_guesses.append(honeycomb)
    
    # 3. Random perturbations of grid
    for i in range(3):
        perturbed = _perturb(grid, scale=0.03 + 0.02*i)
        initial_guesses.append(perturbed)
    
    # 4. Random perturbations of honeycomb
    for i in range(3):
        perturbed = _perturb(honeycomb, scale=0.03 + 0.02*i)
        initial_guesses.append(perturbed)
    
    # 5. Fully random
    for i in range(2):
        random_pos = np.random.uniform(-0.45, 0.45, (N, 2))
        initial_guesses.append(random_pos)
    
    best_pos = None
    best_score = -np.inf
    
    for i, x0 in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        # Allocate time for this optimization
        time_budget = min(15, deadline - elapsed - 5)
        if time_budget < 2:
            break
        
        pos, score = _run_optimization(x0, time_budget)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
    
    # Final check: ensure all positions are valid
    if best_pos is None:
        # Fallback to grid
        best_pos = _initial_guess_grid()
    
    # Clip to ensure inside square (with small margin for safety)
    best_pos = np.clip(best_pos, -0.4999, 0.4999)
    
    return best_pos

# EVOLVE-BLOCK-END
