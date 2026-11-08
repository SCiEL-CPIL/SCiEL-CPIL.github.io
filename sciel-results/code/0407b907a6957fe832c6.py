import numpy as np
from scipy.optimize import minimize
import time
import math

N = 21

def _voronoi_inradii(centers, W, H):
    """Compute the inradius of each circle's Voronoi cell.
    
    For each circle i, the inradius is the minimum of:
    - distance to each wall (x, W-x, y, H-y)
    - half the distance to each other circle j
    
    This is equivalent to the radius of the largest circle centered at
    centers[i] that fits within the Voronoi cell of i (clipped to the rectangle).
    """
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        # Distance to walls
        d_walls = np.array([centers[i, 0], W - centers[i, 0],
                           centers[i, 1], H - centers[i, 1]])
        # Half-distance to other circles
        dists = np.linalg.norm(centers[i] - centers, axis=1) / 2.0
        dists[i] = np.inf
        radii[i] = np.min(np.concatenate([d_walls, dists]))
    return radii

def _soft_min(arr, beta):
    """Numerically stable soft-min approximation."""
    m = np.min(arr)
    exp_terms = np.exp(-beta * (arr - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _soft_min_sum(centers, W, H, beta):
    """Compute sum of soft-min approximations of radii for all circles."""
    n = len(centers)
    total = 0.0
    for i in range(n):
        d_walls = np.array([centers[i, 0], W - centers[i, 0],
                           centers[i, 1], H - centers[i, 1]])
        dists = np.linalg.norm(centers[i] - centers, axis=1) / 2.0
        dists[i] = np.inf
        all_dists = np.concatenate([d_walls, dists])
        total += _soft_min(all_dists, beta)
    return total

def _objective(centers_flat, W, H, beta):
    """Negative sum of soft-min radii (to minimize)."""
    centers = centers_flat.reshape(N, 2)
    return -_soft_min_sum(centers, W, H, beta)

def _objective_grad(centers_flat, W, H, beta):
    """Numerical gradient of the objective."""
    eps = 1e-7
    grad = np.zeros_like(centers_flat)
    base = _objective(centers_flat, W, H, beta)
    for i in range(len(centers_flat)):
        orig = centers_flat[i]
        centers_flat[i] = orig + eps
        f_plus = _objective(centers_flat, W, H, beta)
        centers_flat[i] = orig
        grad[i] = (f_plus - base) / eps
    return grad

def _seed_grid(W, H, seed):
    """Generate initial center positions using a grid with perturbation."""
    np.random.seed(seed)
    n_x = int(np.sqrt(N * W / H))
    n_y = int(np.ceil(N / n_x))
    
    x_coords = np.linspace(0.1, W - 0.1, n_x + 2)[1:-1] if n_x > 1 else np.array([W/2])
    y_coords = np.linspace(0.1, H - 0.1, n_y + 2)[1:-1] if n_y > 1 else np.array([H/2])
    
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    centers = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    
    if len(centers) < N:
        # Add more points if grid is too small
        extra_x = np.linspace(0.1, W - 0.1, N - len(centers) + 1)
        extra_y = np.linspace(0.1, H - 0.1, N - len(centers) + 1)
        extra = np.column_stack((extra_x, extra_y))
        centers = np.vstack([centers, extra])
    
    centers = centers[:N]
    centers += np.random.normal(0, 0.03, centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
    centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
    return centers

def _optimize(W, H, initial_centers, deadline):
    """Two-stage optimization: coarse with soft-min, then fine polish."""
    bounds = [(0, W)] * N + [(0, H)] * N
    
    # Stage 1: Coarse optimization with soft-min
    beta_coarse = 15.0
    res1 = minimize(
        _objective, initial_centers.flatten(),
        args=(W, H, beta_coarse),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() > deadline:
        return initial_centers, -np.inf
    
    centers1 = res1.x.reshape(N, 2)
    
    # Stage 2: Fine polish with higher beta (closer to exact min)
    beta_fine = 50.0
    res2 = minimize(
        _objective, centers1.flatten(),
        args=(W, H, beta_fine),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if time.time() > deadline:
        return centers1, -np.inf
    
    centers2 = res2.x.reshape(N, 2)
    
    # Stage 3: Even finer polish
    beta_vfine = 100.0
    res3 = minimize(
        _objective, centers2.flatten(),
        args=(W, H, beta_vfine),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    if time.time() > deadline:
        return centers2, -np.inf
    
    centers3 = res3.x.reshape(N, 2)
    
    # Compute exact radii for each stage and pick best
    best_centers = centers1
    best_score = -np.inf
    
    for c in [centers1, centers2, centers3]:
        radii = _voronoi_inradii(c, W, H)
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_centers = c
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    best_score = -np.inf
    best_config = None

    # Search over rectangle widths (perimeter = 4 => H = 2 - W)
    widths = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
    
    n_restarts = 8

    for W in widths:
        if time.time() > deadline:
            break
        H = 2.0 - W

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            centers = _seed_grid(W, H, seed=42 + restart)
            
            try:
                final_centers, score = _optimize(W, H, centers, deadline)
                
                if score > best_score:
                    best_score = score
                    radii = _voronoi_inradii(final_centers, W, H)
                    best_config = np.column_stack((final_centers, radii))
                    
            except Exception:
                pass

    # Fallback if no valid result
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
