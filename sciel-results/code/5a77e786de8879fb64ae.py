"""Hexagonal lattice packing with differential evolution for 50 circles."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85


def _fast_radius(centers):
    """Fast radius computation without validity checks."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _compute_radius(centers):
    """Compute the largest common radius with validity checks."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    if min_boundary <= 0:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _hex_lattice(params, row_counts):
    """Generate hexagonal lattice centers from parameters.
    
    params: [dx, dy, h_offs, row_shifts...]
    dx: horizontal spacing
    dy: vertical spacing
    h_offs: horizontal offset for odd rows
    row_shifts: per-row horizontal shift
    """
    dx = params[0]
    dy = params[1]
    h_offs = params[2]
    row_shifts = params[3:3 + len(row_counts)]
    
    centers = []
    n_rows = len(row_counts)
    
    for row_idx, count in enumerate(row_counts):
        y = -0.5 + dy / 2 + row_idx * dy
        # Center the row horizontally
        if count > 1:
            x_start = -0.5 + dx / 2 + row_shifts[row_idx]
            for col_idx in range(count):
                x = x_start + col_idx * dx
                centers.append([x, y])
        else:
            x = 0.0 + row_shifts[row_idx]
            centers.append([x, y])
    
    return np.array(centers[:N])


def _objective(params, row_counts):
    """Objective function for differential evolution (maximize radius)."""
    try:
        centers = _hex_lattice(params, row_counts)
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(np.abs(centers) > 0.5):
            return 0.0
        r = _fast_radius(centers)
        return -r  # Minimize negative radius
    except Exception:
        return 0.0


def _local_search(centers, n_iterations=5000, deadline=None):
    """Local search with adaptive step sizes."""
    best_centers = centers.copy()
    best_radius = _fast_radius(best_centers)
    rng = np.random.RandomState(42)
    
    current_step = 0.01
    no_improve_count = 0
    max_no_improve = 30
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.randint(N)
        orig = best_centers[idx].copy()
        
        improved = False
        for attempt in range(3):
            s = current_step * (0.5 ** attempt)
            perturbation = rng.normal(0, s, size=2)
            candidate = orig + perturbation
            np.clip(candidate, -0.5 + 1e-8, 0.5 - 1e-8, out=candidate)
            best_centers[idx] = candidate
            
            new_radius = _fast_radius(best_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                improved = True
                break
            else:
                best_centers[idx] = orig
        
        if improved:
            no_improve_count = 0
            current_step = min(0.03, current_step * 1.01)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                current_step = max(0.001, current_step * 0.85)
                no_improve_count = 0
    
    return best_centers, best_radius


def _try_row_config(row_counts, deadline):
    """Optimize a specific row configuration using differential evolution."""
    n_params = 3 + len(row_counts)
    
    # Bounds for parameters
    bounds = [(0.05, 0.15), (0.05, 0.15), (-0.05, 0.05)]
    for _ in row_counts:
        bounds.append((-0.05, 0.05))
    
    # Initial guess
    x0 = [0.1, 0.1, 0.0] + [0.0] * len(row_counts)
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            args=(row_counts,),
            x0=x0,
            seed=SEED,
            maxiter=200,
            tol=1e-8,
            popsize=10,
            polish=True,
            init='sobol',
            workers=1,
            callback=None
        )
        
        if result.success or result.fun < 0:
            centers = _hex_lattice(result.x, row_counts)
            r = _fast_radius(centers)
            if r > 0:
                return centers, r
    except Exception:
        pass
    
    return None, 0.0


def construct():
    """Main function that searches for the best packing configuration."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # Try various row configurations for 50 circles
    row_configs = [
        [7, 6, 7, 6, 7, 6, 7, 4],   # 50 circles
        [8, 7, 8, 7, 8, 7, 5],       # 50 circles
        [6, 7, 6, 7, 6, 7, 6, 5],    # 50 circles
        [5, 6, 5, 6, 5, 6, 5, 6, 6], # 50 circles
        [7, 7, 7, 7, 7, 7, 8],       # 50 circles
        [6, 6, 6, 6, 6, 6, 6, 6, 2], # 50 circles
    ]
    
    for row_counts in row_configs:
        if time.time() > deadline - 10:
            break
        
        centers, r = _try_row_config(row_counts, deadline)
        
        if centers is not None and r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Also try a simple grid as fallback
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_centers = np.array([(x, y) for y in ticks for x in ticks])[:N]
    grid_r = _fast_radius(grid_centers)
    if grid_r > best_radius:
        best_radius = grid_r
        best_centers = grid_centers.copy()
    
    # Local search refinement
    if best_centers is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        n_iters = max(500, int(min(10000, 10000 * remaining / 30.0)))
        best_centers, best_radius = _local_search(best_centers, n_iterations=n_iters, deadline=deadline)
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback to grid
        best_centers = grid_centers
        final_radius = _compute_radius(best_centers)
    
    return best_centers


# EVOLVE-BLOCK-END
