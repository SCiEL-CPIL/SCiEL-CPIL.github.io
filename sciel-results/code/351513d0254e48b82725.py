"""Search-based approach for packing 50 equal circles in a unit square."""
import time
import numpy as np
import random

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85  # Use 85% of the 150s budget as safety margin

def _compute_radius(centers):
    """Compute the largest common radius for given centers.
    
    Returns min( min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|) )
    """
    if centers.shape != (N, 2):
        return 0.0
    # Check finiteness
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    # Distance to boundaries: 0.5 - |c_ik|
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances / 2
    # Use broadcasting for speed
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _fast_radius(centers):
    """Fast radius computation without validity checks (for internal use)."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)

def _initial_grid():
    """Generate initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)

def _hexagonal_initializer():
    """Generate hexagonal lattice initial configuration."""
    # For 50 circles, use a hexagonal packing pattern
    # Rows of circles in a hexagonal arrangement
    rows = []
    # Try different row counts
    best_config = None
    best_r = 0
    
    for n_rows in range(5, 12):
        # Distribute 50 circles among rows
        base_per_row = N // n_rows
        extra = N % n_rows
        row_counts = [base_per_row + (1 if i < extra else 0) for i in range(n_rows)]
        
        centers = []
        for row_idx, count in enumerate(row_counts):
            # Hexagonal offset
            offset = 0.0 if row_idx % 2 == 0 else 0.5 / max(count - 1, 1)
            for col_idx in range(count):
                x = -0.45 + 0.9 * col_idx / max(count - 1, 1) + offset * 0.1
                y = -0.45 + 0.9 * row_idx / max(n_rows - 1, 1)
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        r = _fast_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers
    
    return best_config if best_config is not None else _initial_grid()

def _random_initializer(n_points=N, seed=None):
    """Generate random initial positions within the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.5 + 0.05, 0.5 - 0.05, size=(n_points, 2))

def _force_relaxation(centers, max_iters=500, deadline=None):
    """Repulsive force relaxation to spread circles apart.
    
    Each circle repels others and is attracted to center of available space.
    This is much faster than Nelder-Mead for high-dimensional problems.
    """
    c = centers.copy().astype(float)
    step_size = 0.01
    min_step = 0.0001
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces on each circle
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random push if coincident
                    diff = np.random.randn(2) * 0.001
                    dist = np.linalg.norm(diff)
                
                # Strong repulsion when close
                if dist < 0.15:
                    force_mag = 0.5 / (dist + 0.01)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary attraction (push circles away from walls)
        for i in range(N):
            for axis in range(2):
                if c[i, axis] > 0:
                    dist_to_wall = 0.5 - c[i, axis]
                    if dist_to_wall < 0.1:
                        forces[i, axis] -= 0.3 / (dist_to_wall + 0.01)
                else:
                    dist_to_wall = 0.5 + c[i, axis]
                    if dist_to_wall < 0.1:
                        forces[i, axis] += 0.3 / (dist_to_wall + 0.01)
        
        # Apply forces with adaptive step size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            actual_step = min(step_size, 0.5 / max_force)
            c += actual_step * forces
        
        # Clip to bounds
        np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6, out=c)
        
        # Decrease step size
        step_size = max(min_step, step_size * 0.995)
    
    return c

def _evaluate_and_optimize(centers, max_iter=500, deadline=None):
    """Optimize positions using force relaxation (much faster than Nelder-Mead)."""
    try:
        optimized = _force_relaxation(centers, max_iters=max_iter, deadline=deadline)
        r = _fast_radius(optimized)
        return optimized, r
    except Exception:
        return centers, _compute_radius(centers)

def _local_search(centers, n_iterations=2000, step_size=0.02, deadline=None):
    """Improved local search with adaptive step sizes and multiple attempts per point."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    rng = np.random.RandomState(42)
    
    # Adaptive step size
    current_step = step_size
    no_improve_count = 0
    max_no_improve = 50
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to perturb
        idx = rng.randint(N)
        orig = best_centers[idx].copy()
        
        # Try multiple perturbations with decreasing step
        improved = False
        for attempt in range(3):
            s = current_step * (0.5 ** attempt)
            perturbation = rng.normal(0, s, size=2)
            candidate = orig + perturbation
            
            # Keep within bounds
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
            # Slightly increase step if improving
            current_step = min(0.05, current_step * 1.02)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Decrease step size
                current_step = max(0.001, current_step * 0.8)
                no_improve_count = 0
    
    return best_centers, best_radius

def construct():
    """Main function that searches for the best packing configuration."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # Start with multiple initialization strategies
    initializers = [
        _initial_grid(),
        _hexagonal_initializer(),
        _random_initializer(seed=SEED),
        _random_initializer(seed=SEED + 1),
        _random_initializer(seed=SEED + 2),
    ]
    
    for init_centers in initializers:
        r = _compute_radius(init_centers)
        if r > best_radius:
            best_radius = r
            best_centers = init_centers.copy()
    
    # Run force relaxation on each initial configuration
    for init_centers in initializers:
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 2.0:
            break
        
        # Scale iterations with remaining time
        max_iter = max(50, int(min(500, 500 * remaining_time / 30.0)))
        
        optimized, _ = _evaluate_and_optimize(init_centers, max_iter=max_iter, deadline=deadline)
        
        r = _compute_radius(optimized)
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()
    
    # Run local search on the best solution
    if time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 3.0
        n_iters = max(200, int(min(5000, 5000 * remaining / 20.0)))
        best_centers, best_radius = _local_search(best_centers, n_iterations=n_iters, deadline=deadline)
    
    # Second round of local search with different seed if time permits
    if time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        n_iters = max(100, int(min(3000, 3000 * remaining / 15.0)))
        # Slightly perturb best solution for diversity
        rng = np.random.RandomState(123)
        perturbed = best_centers + rng.normal(0, 0.005, size=(N, 2))
        np.clip(perturbed, -0.5 + 1e-8, 0.5 - 1e-8, out=perturbed)
        best_centers2, best_radius2 = _local_search(perturbed, n_iterations=n_iters, deadline=deadline)
        if best_radius2 > best_radius:
            best_radius = best_radius2
            best_centers = best_centers2.copy()
    
    # Final check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback to grid
        best_centers = _initial_grid()
    
    return best_centers

# EVOLVE-BLOCK-END
