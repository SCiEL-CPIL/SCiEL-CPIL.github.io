"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    # Pairwise distances using vectorized computation
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _force_directed_optimize(centres, time_limit=100.0):
    """
    Force-directed optimization: push apart closest pairs and pull toward center.
    Much faster than general-purpose optimizers for this problem.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    n = len(centres)
    c = centres.copy()
    
    # Adaptive step size
    step = 0.005
    min_step = 1e-7
    patience = 0
    max_patience = 500
    
    while time.time() - start_time < time_limit:
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum distance pair(s)
        min_dist = np.min(dist)
        
        # Compute forces: repulsive from close neighbors, attractive to center
        forces = np.zeros_like(c)
        
        # Repulsive forces: for each pair, push apart if too close
        # Use a soft repulsion: force ~ 1/d^2 for d < threshold
        threshold = min_dist * 2.0 + 0.01
        mask = (dist < threshold) & (dist > 1e-10)
        
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = dist[i, j]
                    # Force magnitude: stronger when closer
                    f_mag = 1.0 / (d * d + 1e-6)
                    direction = (c[i] - c[j]) / d
                    forces[i] += f_mag * direction
                    forces[j] -= f_mag * direction
        
        # Wall forces: push away from walls if too close
        for i in range(n):
            for k in range(3):
                if c[i, k] > 0.45:
                    forces[i, k] -= (c[i, k] - 0.45) * 10.0
                elif c[i, k] < -0.45:
                    forces[i, k] += (-0.45 - c[i, k]) * 10.0
        
        # Centering force: weak attraction to center to prevent drift
        forces -= 0.01 * c
        
        # Normalize forces to unit magnitude for stable stepping
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        # Apply forces with adaptive step
        c += step * forces
        
        # Clip to valid range
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check if improved
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
            patience = 0
            # Increase step slightly on improvement
            step = min(step * 1.05, 0.02)
        else:
            patience += 1
            # Decrease step when stuck
            step = max(step * 0.98, min_step)
        
        # If stuck for too long, add random perturbation (jiggle)
        if patience > max_patience:
            rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
            c += rng.normal(0, 0.01, size=c.shape)
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
            patience = 0
            step = 0.005
        
        # Periodic check
        if (time.time() - start_time) % 1.0 < 0.01:
            pass
    
    return best_centres, best_radius


def _make_fcc_packing():
    """
    Create an FCC-inspired packing for 50 spheres.
    FCC has 4 spheres per unit cell. We use a 3x3x4 grid of FCC cells,
    then select 50 spheres with the best spacing.
    """
    # FCC basis vectors
    a = 1.0  # lattice parameter
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a
    
    # Generate FCC lattice points in a 3x3x4 supercell
    nx, ny, nz = 3, 3, 4
    points = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in basis:
                    p = np.array([ix, iy, iz]) * a + b
                    # Scale to fit in [-0.5, 0.5]^3
                    # Total extent: (nx-1+1)*a = nx*a = 3a in x, etc.
                    # We want this to fit in 1.0, so scale = 1.0 / (nx * a)
                    p = p / (nx * a) - 0.5
                    points.append(p)
    
    points = np.array(points)
    
    # We have nx*ny*nz*4 = 3*3*4*4 = 144 points, need 50
    # Select 50 with maximum minimum distance using greedy algorithm
    selected = [0]  # Start with first point
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        # Find point in remaining that maximizes min distance to selected
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(points[selected] - points[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = points[selected]
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.005, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_random_packing(seed):
    """Create a random initial packing."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def _make_grid_packing():
    """Create a grid-based initial packing."""
    # 4x4x4 grid = 64 points, select 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 with best spacing
    selected = [0]
    remaining = list(range(1, len(grid)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(grid[selected] - grid[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = grid[selected]
    
    # Add small perturbation
    rng = np.random.RandomState(123)
    centres += rng.normal(0, 0.003, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    best_centres = None
    best_radius = 0.0
    
    # Strategy 1: FCC-inspired packing
    if time.time() < deadline - 30:
        try:
            init = _make_fcc_packing()
            centres, radius = _force_directed_optimize(init, time_limit=min(40, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 2: Grid-based packing
    if time.time() < deadline - 20 and best_centres is not None:
        try:
            init = _make_grid_packing()
            centres, radius = _force_directed_optimize(init, time_limit=min(25, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with short optimizations
    if time.time() < deadline - 10:
        for seed in [1, 7, 13, 42, 99, 123, 456]:
            if time.time() > deadline - 5:
                break
            try:
                init = _make_random_packing(seed)
                # Short optimization for random starts
                time_left = deadline - time.time() - 2
                if time_left < 5:
                    break
                centres, radius = _force_directed_optimize(init, time_limit=min(15, time_left))
                if radius > best_radius:
                    best_radius = radius
                    best_centres = centres
            except Exception:
                continue
    
    # If no valid result, fallback to simple grid
    if best_centres is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Final validation
    if not np.all(np.isfinite(best_centres)) or np.any(np.abs(best_centres) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
