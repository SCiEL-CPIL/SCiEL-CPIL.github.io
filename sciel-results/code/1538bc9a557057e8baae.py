"""Combined program: structured grid initializations + iterative local refinement."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _compute_radius(points):
    """Compute the maximum common radius for given centers."""
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(np.min(wall_dists, axis=1))
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _make_grid(nx, ny, nz):
    """Create a grid of points in [-0.5, 0.5]^3 with nx*ny*nz >= N."""
    if nx * ny * nz < N:
        return None
    x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in z for y in y for x in x])
    return grid[:N]

def _refine(points, time_limit, seed=42):
    """Iterative local refinement: push points apart and away from walls."""
    rng = np.random.default_rng(seed)
    current = points.copy()
    current_r = _compute_radius(current)
    
    # Precompute pairs for efficiency
    n = len(current)
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    step_size = 0.05
    patience = 0
    max_patience = 50
    
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Shuffle order for stochasticity
        order = rng.permutation(n)
        
        for i in order:
            if time.time() - start > time_limit:
                break
            
            # Compute forces on point i
            # From other points
            other_indices = np.concatenate([idx_i[idx_i != i], idx_j[idx_j != i]])
            other_indices = np.unique(other_indices)
            
            if len(other_indices) > 0:
                vecs = current[other_indices] - current[i]
                dists = np.linalg.norm(vecs, axis=1)
                # Avoid division by zero
                valid = dists > 1e-10
                if np.any(valid):
                    vecs = vecs[valid]
                    dists = dists[valid]
                    # Force away from neighbors
                    forces = -vecs / (dists[:, np.newaxis] * dists[:, np.newaxis])
                    force_sum = np.sum(forces, axis=0)
                else:
                    force_sum = np.zeros(3)
            else:
                force_sum = np.zeros(3)
            
            # From walls
            wall_force = np.zeros(3)
            for d in range(3):
                pos = current[i, d]
                if pos > 0:
                    wall_force[d] = -1.0  # Push left
                else:
                    wall_force[d] = 1.0   # Push right
            
            # Normalize force_sum
            norm_fs = np.linalg.norm(force_sum)
            if norm_fs > 0:
                force_sum = force_sum / norm_fs
            
            # Combine forces
            total_force = force_sum + 0.5 * wall_force
            norm_tf = np.linalg.norm(total_force)
            if norm_tf > 0:
                total_force = total_force / norm_tf
            
            # Try moving
            new_point = current[i].copy()
            new_point += step_size * total_force
            # Clamp to cube
            new_point = np.clip(new_point, -0.5, 0.5)
            
            # Check if improvement
            old_point = current[i].copy()
            current[i] = new_point
            new_r = _compute_radius(current)
            
            if new_r > current_r + 1e-12:
                current_r = new_r
                improved = True
                patience = 0
            else:
                current[i] = old_point
                patience += 1
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
        
        if patience > max_patience:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate initial grids
    grids = []
    # 5x5x4
    g1 = _make_grid(5, 5, 4)
    if g1 is not None:
        grids.append(g1)
    # 4x5x5
    g2 = _make_grid(4, 5, 5)
    if g2 is not None:
        grids.append(g2)
    # 5x4x5
    g3 = _make_grid(5, 4, 5)
    if g3 is not None:
        grids.append(g3)
    # 4x4x7 (112 points, take 100)
    g4 = _make_grid(4, 4, 7)
    if g4 is not None:
        grids.append(g4)
    # 7x4x4
    g5 = _make_grid(7, 4, 4)
    if g5 is not None:
        grids.append(g5)
    # 4x7x4
    g6 = _make_grid(4, 7, 4)
    if g6 is not None:
        grids.append(g6)
    
    # Also add a centered 5x5x4 with better spacing
    x5 = np.linspace(-0.4, 0.4, 5)
    y5 = np.linspace(-0.4, 0.4, 5)
    z4 = np.linspace(-0.3, 0.3, 4)
    grid_centered = np.array([(x, y, z) for z in z4 for y in y5 for x in x5])
    grids.append(grid_centered)
    
    best_result = None
    best_score = 0.0
    
    # First pass: evaluate all initial grids
    for grid in grids:
        if time.time() > deadline:
            break
        s = _compute_radius(grid)
        if s > best_score:
            best_score = s
            best_result = grid.copy()
    
    # Second pass: refine the best initial grid
    if best_result is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            refined = _refine(best_result, remaining_time - 5, seed=42)
            s_refined = _compute_radius(refined)
            if s_refined > best_score:
                best_score = s_refined
                best_result = refined
    
    # Try refining a few other good initializations if time permits
    if time.time() < deadline - 20:
        # Try the second best grid
        scored_grids = [( _compute_radius(g), g) for g in grids]
        scored_grids.sort(key=lambda x: x[0], reverse=True)
        for score_g, grid_g in scored_grids[1:3]:
            if time.time() > deadline - 10:
                break
            remaining = deadline - time.time() - 5
            if remaining < 5:
                break
            refined = _refine(grid_g, remaining, seed=123)
            s = _compute_radius(refined)
            if s > best_score:
                best_score = s
                best_result = refined
    
    if best_result is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_result
# EVOLVE-BLOCK-END
