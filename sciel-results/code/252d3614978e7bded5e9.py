"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
CUBE_HALF = 0.5


def _pairwise_distances(coords):
    """Return (n, n) matrix of Euclidean distances."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _compute_radius(coords):
    """Compute the valid packing radius for given centers."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n != N:
        return 0.0
    if not np.all(np.isfinite(coords)):
        return 0.0
    # Check inside cube
    if np.any(coords < -CUBE_HALF - 1e-12) or np.any(coords > CUBE_HALF + 1e-12):
        # Clamp for radius computation but mark as invalid if significantly outside
        pass
    # Wall distances
    wall_dists = np.min(np.stack([
        coords[:, 0] + CUBE_HALF,
        CUBE_HALF - coords[:, 0],
        coords[:, 1] + CUBE_HALF,
        CUBE_HALF - coords[:, 1],
        coords[:, 2] + CUBE_HALF,
        CUBE_HALF - coords[:, 2]
    ]), axis=0)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pairwise
    if n > 1:
        diff = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
    else:
        min_pair = np.inf
    r = 0.5 * min(min_wall, min_pair)
    return float(r)


def _initial_grid(n):
    """Create a simple cubic grid of n points in the unit cube centered at origin."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n].copy()


def _random_initial(n, seed):
    """Generate random initial positions inside the cube."""
    rng = np.random.RandomState(seed)
    coords = rng.uniform(-CUBE_HALF, CUBE_HALF, size=(n, 3))
    return coords


def _repulsion_step(coords, alpha, n_iter=50, temperature=0.01):
    """
    Perform gradient ascent on the minimum distance using repulsion forces.
    This is a simplified version of the "soft" repulsion approach.
    """
    n = coords.shape[0]
    best_coords = coords.copy()
    best_radius = _compute_radius(coords)
    
    for it in range(n_iter):
        # Compute all pairwise distances
        diff = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(d, np.inf)
        
        # Compute forces: each pair repels with strength proportional to 1/d^2
        # But we want to maximize the minimum distance, so we focus on the closest pairs
        # Use a soft version: force = diff / d^3 (normalized)
        force = np.zeros_like(coords)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dist = d[i, j]
                if dist < 1e-6:
                    # Random nudge for coincident points
                    force[i] += np.random.randn(3) * 0.01
                    force[j] -= np.random.randn(3) * 0.01
                    continue
                # Repulsion strength
                strength = 1.0 / (dist ** 2)
                dir_ij = (coords[j] - coords[i]) / dist
                force[i] -= dir_ij * strength
                force[j] += dir_ij * strength
        
        # Wall repulsion: push away from walls
        for i in range(n):
            for k in range(3):
                # Distance to negative wall
                dist_neg = coords[i, k] + CUBE_HALF
                if dist_neg < 0.1:
                    force[i, k] += 5.0 / max(dist_neg, 0.01)
                # Distance to positive wall
                dist_pos = CUBE_HALF - coords[i, k]
                if dist_pos < 0.1:
                    force[i, k] -= 5.0 / max(dist_pos, 0.01)
        
        # Update positions
        coords = coords + alpha * force
        
        # Keep inside cube
        coords = np.clip(coords, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        # Evaluate
        current_radius = _compute_radius(coords)
        if current_radius > best_radius:
            best_radius = current_radius
            best_coords = coords.copy()
        
        # Decay alpha
        alpha *= 0.98
    
    return best_coords, best_radius


def _local_optimization(coords, max_iters=1000, tol=1e-8):
    """
    Local optimization using a simple gradient-based method.
    """
    n = coords.shape[0]
    best_coords = coords.copy()
    best_radius = _compute_radius(coords)
    
    step_size = 0.001
    
    for it in range(max_iters):
        # Find the closest pair
        diff = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(d, np.inf)
        
        # Find closest pair
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx[0], idx[1]
        min_dist = d[i, j]
        
        # Also check wall distances
        wall_dists = []
        for k in range(3):
            wall_dists.append(coords[:, k] + CUBE_HALF)
            wall_dists.append(CUBE_HALF - coords[:, k])
        wall_dists = np.array(wall_dists)
        min_wall = np.min(wall_dists)
        
        # Determine which constraint is active
        if min_dist < min_wall:
            # Move i and j apart
            dir_ij = (coords[j] - coords[i]) / max(min_dist, 1e-9)
            coords[i] -= step_size * dir_ij
            coords[j] += step_size * dir_ij
        else:
            # Find the wall that's closest
            min_wall_idx = np.argmin(wall_dists)
            # Determine which point and which wall
            # wall_dists layout: [x_neg, x_pos, y_neg, y_pos, z_neg, z_pos] for each point
            point_idx = min_wall_idx // 6
            axis = (min_wall_idx % 6) // 2
            is_neg = (min_wall_idx % 2) == 0
            
            if is_neg:
                coords[point_idx, axis] += step_size
            else:
                coords[point_idx, axis] -= step_size
        
        # Keep inside cube
        coords = np.clip(coords, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        # Evaluate
        current_radius = _compute_radius(coords)
        if current_radius > best_radius + tol:
            best_radius = current_radius
            best_coords = coords.copy()
        elif current_radius < best_radius - 1e-6:
            # Revert if we made it worse
            coords = best_coords.copy()
        
        # Decay step size
        step_size *= 0.995
        
        if step_size < 1e-10:
            break
    
    return best_coords, best_radius


def _simulated_annealing(coords, max_iters=5000, T0=0.01, T1=1e-5, cooling=0.999):
    """
    Simulated annealing to find a better packing.
    """
    n = coords.shape[0]
    best_coords = coords.copy()
    best_radius = _compute_radius(coords)
    current_coords = coords.copy()
    current_radius = best_radius
    T = T0
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        # Propose a move: pick a random sphere and move it slightly
        i = rng.randint(n)
        delta = rng.normal(0, 0.01, size=3)
        new_coords = current_coords.copy()
        new_coords[i] = new_coords[i] + delta
        new_coords = np.clip(new_coords, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        new_radius = _compute_radius(new_coords)
        
        # Accept or reject
        if new_radius >= current_radius:
            accept = True
        else:
            delta_e = current_radius - new_radius
            if delta_e < 1e-12:
                accept = True
            else:
                accept_prob = np.exp(-delta_e / max(T, 1e-10))
                accept = rng.random() < accept_prob
        
        if accept:
            current_coords = new_coords
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_coords = current_coords.copy()
        
        T *= cooling
    
    return best_coords, best_radius


def construct():
    """
    Construct the centers of 100 equal spheres in the unit cube.
    Uses a combination of initial grid, repulsion-based optimization,
    local optimization, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with a grid
    coords = _initial_grid(N)
    best_coords = coords.copy()
    best_radius = _compute_radius(coords)
    
    # Step 1: Repulsion-based optimization
    coords, radius = _repulsion_step(coords, alpha=0.01, n_iter=200)
    if radius > best_radius:
        best_radius = radius
        best_coords = coords.copy()
    
    if time.time() > deadline:
        return best_coords
    
    # Step 2: Local optimization
    coords, radius = _local_optimization(coords, max_iters=2000)
    if radius > best_radius:
        best_radius = radius
        best_coords = coords.copy()
    
    if time.time() > deadline:
        return best_coords
    
    # Step 3: Simulated annealing with multiple restarts
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        
        # Start from current best or random
        if restart == 0:
            sa_coords = best_coords.copy()
        else:
            sa_coords = _random_initial(N, seed=restart + 100)
        
        sa_coords, sa_radius = _simulated_annealing(
            sa_coords, 
            max_iters=3000, 
            T0=0.005, 
            T1=1e-6, 
            cooling=0.9995
        )
        
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_coords = sa_coords.copy()
        
        # Follow up with local optimization
        if time.time() > deadline - 10.0:
            break
        
        lo_coords, lo_radius = _local_optimization(sa_coords, max_iters=1000)
        if lo_radius > best_radius:
            best_radius = lo_radius
            best_coords = lo_coords.copy()
    
    return best_coords
