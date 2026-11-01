import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _force_directed_step(centers, step_size, rng):
    """
    Force-directed repulsion step: push apart close pairs and away from walls.
    """
    c = centers.copy()
    
    # Compute pairwise distances
    dists = cdist(c, c, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    
    # Find closest pair
    min_dist = np.min(dists)
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    
    # Apply repulsive force to closest pair
    if min_dist < 0.3:  # Only if they're close enough to matter
        diff = c[i] - c[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            force = diff / norm * step_size * 0.5
            c[i] += force
            c[j] -= force
    
    # Push particles away from walls
    for k in range(3):
        # Left wall
        close_left = c[:, k] < -0.4
        if np.any(close_left):
            c[close_left, k] += step_size * 0.3
        # Right wall
        close_right = c[:, k] > 0.4
        if np.any(close_right):
            c[close_right, k] -= step_size * 0.3
    
    # Add small random perturbation
    c += rng.normal(0, step_size * 0.1, c.shape)
    
    # Clip to cube
    c = np.clip(c, -0.5, 0.5)
    
    return c

def _adaptive_local_search(centers, deadline, rng):
    """
    Adaptive local search with force-directed steps and annealing.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Parameters
    step_size = 0.02
    min_step = 0.001
    decay = 0.999
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Force-directed step
        c_new = _force_directed_step(c, step_size, rng)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r + 1e-12:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Decay step size
        step_size = max(min_step, step_size * decay)
        
        # If stuck, do a random kick to escape local optimum
        if no_improve_count > max_no_improve:
            # Kick a random subset of particles
            kick_indices = rng.choice(N, size=15, replace=False)
            c[kick_indices] += rng.normal(0, 0.08, (15, 3))
            c = np.clip(c, -0.5, 0.5)
            r_curr = _compute_radius(c)
            if r_curr > best_r:
                best_c = c.copy()
                best_r = r_curr
            no_improve_count = 0
            # Reset step size
            step_size = 0.02

    return best_c, best_r

def _init_fcc_like():
    """
    FCC-like initialization: face-centered cubic lattice distorted to fit 100 points.
    """
    # FCC lattice in unit cell: corners + face centers
    # Scale to fit in [-0.5, 0.5]^3
    points = []
    
    # Generate FCC points in a 4x4x4 supercell
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                points.append([i*0.25 - 0.45, j*0.25 - 0.45, k*0.25 - 0.45])
                # Face centers
                if i < 3:
                    points.append([i*0.25 + 0.125 - 0.45, j*0.25 - 0.45, k*0.25 - 0.45])
                if j < 3:
                    points.append([i*0.25 - 0.45, j*0.25 + 0.125 - 0.45, k*0.25 - 0.45])
                if k < 3:
                    points.append([i*0.25 - 0.45, j*0.25 - 0.45, k*0.25 + 0.125 - 0.45])
    
    points = np.array(points)
    
    # Keep only points inside [-0.45, 0.45]^3
    mask = np.all(np.abs(points) < 0.45, axis=1)
    points = points[mask]
    
    # If we have more than N, randomly select N
    if len(points) > N:
        idx = np.random.default_rng(42).choice(len(points), N, replace=False)
        points = points[idx]
    
    # If we have fewer, pad with random points
    if len(points) < N:
        rng = np.random.default_rng(42)
        pad = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, pad])
    
    return points

def _init_bcc_like():
    """
    BCC-like initialization: body-centered cubic lattice.
    """
    points = []
    
    # Generate BCC points in a 4x4x4 supercell
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                points.append([i*0.25 - 0.45, j*0.25 - 0.45, k*0.25 - 0.45])
                # Body center
                points.append([i*0.25 + 0.125 - 0.45, j*0.25 + 0.125 - 0.45, k*0.25 + 0.125 - 0.45])
    
    points = np.array(points)
    
    # Keep only points inside [-0.45, 0.45]^3
    mask = np.all(np.abs(points) < 0.45, axis=1)
    points = points[mask]
    
    # If we have more than N, randomly select N
    if len(points) > N:
        idx = np.random.default_rng(43).choice(len(points), N, replace=False)
        points = points[idx]
    
    # If we have fewer, pad with random points
    if len(points) < N:
        rng = np.random.default_rng(43)
        pad = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, pad])
    
    return points

def _init_grid():
    """
    Simple grid initialization.
    """
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    init = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    return init

def construct():
    start_time = time.time()
    # Safety margin: 110s is well within 120s.
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Multiple initializations
    inits = [
        _init_fcc_like(),
        _init_bcc_like(),
        _init_grid(),
    ]
    
    # Add perturbed versions
    for i in range(3):
        perturbed = inits[i] + rng.normal(0, 0.02, inits[i].shape)
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)
    
    # Sort by initial radius to prioritize good starts
    init_radii = [_compute_radius(i) for i in inits]
    sorted_idx = np.argsort(init_radii)[::-1]
    inits = [inits[i] for i in sorted_idx]
    
    # Time allocation: focus on top 3
    weights = [0.4, 0.3, 0.2, 0.05, 0.05]
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        slice_duration = (deadline - start_time) * weights[i]
        slice_deadline = time.time() + slice_duration
        slice_deadline = min(slice_deadline, deadline - 1.0)
        
        c_opt, r_opt = _adaptive_local_search(init, slice_deadline, rng)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = c_opt.copy()
            
    # Final polish on best found
    if best_centers is not None and time.time() < deadline - 2.0:
        c_final, r_final = _adaptive_local_search(best_centers, deadline - 0.5, rng)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c_final.copy()
            
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers

# EVOLVE-BLOCK-END
