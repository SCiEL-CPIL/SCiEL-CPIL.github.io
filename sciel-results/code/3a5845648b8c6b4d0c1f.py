"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import random

def _radius(pos):
    """Compute the maximum common radius for a given configuration."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    if pos.shape[0] != N:
        return 0.0
    # Check all points are finite
    if not np.all(np.isfinite(pos)):
        return 0.0
    # Check all points are inside the square [-0.5, 0.5]^2
    if np.any(np.abs(pos) > 0.5 + 1e-12):
        return 0.0
    # Compute minimum distance between pairs
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to boundary
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair_dist / 2.0, min_boundary)

def _objective(pos, r_target):
    """Objective function: maximize minimum separation and boundary distance.
    We minimize the negative of the radius, but use a differentiable soft version."""
    pos = pos.reshape(-1, 2)
    # Soft minimum of pair distances
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, 1e10)
    
    # Soft min using log-sum-exp trick (more stable)
    # For pairs
    min_pair = np.min(dists)
    # For boundary
    boundary_dists = 0.5 - np.abs(pos)
    min_boundary = np.min(boundary_dists)
    
    # The actual radius is min(min_pair/2, min_boundary)
    # We want to maximize this, so minimize -radius
    # But for smoothness, use a soft version
    r = min(min_pair / 2.0, min_boundary)
    
    # Penalty for being outside square
    out_penalty = np.sum(np.maximum(0, np.abs(pos) - 0.5 + 1e-6))
    
    return -r + out_penalty * 100.0

def _gradient_descent(pos0, r_target, max_iter=5000, lr=0.001, deadline=None):
    """Custom gradient descent with momentum for the packing problem."""
    pos = pos0.copy()
    best_pos = pos.copy()
    best_r = _radius(pos)
    momentum = np.zeros_like(pos)
    lr0 = lr
    decay = 0.999
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        pos_flat = pos.reshape(-1, 2)
        dists = cdist(pos_flat, pos_flat)
        np.fill_diagonal(dists, 1e10)
        
        # Pairwise repulsion: push apart if too close
        force = np.zeros_like(pos_flat)
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                d = dists[i_idx, j_idx]
                if d < 2 * r_target:
                    # Repulsive force proportional to overlap
                    direction = pos_flat[i_idx] - pos_flat[j_idx]
                    if d > 1e-12:
                        direction = direction / d
                    else:
                        direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                        direction = direction / (np.linalg.norm(direction) + 1e-12)
                    overlap = 2 * r_target - d
                    force[i_idx] += direction * overlap
                    force[j_idx] -= direction * overlap
        
        # Boundary constraint: push inward if too close to boundary
        for i_idx in range(N):
            for axis in range(2):
                coord = pos_flat[i_idx, axis]
                if coord > 0.5 - r_target:
                    force[i_idx, axis] -= (coord - (0.5 - r_target))
                elif coord < -0.5 + r_target:
                    force[i_idx, axis] += ((-0.5 + r_target) - coord)
        
        # Update with momentum
        momentum = 0.9 * momentum + 0.1 * force
        pos = pos + lr0 * momentum
        lr0 *= decay
        
        # Clamp to square
        pos = np.clip(pos, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check improvement
        r = _radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
    
    return best_pos, best_r

def _langevin_dynamics(pos0, r_target, max_iter=20000, dt=0.0001, temp0=0.01, temp_end=0.001, deadline=None):
    """Langevin dynamics for packing optimization."""
    pos = pos0.copy()
    best_pos = pos.copy()
    best_r = _radius(pos)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        progress = i / max_iter
        temp = temp0 * (temp_end / temp0) ** progress
        
        # Compute forces
        pos_flat = pos.reshape(-1, 2)
        dists = cdist(pos_flat, pos_flat)
        np.fill_diagonal(dists, 1e10)
        
        force = np.zeros_like(pos_flat)
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                d = dists[i_idx, j_idx]
                if d < 2 * r_target:
                    direction = pos_flat[i_idx] - pos_flat[j_idx]
                    if d > 1e-12:
                        direction = direction / d
                    else:
                        direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                        direction = direction / (np.linalg.norm(direction) + 1e-12)
                    overlap = 2 * r_target - d
                    force[i_idx] += direction * overlap
                    force[j_idx] -= direction * overlap
        
        # Boundary forces
        for i_idx in range(N):
            for axis in range(2):
                coord = pos_flat[i_idx, axis]
                if coord > 0.5 - r_target:
                    force[i_idx, axis] -= (coord - (0.5 - r_target))
                elif coord < -0.5 + r_target:
                    force[i_idx, axis] += ((-0.5 + r_target) - coord)
        
        # Langevin update: pos += force*dt + random noise
        noise = np.random.normal(0, np.sqrt(2 * temp * dt), size=pos.shape)
        pos = pos + force * dt + noise
        
        # Clamp to square
        pos = np.clip(pos, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check improvement
        r = _radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
    
    return best_pos, best_r

def _generate_hexagonal(N_circles):
    """Generate a hexagonal close-packed initial configuration."""
    # Hexagonal lattice spacing
    # For N circles, arrange in rows
    # Each row has alternating offsets
    positions = []
    # Try different numbers of rows
    best_config = None
    best_r = 0
    
    for n_rows in range(1, N_circles + 1):
        n_cols = int(np.ceil(N_circles / n_rows))
        # Hexagonal spacing
        dx = 1.0 / (n_cols + 0.5)
        dy = 1.0 / (n_rows + 0.5)
        
        pos = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = (col + 0.5) * dx - 0.5
                y = (row + 0.5) * dy - 0.5
                if row % 2 == 1:
                    x += dx / 2
                pos.append([x, y])
        
        if len(pos) >= N_circles:
            pos = pos[:N_circles]
            pos = np.array(pos)
            r = _radius(pos)
            if r > best_r:
                best_r = r
                best_config = pos
    
    if best_config is not None:
        return best_config
    else:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N_circles]

def _generate_square_grid(N_circles):
    """Generate a square grid initial configuration."""
    m = int(np.ceil(np.sqrt(N_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_circles]

def _generate_random(N_circles, seed=42):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-0.4, 0.4, size=(N_circles, 2))
    return pos

def _generate_jittered_grid(N_circles, seed=42):
    """Generate a jittered grid initial configuration."""
    base = _generate_square_grid(N_circles)
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(-0.02, 0.02, size=base.shape)
    pos = base + jitter
    pos = np.clip(pos, -0.45, 0.45)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    best_pos = None
    best_r = 0.0
    
    # Generate multiple initial configurations
    initializations = []
    
    # 1. Square grid
    initializations.append(_generate_square_grid(N))
    
    # 2. Hexagonal
    initializations.append(_generate_hexagonal(N))
    
    # 3. Random
    for seed in [42, 123, 456, 789]:
        initializations.append(_generate_random(N, seed))
    
    # 4. Jittered grid
    for seed in [42, 123, 456, 789]:
        initializations.append(_generate_jittered_grid(N, seed))
    
    # Optimize each initialization
    for init_idx, pos0 in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # First, evaluate the initial configuration
        r_init = _radius(pos0)
        if r_init > best_r:
            best_r = r_init
            best_pos = pos0.copy()
        
        # Try gradient descent
        pos_gd, r_gd = _gradient_descent(pos0, r_init, max_iter=3000, lr=0.002, deadline=deadline)
        if r_gd > best_r:
            best_r = r_gd
            best_pos = pos_gd.copy()
        
        # Try Langevin dynamics
        pos_ld, r_ld = _langevin_dynamics(pos0, best_r, max_iter=15000, dt=0.00005, temp0=0.005, temp_end=0.0001, deadline=deadline)
        if r_ld > best_r:
            best_r = r_ld
            best_pos = pos_ld.copy()
        
        # Refine with gradient descent from the best found so far
        if best_pos is not None and time.time() < deadline:
            pos_refine, r_refine = _gradient_descent(best_pos, best_r, max_iter=5000, lr=0.001, deadline=deadline)
            if r_refine > best_r:
                best_r = r_refine
                best_pos = pos_refine.copy()
    
    # Final refinement pass
    if best_pos is not None and time.time() < deadline:
        # Multiple refinement passes
        for _ in range(3):
            if time.time() > deadline:
                break
            pos_final, r_final = _gradient_descent(best_pos, best_r, max_iter=10000, lr=0.0005, deadline=deadline)
            if r_final > best_r:
                best_r = r_final
                best_pos = pos_final.copy()
            else:
                break
    
    if best_pos is None:
        # Fallback
        best_pos = _generate_square_grid(N)
    
    # Ensure all points are strictly inside the square
    best_pos = np.clip(best_pos, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_pos
# EVOLVE-BLOCK-END
