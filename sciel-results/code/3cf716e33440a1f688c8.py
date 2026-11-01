"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if len(centers) < 2:
        return 0.5
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _objective_and_constraints(centers, r_target):
    """Not used directly; we use a different approach."""
    pass


def _soft_min(values, temperature=0.01):
    """Soft minimum using log-sum-exp trick."""
    # For soft min: -log(sum(exp(-x/T))) * T
    scaled = -values / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum * temperature


def _repulsion_anneal(centers, n_steps=200, n_inner=10, dt_start=0.01, dt_end=0.001):
    """Repulsion-based annealing to push spheres apart."""
    centers = centers.copy()
    n = len(centers)
    
    for step in range(n_steps):
        # Annealing schedule
        t = step / max(n_steps - 1, 1)
        dt = dt_start * (dt_end / dt_start) ** t
        
        # Compute forces
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random push
                    diff = np.random.randn(3) * 0.001
                    dist = np.linalg.norm(diff)
                if dist < 0.3:  # Only interact when close
                    force_mag = 1.0 / (dist ** 2 + 1e-6)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
            
            # Wall forces
            for k in range(3):
                for boundary in [0.5, -0.5]:
                    dist_to_wall = abs(centers[i, k] - boundary)
                    if dist_to_wall < 0.1:
                        force_mag = 1.0 / (dist_to_wall ** 2 + 1e-6)
                        if centers[i, k] > boundary:
                            forces[i, k] -= force_mag
                        else:
                            forces[i, k] += force_mag
        
        # Update positions
        centers += dt * forces
        
        # Keep within bounds
        centers = np.clip(centers, -0.49, 0.49)
    
    return centers


def _gradient_optimize(centers, max_iter=500, tol=1e-8):
    """Gradient-based optimization to maximize the minimum distance."""
    centers = centers.copy()
    n = len(centers)
    
    def objective(x):
        """Minimize negative soft-min of distances."""
        c = x.reshape(n, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        wall_flat = wall_dists.flatten()
        
        # Combine all distances (pairwise and wall)
        all_dists = np.concatenate([dists[np.triu_indices(n, k=1)], wall_flat])
        
        # Soft min
        temp = 0.01
        return _soft_min(all_dists, temp)
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    result = minimize(
        objective,
        centers.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
    )
    
    return result.x.reshape(n, 3)


def _local_polish(centers, n_iterations=100):
    """Local polish using direct optimization of the hard min."""
    centers = centers.copy()
    n = len(centers)
    
    for _ in range(n_iterations):
        # Find the critical pair/wall
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(wall_dists)
        i_wall = min_wall_idx // 3
        k_wall = min_wall_idx % 3
        min_wall_val = wall_dists[i_wall, k_wall]
        
        min_pair_idx = np.argmin(dists)
        i_pair = min_pair_idx // n
        j_pair = min_pair_idx % n
        min_pair_val = dists[i_pair, j_pair]
        
        # Determine which constraint is binding
        if min_pair_val / 2.0 < min_wall_val:
            # Pair constraint is binding
            diff = centers[i_pair] - centers[j_pair]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            direction = diff / dist
            # Move apart
            centers[i_pair] += 0.001 * direction
            centers[j_pair] -= 0.001 * direction
        else:
            # Wall constraint is binding
            # Move away from wall
            if centers[i_wall, k_wall] > 0:
                centers[i_wall, k_wall] -= 0.001
            else:
                centers[i_wall, k_wall] += 0.001
        
        # Keep within bounds
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers


def _bcc_initialization():
    """BCC lattice initialization."""
    # BCC has points at corners and body center
    # For 50 spheres, use a subset of BCC-like positions
    centers = []
    
    # Start with a grid and add body centers
    m = 4  # 4x4x4 grid gives 64 positions, take 50
    for x in range(m):
        for y in range(m):
            for z in range(m):
                # Corner-like position
                cx = (x + 0.5) / m - 0.5
                cy = (y + 0.5) / m - 0.5
                cz = (z + 0.5) / m - 0.5
                centers.append([cx, cy, cz])
    
    centers = np.array(centers[:N])
    return centers


def _random_initialization():
    """Random initialization with some structure."""
    # Start with a grid and add random perturbation
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.randn(N, 3) * 0.02
    centers = np.clip(centers, -0.45, 0.45)
    return centers


def construct():
    """Main entry point: find optimal packing of 50 spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Choose initialization
        if restart == 0:
            # BCC-like
            centers = _bcc_initialization()
        elif restart == 1:
            # Random perturbed grid
            centers = _random_initialization()
        else:
            # More random
            centers = np.random.uniform(-0.4, 0.4, (N, 3))
        
        # Repulsion annealing
        if time.time() < deadline - 30:
            centers = _repulsion_anneal(centers, n_steps=150, n_inner=10, 
                                         dt_start=0.008, dt_end=0.0005)
        
        # Gradient optimization
        if time.time() < deadline - 20:
            centers = _gradient_optimize(centers, max_iter=300, tol=1e-7)
        
        # Local polish
        if time.time() < deadline - 10:
            centers = _local_polish(centers, n_iterations=80)
        
        # Evaluate
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # If we didn't find anything, use the grid fallback
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
