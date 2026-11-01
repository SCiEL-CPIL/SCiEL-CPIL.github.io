"""Simulated Annealing packing with force-directed initialization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    
    # Wall distances: min distance to any wall
    # Square is [-0.5, 0.5]^2
    # dist to wall = min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5) for each point
    # Actually, min(0.5 - |x|, 0.5 - |y|) is the distance to the nearest wall for a point at (x,y)
    # Wait, the circle must be inside. So x + r <= 0.5 and x - r >= -0.5 => r <= 0.5 - |x|
    # Same for y.
    # So r_wall_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # Global r_wall = min_i r_wall_i
    
    wall_dists = np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1]))
    r_wall = np.min(wall_dists)
    
    # Pairwise distances
    # r_pair = min_{i<j} |c_i - c_j| / 2
    if n > 1:
        # Efficient pairwise distance calculation
        # diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists, np.inf)
        # r_pair = np.min(dists) / 2.0
        
        # To save memory/time for 50 points, simple loops or vectorized is fine.
        # 50x50 matrix is small.
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
    else:
        r_pair = 0.5 # Max possible is 0.5
        
    return min(r_wall, r_pair)

def _construct_initial():
    """
    Generate a good initial configuration using a force-directed layout 
    on a grid perturbation to avoid symmetry traps.
    """
    # Start with a grid
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid of m x m points
    # We only need N points.
    # Let's create a slightly perturbed grid to break symmetry.
    rng = np.random.default_rng(42)
    
    # Grid coordinates
    x = (np.arange(m) + 0.5) / m - 0.5
    y = (np.arange(m) + 0.5) / m - 0.5
    grid_x, grid_y = np.meshgrid(x, y)
    centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))[:N]
    
    # Add small random noise to break symmetry
    centers += rng.normal(0, 0.01, size=centers.shape)
    
    # Clip to square
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return centers

def _force_step(cents, r_target, rng):
    """
    Move points based on repulsive forces to increase the minimum distance.
    This is a heuristic step to guide the annealing.
    """
    n = len(cents)
    forces = np.zeros_like(cents)
    
    # Calculate pairwise forces (repulsive)
    # We want to push points apart if they are closer than 2*r_target
    # F = k * (1/d^2) * direction
    
    # Vectorized pairwise differences
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Force magnitude: 1/d^2 if d < 2*r_target, else 0
    # Actually, we want to maximize the min distance.
    # A common heuristic is to push all points apart proportional to 1/d^2.
    # Let's try a simpler "relaxation" step.
    # For each point, sum up vectors from neighbors, weighted by 1/d^2.
    
    # Avoid division by zero (diagonal is inf, so 1/inf = 0)
    inv_dist_sq = 1.0 / (dists**2)
    # Mask diagonal just in case
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Force on i is sum_j (c_i - c_j) / |c_i - c_j|^3
    # = sum_j diff[i,j] / dists[i,j]^3
    # diff is (i,j) vector.
    # We need diff / dist^3.
    # dist^3 = dists * dist^2.
    # So factor = 1 / (dists * dists^2) = 1/dists^3.
    
    # Handle zeros in dists (shouldn't happen with inf diagonal, but safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        force_factor = np.where(dists > 1e-6, 1.0 / (dists**3), 0.0)
    
    # Force vector for i is sum_j force_factor[i,j] * diff[i,j]
    # diff[i,j] = c_i - c_j
    # So forces[i] = sum_j (1/d^3) * (c_i - c_j)
    
    forces = np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    
    # Add wall repulsion
    # If a point is close to a wall, push it inwards.
    # x < -0.5 + r_target: push right (positive x)
    # x > 0.5 - r_target: push left (negative x)
    
    margin = r_target
    # x forces
    forces[:, 0] += np.where(cents[:, 0] < -0.5 + margin, 1.0, 0.0)
    forces[:, 0] -= np.where(cents[:, 0] > 0.5 - margin, 1.0, 0.0)
    
    # y forces
    forces[:, 1] += np.where(cents[:, 1] < -0.5 + margin, 1.0, 0.0)
    forces[:, 1] -= np.where(cents[:, 1] > 0.5 - margin, 1.0, 0.0)
    
    return forces

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initial configuration
    cents = _construct_initial()
    best_cents = cents.copy()
    best_r = _score(best_cents)
    
    # Simulated Annealing Parameters
    current_r = best_r
    T = 0.05  # Initial temperature
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    rng = np.random.default_rng(123)
    
    # We will use a hybrid approach:
    # 1. Force-directed relaxation to get close to a local optimum.
    # 2. SA to escape local optima.
    
    # Pre-allocate for speed
    # n = N
    
    while time.time() - start_time < deadline:
        # Generate a candidate move
        # Mix of random perturbation and force-directed step
        
        # Random perturbation
        step_size = T * 0.1
        perturbation = rng.normal(0, step_size, size=cents.shape)
        
        # Force-directed step (scaled)
        # We compute forces based on current best_r to guide movement
        # But forces depend on current positions.
        # To save time, we only do force step every few iterations or if random fails?
        # Let's just do random perturbation for SA, it's simpler and robust.
        # However, pure random SA is slow.
        # Let's try "Adaptive" step size based on current T.
        
        new_cents = cents + perturbation
        
        # Keep inside square (hard constraint for validity, but we can allow slight violation in score calc? 
        # No, score handles it. But if points go far outside, score drops.
        # Better to clip to [-0.5, 0.5] to keep them relevant?
        # If we clip, we lose gradient info.
        # Let's just let them go, score will punish it.
        # But if they go to 10.0, next step might be hard to recover.
        # Clip to [-1, 1] to prevent explosion.
        new_cents = np.clip(new_cents, -1.0, 1.0)
        
        new_r = _score(new_cents)
        
        # Metropolis Criterion
        delta_r = new_r - current_r
        if delta_r > 0 or (T > T_min and rng.random() < np.exp(delta_r / T)):
            # Accept
            cents = new_cents
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_cents = cents.copy()
        
        # Cool down
        T *= alpha
        
        # If T is very low, we are stuck in a local minimum.
        # Occasionally "reheat" or do a larger jump?
        # Simple reheat:
        if T < T_min:
            T = 0.01 # Small reheat to explore
            # Or restart from best?
            # Let's just reheat slightly to allow small moves.
            
    return best_cents
# EVOLVE-BLOCK-END
