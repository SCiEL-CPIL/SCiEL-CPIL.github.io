import numpy as np
import time
import sys

N = 200
TIME_LIMIT = 110.0  # Safety margin from 120s
START_TIME = time.time()

def _compute_radius(centres):
    """Compute packing radius. Returns r if valid, 0.0 if invalid."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)

def _generate_fibonacci_init(n_points):
    """Generate a Fibonacci lattice (sunflower pattern) initialization."""
    # Golden angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # We want points in [-0.5, 0.5]^2.
    # The sunflower pattern usually maps to a unit disk or square.
    # Let's map to the square [-0.5, 0.5] x [-0.5, 0.5].
    # Standard sunflower:
    # r = sqrt(i/N) * R
    # theta = i * golden_angle
    # x = r * cos(theta)
    # y = r * sin(theta)
    # This fills a circle. To fill a square uniformly, we can use the same angle but scale r to fit the square bounds better, or just use the circular distribution and let the optimizer push points to the corners.
    # However, a square-uniform distribution is better.
    # A common trick for uniform square sampling with low discrepancy is using the Halton sequence or similar, but Fibonacci is requested.
    # Let's stick to the classic sunflower but clip/scale to square.
    
    i = np.arange(n_points)
    r = np.sqrt(i / n_points) * 0.5 # Scale to max radius 0.5 (fits in square corner distance ~0.707, so 0.5 is safe inside)
    theta = i * golden_angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack((x, y))
    
    # Add small random jitter to break symmetry and help escape local minima
    points += np.random.randn(n_points, 2) * 0.01
    
    # Clip to valid region
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _force_optimize(centres, time_budget):
    """
    Physics-based optimization using repulsive forces.
    We simulate particles that repel each other if they are too close,
    and are attracted to the center or constrained by the walls.
    """
    deadline = time.time() + time_budget
    best_centres = centres.copy()
    best_r = _compute_radius(centres)
    
    # Simulation parameters
    # We want to maximize the minimum distance.
    # This is equivalent to minimizing a potential energy where energy increases if distance < target.
    # Since we don't know the target, we use a generic repulsive potential: U = 1/d^2 or similar.
    # However, to maximize the *minimum* distance, we can use a "soft" repulsion that is very strong for small d.
    # Let's use a force proportional to 1/d^2 for d < some threshold, or just always repulsive.
    # To keep them inside, we add a force pushing back if they hit the wall.
    
    # Adaptive step size
    dt = 0.005
    damping = 0.9
    
    velocities = np.zeros_like(centres)
    
    # We need to handle the "maximize minimum distance" objective.
    # A simple gradient descent on -min_dist is non-differentiable.
    # A force-based approach:
    # F_ij = k * (r_target - d_ij) / d_ij * unit_vec  if d_ij < r_target
    # But we don't know r_target.
    # Alternative: Minimize sum of 1/d_ij^2? No, that minimizes all distances.
    # We want to maximize the minimum.
    # Let's use a "pressure" model. If we assume the circles are incompressible, we can just let them repel.
    # The system will expand until constrained by walls or other circles.
    # To maximize the *common* radius, we need to push the tightest pair apart.
    # Force method:
    # For each pair, if d_ij < current_best_r * 2 (or some threshold), apply repulsion.
    # Actually, let's just apply repulsion to all pairs with a potential that is very steep.
    # U = sum 1/d_ij^2. Minimizing this tends to spread points out, but might not maximize the *minimum* specifically if one pair is very close and others are far.
    # Better: U = sum max(0, (r_target - d_ij))^2.
    # We can adapt r_target during the process. Start with a small r_target and increase it?
    # Or, simply use the gradient of the soft-min as before, but with a physics loop?
    # The directive says "repulsive force-based simulation that treats circle boundaries as rigid constraints".
    # This implies: if d_ij < 2r, push apart. If wall_dist < r, push in.
    # We can estimate r from the current configuration.
    
    step_count = 0
    while time.time() < deadline:
        # Calculate current bottleneck radius
        from scipy.spatial.distance import cdist
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        wx = 0.5 - np.abs(centres[:, 0])
        wy = 0.5 - np.abs(centres[:, 1])
        wall = np.minimum(wx, wy)
        min_w = np.min(wall)
        
        current_r = min(min_d / 2.0, min_w)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
            
        # If we've hit a local optimum and can't improve, break or jitter
        # Check for stagnation
        if step_count > 100 and best_r < current_r * 1.0001: # heuristic
             # Add noise
             centres += np.random.randn(N, 2) * 0.001
             centres = np.clip(centres, -0.499, 0.499)
             velocities = np.zeros_like(centres)
             step_count = 0
             continue
             
        step_count += 1
        
        # Calculate forces
        forces = np.zeros_like(centres)
        
        # 1. Wall forces
        # If a point is too close to the wall relative to current_r, push it back
        # We use current_r as the target radius.
        # Force is linear or quadratic? Linear is stable.
        wall_force_strength = 10.0
        for i in range(N):
            # X direction
            if centres[i, 0] > 0:
                dist_to_wall = 0.5 - centres[i, 0]
                if dist_to_wall < current_r:
                    forces[i, 0] -= wall_force_strength * (current_r - dist_to_wall)
            else:
                dist_to_wall = 0.5 + centres[i, 0]
                if dist_to_wall < current_r:
                    forces[i, 0] += wall_force_strength * (current_r - dist_to_wall)
            
            # Y direction
            if centres[i, 1] > 0:
                dist_to_wall = 0.5 - centres[i, 1]
                if dist_to_wall < current_r:
                    forces[i, 1] -= wall_force_strength * (current_r - dist_to_wall)
            else:
                dist_to_wall = 0.5 + centres[i, 1]
                if dist_to_wall < current_r:
                    forces[i, 1] += wall_force_strength * (current_r - dist_to_wall)
        
        # 2. Pairwise repulsive forces
        # We only apply force if d_ij < 2 * current_r (i.e., they are overlapping or too close for the current radius)
        # This effectively "inflates" the circles to radius current_r.
        # To improve, we need to allow current_r to grow.
        # The forces will push points apart. If they can move apart without hitting walls/other points, current_r will increase in the next step.
        
        # Vectorized calculation of pairwise forces is tricky with conditional logic.
        # We can use a soft repulsive potential that is active for all d, but very strong for small d.
        # U = 1/d^2. F = 2/d^3 * unit_vec.
        # This is expensive.
        # Let's stick to the cdist we already have.
        
        # dists is (N, N). 
        # We want force on i from j.
        # F_i = sum_j k * (d_target - d_ij) / d_ij * (c_i - c_j) / d_ij  for d_ij < d_target
        # Let d_target = 2 * current_r.
        
        d_target = 2.0 * current_r
        
        # Mask pairs that are too close
        # d_ij < d_target
        # Note: dists has inf on diagonal.
        mask = dists < d_target
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            # Calculate vector from j to i: c_i - c_j
            # diff[i, j] = c_i - c_j
            diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
            
            # Unit vector
            # Avoid division by zero
            inv_d = np.zeros_like(dists)
            valid = dists > 1e-6
            inv_d[valid] = 1.0 / dists[valid]
            
            # Force magnitude factor: k * (d_target - d_ij)
            # We want the force to be zero if d_ij >= d_target.
            # F_mag = k * (d_target - d_ij)
            # Direction: (c_i - c_j) / d_ij
            # So F_vec = k * (d_target - d_ij) / d_ij * (c_i - c_j)
            # F_vec = k * (d_target * inv_d - 1) * (c_i - c_j)
            
            k = 5.0 # Stiffness
            factor = np.zeros_like(dists)
            factor[mask] = k * (d_target - dists[mask]) * inv_d[mask]
            
            # F_i = sum_j factor[i, j] * diff[i, j]
            # factor is (N, N), diff is (N, N, 2)
            forces += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Update velocities and positions
        velocities = damping * velocities + forces * dt
        centres = centres + velocities
        
        # Clip to bounds
        centres = np.clip(centres, -0.499, 0.499)
        
        # Optional: Adaptive step size
        # If we move too much, we might oscillate.
        # Check if we improved significantly.
        
    return best_centres, best_r

def construct():
    start_time = time.time()
    # Hard deadline: 110 seconds from start to ensure we return well before 120s
    deadline = start_time + 110.0
    
    # Generate Fibonacci initialization
    best_centres = _generate_fibonacci_init(N)
    best_r = _compute_radius(best_centres)
    
    # Force-based optimization
    remaining = deadline - time.time()
    if remaining > 10:
        best_centres, best_r = _force_optimize(best_centres, remaining - 5)
    
    # Final validation
    final_r = _compute_radius(best_centres)
    if final_r <= 0:
        # Fallback to grid if something went wrong
        m = 14
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centres
