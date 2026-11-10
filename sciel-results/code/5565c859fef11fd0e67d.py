"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit
    
    # Initial random positions inside the square [-0.5, 0.5]^2
    # Using a jittered grid to avoid initial overlaps which cause huge repulsive forces
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    positions = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    positions += np.random.uniform(-0.01, 0.01, size=positions.shape)
    
    # Clamp to ensure they are inside
    positions = np.clip(positions, -0.5, 0.5)
    
    # Simulation parameters
    dt = 0.005
    temperature = 0.05
    damping = 0.9
    repulsion_strength = 1.0
    wall_strength = 1.0
    
    best_positions = positions.copy()
    best_radius = _compute_radius(positions)
    
    # Precompute indices for distance calculations
    # We will do this in chunks or vectorized fully if memory permits.
    # N=200 is small enough for full NxN distance matrix (40k entries)
    
    step = 0
    while time.time() < deadline:
        # Calculate forces
        # Distances between all pairs
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Repulsive force: F ~ 1/d^2 (stiff)
        # We want to maximize min distance. 
        # A common potential is U = 1/d^2, F = -grad U = 2/d^3 * r_vec
        # Let's use a simpler soft repulsion: F = (1/d - r_target) * r_vec? 
        # No, standard sphere packing uses repulsion when d < 2r. 
        # Since r is unknown, we use a potential that pushes them apart always.
        # U = 1/d^2. Force on i from j is -grad_i U = -(-2/d^3 * (r_i - r_j)) = 2/d^3 * (r_i - r_j)
        # Wait, grad_i (1/|r_i - r_j|^2) = -2 (r_i - r_j) / |r_i - r_j|^4.
        # So Force = -grad U = 2 (r_i - r_j) / d^4.
        
        # Normalize by distance to get unit vector, then scale by strength/d^2
        # force_vec = (r_i - r_j) / d * (strength / d^2) = strength * (r_i - r_j) / d^3
        
        inv_dist3 = 1.0 / (dist**3)
        np.fill_diagonal(inv_dist3, 0.0)
        
        forces = repulsion_strength * np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        # Wall forces: Push back if |x| > 0.5 - epsilon or similar
        # Soft wall potential: U_wall = k * (max(0, |x| - (0.5 - margin)))^2
        # Let's use a hard constraint with soft penalty.
        # If x > 0.5, force = -k * (x - 0.5)
        # If x < -0.5, force = -k * (x + 0.5)
        
        # Margin to stay strictly inside
        margin = 0.01
        x, y = positions[:, 0], positions[:, 1]
        
        fx_wall = np.zeros_like(x)
        fy_wall = np.zeros_like(y)
        
        # Right wall
        mask = x > (0.5 - margin)
        fx_wall[mask] -= wall_strength * (x[mask] - (0.5 - margin))
        
        # Left wall
        mask = x < (-0.5 + margin)
        fx_wall[mask] -= wall_strength * (x[mask] - (-0.5 + margin))
        
        # Top wall
        mask = y > (0.5 - margin)
        fy_wall[mask] -= wall_strength * (y[mask] - (0.5 - margin))
        
        # Bottom wall
        mask = y < (-0.5 + margin)
        fy_wall[mask] -= wall_strength * (y[mask] - (-0.5 + margin))
        
        forces[:, 0] += fx_wall
        forces[:, 1] += fy_wall
        
        # Langevin update
        # v = damping * v + (F/m) * dt + random_noise
        # We don't track velocity explicitly in this simple loop, let's add it or use position-based?
        # Velocity Verlet or simple Euler with velocity is better.
        
        # Let's implement simple velocity tracking
        if not hasattr(construct, '_velocities'):
            construct._velocities = np.zeros_like(positions)
        
        velocities = construct._velocities
        
        # Update velocities
        # Noise scaling with temperature
        noise = np.random.normal(0, np.sqrt(temperature * dt), size=positions.shape)
        
        velocities = damping * velocities + (forces * dt) + noise
        
        # Update positions
        positions += velocities * dt
        
        # Hard clamp to prevent escaping the square entirely (safety)
        positions = np.clip(positions, -0.5, 0.5)
        
        # Store best
        current_radius = _compute_radius(positions)
        if current_radius > best_radius:
            best_radius = current_radius
            best_positions = positions.copy()
            
        # Cool down slowly
        temperature *= 0.9999
        if temperature < 0.001:
             temperature = 0.001
             
        step += 1
        
        # Periodic check to ensure we don't spend too long on one eval
        if step % 100 == 0:
             if time.time() > deadline - 5.0:
                 break

    return best_positions

def _compute_radius(positions):
    """Compute the maximum possible radius for given positions."""
    if positions.shape[0] == 0:
        return 0.0
    
    # Min distance to walls
    # Distance from center to nearest wall
    # Square is [-0.5, 0.5]^2
    # dist_wall = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    x, y = positions[:, 0], positions[:, 1]
    dist_wall = np.min([
        0.5 - np.max(x),
        0.5 + np.min(x),
        0.5 - np.max(y),
        0.5 + np.min(y)
    ])
    
    # Min distance between centers
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    # Radius is half the min of these
    r = min(dist_wall, min_dist / 2.0)
    
    # Ensure non-negative
    if r < 0:
        return 0.0
    return r
# EVOLVE-BLOCK-END
