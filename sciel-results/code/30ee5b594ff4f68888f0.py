"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _hexagonal_lattice():
    """Generate a good initial hexagonal lattice packing."""
    # Estimate spacing for 300 circles in unit circle
    # Area approx: 300 * pi * r^2 = pi * 1^2 => r ~ 1/sqrt(300) ~ 0.057
    # Diameter d ~ 0.114
    # Hex lattice spacing a = d
    # Try a range of spacings to find best initial count/fit
    best_r = 0.0
    best_points = None
    
    for spacing in np.linspace(0.05, 0.12, 50):
        a = spacing
        h = a * np.sqrt(3) / 2
        
        points = []
        # Generate grid large enough to cover unit circle
        for j in range(-20, 21):
            for i in range(-20, 21):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                # Check if inside unit circle with margin
                if x*x + y*y < 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            continue
            
        points = np.array(points)
        
        # If we have more than N points, select N closest to center
        if len(points) > N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            selected = points[idx]
        elif len(points) == N:
            selected = points
        else:
            # If too few, we can't use this spacing directly without adding points
            # But for initialization, let's just skip if significantly fewer
            continue
            
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
            
    return best_points, best_r

def _md_optimize(points, max_time=140.0):
    """
    Optimize packing using Molecular Dynamics (Repulsive Forces).
    """
    if points is None:
        return None, 0.0
        
    start_time = time.time()
    deadline = start_time + max_time
    
    pos = points.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    
    # Initial radius estimate
    current_r = _compute_radius(pos)
    best_pos = pos.copy()
    best_r = current_r
    
    # Parameters
    k_rep = 100.0      # Repulsion strength
    k_wall = 50.0      # Wall repulsion strength
    dt = 0.005         # Time step
    damping = 0.98     # Velocity damping for cooling
    cool_rate = 0.999  # Cooling rate for dt or forces
    
    # Precompute indices for efficiency if needed, but vectorized ops are fast enough for N=300
    
    for step in range(100000):
        if time.time() > deadline - 5:
            break
            
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion: F = k_rep * (1/d^2) * r_hat  (for d < 2r)
        # To make it robust, we use a soft repulsion that scales with overlap
        # Actually, for packing, we want to maximize min distance.
        # A common potential is U = k * (1/d)^12 - k * (2/d)^6 (Lennard-Jones like)
        # Or simpler: U = k * max(0, 2r - d)^2
        # Since r is changing, it's better to use a force that pushes apart if too close.
        # Let's use: F_ij = k_rep / d^2 * (r_vec) if d < threshold, else 0?
        # Better: Use a potential that minimizes max overlap.
        # Let's try a simple repulsive force: F = k * (1/d^2) * unit_vec
        # This pushes them apart. The equilibrium will be determined by the container.
        
        # Vectorized pairwise calculation
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        # This creates a 300x300 matrix, which is fine.
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-10
        dist = np.sqrt(dist_sq)
        
        # Force magnitude: k / d^2
        # Only apply if d < some cutoff? No, let's apply everywhere but dampen long range?
        # For packing, long range repulsion helps spread them out.
        # F_vec = k * diff / d^3
        force_mag = k_rep / dist_sq
        # Zero out self-interaction
        np.fill_diagonal(force_mag, 0.0)
        
        # F_ij = force_mag * diff / d
        # force_mag shape (N,N), diff shape (N,N,2)
        # We need to divide diff by dist (N,N,1)
        dist_reshaped = dist[:, :, np.newaxis]
        # Avoid div by 0 again
        dist_reshaped[dist_reshaped == 0] = 1e-10
        
        force_vectors = force_mag[:, :, np.newaxis] * diff / dist_reshaped
        # Sum forces on each particle i from all j
        forces += np.sum(force_vectors, axis=1)
        
        # Wall repulsion: F_wall = k_wall * (1 - r) * (pos/r) for r > 1 - margin?
        # Actually, force should push inward if pos > 1.
        # r_vec = pos / |pos|
        # F = -k_wall * max(0, |pos| - (1 - r_est)) * r_vec ?
        # Let's use a potential: U_wall = k_wall * max(0, |pos| - 1)^2
        # F = -grad U = -2 * k_wall * max(0, |pos| - 1) * r_vec
        # But we want them inside.
        # Let's use a soft wall: F = -k_wall * pos / |pos| * max(0, |pos| - (1 - current_r))
        # This pushes them back if they go beyond (1 - current_r).
        # However, current_r changes.
        # Simpler: Push back if |pos| > 0.95 (approx boundary)
        # F = -k_wall * pos * max(0, |pos| - 0.95)
        
        norms = np.linalg.norm(pos, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        unit_vecs = pos / norms
        
        # Penalty if norm > 0.95
        excess = np.maximum(0.0, norms - 0.95)
        wall_forces = -k_wall * excess * unit_vecs
        forces += wall_forces
        
        # Update velocity (Velocity Verlet)
        vel = (vel + forces * dt) * damping
        pos_new = pos + vel * dt
        
        # Check validity and update best
        # Ensure no point is outside unit circle
        # If any point is outside, we might need to project it back or penalize heavily.
        # The wall force should prevent this, but let's be safe.
        norms_new = np.linalg.norm(pos_new, axis=1)
        if np.any(norms_new > 1.0):
            # Project back to unit circle
            mask = norms_new > 1.0
            pos_new[mask] = pos_new[mask] / norms_new[mask, np.newaxis]
            vel[mask] = 0.0 # Kill velocity for violated points
            
        r_new = _compute_radius(pos_new)
        
        if r_new > best_r:
            best_r = r_new
            best_pos = pos_new.copy()
            
        pos = pos_new
        
        # Cool down
        dt *= cool_rate
        k_rep *= cool_rate
        
    return best_pos, best_r

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Generate initial hexagonal lattice
    hex_points, hex_r = _hexagonal_lattice()
    
    if hex_points is None:
        # Fallback: Random points
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        hex_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        hex_r = _compute_radius(hex_points)
        
    # 2. Optimize with MD
    optimized_points, opt_r = _md_optimize(hex_points, max_time=140.0)
    
    # 3. Final check
    if optimized_points is not None and opt_r > hex_r:
        return optimized_points
    else:
        return hex_points
# EVOLVE-BLOCK-END
