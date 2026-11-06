"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    n = len(centers)
    # Minimum distance between any pair of centers
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Minimum distance from any center to cube boundary
    min_boundary = float('inf')
    for c in centers:
        for k in range(3):
            dist_to_boundary = 0.5 - abs(c[k])
            if dist_to_boundary < min_boundary:
                min_boundary = dist_to_boundary
    
    return min(min_pair_dist / 2.0, min_boundary)

def _optimize(centers, time_budget):
    """Iteratively optimize sphere positions using repulsion forces."""
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Parameters
    learning_rate = 0.05
    decay = 0.999
    min_force = 1e-10
    
    while time.time() - start_time < time_budget:
        # Compute forces on each center
        forces = np.zeros_like(centers)
        
        # Repulsion from other spheres
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    # Coincident centers - add random perturbation
                    diff = np.random.randn(3) * 1e-3
                    dist = np.linalg.norm(diff)
                
                # Strong repulsion when close
                force_mag = 1.0 / (dist ** 2 + 1e-6)
                force_vec = force_mag * diff / dist
                
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Boundary repulsion (push inward if too close to wall)
        margin = 0.02
        for i in range(n):
            for k in range(3):
                if centers[i][k] > 0.5 - margin:
                    forces[i][k] -= (centers[i][k] - (0.5 - margin)) * 10.0
                elif centers[i][k] < -0.5 + margin:
                    forces[i][k] += ((-0.5 + margin) - centers[i][k]) * 10.0
        
        # Apply forces with learning rate
        centers += learning_rate * forces
        
        # Clamp to cube with small margin
        centers = np.clip(centers, -0.49, 0.49)
        
        # Compute current radius
        current_radius = _compute_radius(centers)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Decay learning rate
        learning_rate *= decay
        
        # Early exit if forces are very small
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < min_force:
            break
    
    return best_centers, best_radius

def construct():
    """Construct 50 sphere centers in a unit cube for maximum radius."""
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Try multiple random initializations and keep the best
    best_centers = None
    best_radius = 0.0
    
    # Initial configuration: jittered grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial = grid[:N].copy()
    
    # Add small random jitter
    np.random.seed(42)
    initial += np.random.randn(N, 3) * 0.01
    
    # Clamp to valid range
    initial = np.clip(initial, -0.45, 0.45)
    
    # Optimize from this initial configuration
    time_remaining = deadline - (time.time() - start_time)
    if time_remaining > 5:
        centers, radius = _optimize(initial, time_remaining - 5)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
    
    # Try a few more random starts if time permits
    for attempt in range(3):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random initialization
        rand_centers = np.random.uniform(-0.4, 0.4, (N, 3))
        
        time_remaining = deadline - (time.time() - start_time)
        if time_remaining < 5:
            break
        
        centers, radius = _optimize(rand_centers, time_remaining - 5)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
    
    # If we still have time, do a final fine-tuning
    if best_centers is not None and time.time() - start_time < deadline - 5:
        time_remaining = deadline - (time.time() - start_time) - 5
        if time_remaining > 2:
            centers, radius = _optimize(best_centers, time_remaining)
            if radius > best_radius:
                best_radius = radius
                best_centers = centers
    
    # Final safety check: ensure all centers are strictly inside the cube
    if best_centers is not None:
        best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers if best_centers is not None else initial
# EVOLVE-BLOCK-END
