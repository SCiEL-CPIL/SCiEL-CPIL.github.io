"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct centers of 100 equal spheres in a unit cube [-0.5, 0.5]^3
    to maximize the minimum distance between any two centers and the
    distance to the cube boundary.
    
    Uses a force-directed relaxation approach:
    1. Initialize with various strategies (grid, random, lattice).
    2. Iteratively apply repulsive forces between points and boundary forces.
    3. Use a cooling schedule to converge to a local optimum.
    4. Keep the best result from multiple restarts.
    """
    np.random.seed(42)
    
    best_centers = None
    best_score = 0.0
    
    def score_function(centers):
        """Compute the packing score (minimum radius)."""
        if centers.shape != (N, 3):
            return 0.0
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        # Distance to boundary
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)

    def relax(centers, iterations=1000, initial_step=0.05):
        """
        Force-directed relaxation.
        centers: (N, 3) array
        """
        centers = centers.copy()
        step = initial_step
        
        for i in range(iterations):
            # Compute pairwise differences
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            mask = dist_matrix < 1e-8
            dist_matrix[mask] = 1e-8
            
            # Repulsive force: F = diff / d^3 (inverse square law for force, but we use 1/d^2 for stability in position updates)
            # We want to maximize min distance.
            # Gradient of min distance is dominated by the closest pairs.
            # A common heuristic is to push apart all pairs with strength inversely proportional to distance squared.
            
            # Normalize diff by distance squared to get direction and magnitude
            # diff / d^2
            inv_d2 = np.zeros_like(dist_matrix)
            inv_d2[~mask] = 1.0 / (dist_matrix[~mask] ** 2)
            
            # Force on i from j is (c_i - c_j) / d^2
            # Sum over j
            forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
            
            # Boundary forces
            # Push points away from boundaries if they are too close
            # Let's use a quadratic penalty for boundary violation or proximity
            # Target: keep points at least 'r' away from boundary? No, we just maximize min dist to boundary.
            # Force: if x > 0, force is -k*x^2? Or just push towards center if close to edge?
            # Simple approach: linear repulsion from walls if within 'margin'
            margin = 0.1
            for k in range(3):
                # Left wall
                d_left = 0.5 + centers[:, k]
                # If d_left < margin, push right
                force_left = np.where(d_left < margin, (margin - d_left) * 10.0, 0.0)
                
                # Right wall
                d_right = 0.5 - centers[:, k]
                # If d_right < margin, push left
                force_right = np.where(d_right < margin, -(margin - d_right) * 10.0, 0.0)
                
                forces[:, k] += force_left + force_right
            
            # Update positions
            # Normalize forces to prevent huge jumps
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            normalized_forces = forces / force_norms
            
            centers += normalized_forces * step
            
            # Clip to cube
            centers = np.clip(centers, -0.5, 0.5)
            
            # Cool down
            step *= 0.995
            
            if step < 1e-6:
                break
        
        return centers

    def run_optimization(initial_centers, iterations=1000):
        """Run relaxation from a given initial configuration."""
        centers = relax(initial_centers, iterations=iterations, initial_step=0.05)
        return centers, score_function(centers)

    # Strategy 1: Jittered Grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_grid = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    jittered = initial_grid + np.random.uniform(-0.01, 0.01, size=(N, 3))
    jittered = np.clip(jittered, -0.5, 0.5)
    
    centers, sc = run_optimization(jittered, iterations=2000)
    if sc > best_score:
        best_score = sc
        best_centers = centers.copy()

    # Strategy 2: Random Initializations
    for restart in range(10):
        initial = np.random.uniform(-0.45, 0.45, size=(N, 3))
        centers, sc = run_optimization(initial, iterations=2000)
        if sc > best_score:
            best_score = sc
            best_centers = centers.copy()

    # Strategy 3: FCC-like Lattice
    # Try to pack into a lattice structure
    # For N=100, a 5x5x4 grid or similar might work well
    # Let's try a few grid shapes
    shapes = [(5, 5, 4), (4, 5, 5), (5, 4, 5), (10, 5, 2), (2, 5, 10)]
    for shape in shapes:
        if np.prod(shape) < N:
            continue
        
        # Create grid points
        nx, ny, nz = shape
        xs = np.linspace(-0.45, 0.45, nx)
        ys = np.linspace(-0.45, 0.45, ny)
        zs = np.linspace(-0.45, 0.45, nz)
        
        grid_pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        
        # Randomly select N points or take the first N
        # Taking first N might bias towards one corner, so random selection is better
        if len(grid_pts) > N:
            idx = np.random.choice(len(grid_pts), N, replace=False)
            initial = grid_pts[idx]
        else:
            initial = grid_pts
        
        # Add jitter
        initial = initial + np.random.uniform(-0.01, 0.01, size=(N, 3))
        initial = np.clip(initial, -0.5, 0.5)
        
        centers, sc = run_optimization(initial, iterations=2000)
        if sc > best_score:
            best_score = sc
            best_centers = centers.copy()

    # Final polish: Run relaxation again on the best solution with finer steps
    if best_centers is not None:
        final_centers = relax(best_centers, iterations=5000, initial_step=0.01)
        final_sc = score_function(final_centers)
        if final_sc > best_score:
            best_score = final_sc
            best_centers = final_centers

    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
