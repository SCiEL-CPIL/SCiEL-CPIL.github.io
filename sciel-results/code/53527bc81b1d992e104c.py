"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n, seed=42):
    """Initialize points on a hexagonal lattice, clipped to the unit disk."""
    rng = np.random.RandomState(seed)
    
    # Estimate initial spacing
    # Area of disk = pi. Area per circle ~ pi * r^2. 
    # For hex packing, density is pi / (2 * sqrt(3)) * (1/r^2) ? No.
    # Density of hex packing is pi / (2 * sqrt(3)) / (r^2) where r is radius.
    # Let's just start with a reasonable spacing.
    # r ~ 1/sqrt(n) is the upper bound if they were squares.
    # For hex, r is slightly larger.
    # Let's start with spacing s such that we get roughly n points.
    # Area of hexagon with side s/2 is (3*sqrt(3)/2) * (s/2)^2 = 3*sqrt(3)*s^2/8.
    # Number of points ~ Area / Area_per_point ~ pi / (3*sqrt(3)*s^2/8) = 8*pi / (3*sqrt(3)*s^2).
    # n = 300 => s^2 = 8*pi / (3*sqrt(3)*300) ~ 25.13 / (1558) ~ 0.016 => s ~ 0.126.
    # Let's try a range of spacings to ensure we have enough points.
    
    best_points = None
    best_score = -1
    
    for attempt in range(10):
        # Perturb spacing slightly
        base_spacing = 0.12
        spacing = base_spacing * (0.95 + 0.1 * rng.random())
        
        k = int(1.0 / spacing) + 2
        # Generate grid
        i_range = range(-k, k + 1)
        j_range = range(-k, k + 1)
        
        # Vectorized generation
        i_vals = np.array(list(i_range))
        j_vals = np.array(list(j_range))
        I, J = np.meshgrid(i_vals, j_vals, indexing='ij')
        
        x = (I + 0.5 * (J % 2)) * spacing
        y = J * spacing * np.sqrt(3.0) / 2.0
        
        points = np.column_stack((x, y))
        radius = np.hypot(points[:, 0], points[:, 1])
        
        # Clip to disk
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) < n:
            continue
            
        # If we have more than n, pick the ones closest to the center?
        # Or just take the first n? Taking closest to center usually yields a more compact core.
        # But we want to fill the disk. 
        # Actually, for packing, we want to maximize the minimum distance.
        # A good initial configuration is a subset of the lattice that fills the disk uniformly.
        # Sorting by radius and taking the first n does this.
        points = points[np.argsort(radius, kind="stable")[:n]]
        
        # Add small random perturbation to break symmetry
        points = points + rng.randn(n, 2) * 1e-3
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    if best_points is None:
        # Fallback: simple grid
        points = np.zeros((n, 2))
        # Just place them in a rough circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = np.sqrt(np.random.rand(n))
        points[:, 0] = radii * np.cos(angles)
        points[:, 1] = radii * np.sin(angles)
        best_points = points
        
    return best_points

def _physics_refine(points, time_limit=100.0):
    """Refine packing using a physics-based simulation (spring/repulsion model)."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    p = points.copy()
    n = len(p)
    
    # We want to maximize the minimum distance.
    # This is equivalent to minimizing the potential energy of a system where
    # particles repel each other with a force that increases sharply as they get close.
    # We also need to keep them inside the unit circle.
    
    # Let's use a simple iterative relaxation scheme.
    # At each step, calculate forces.
    # Force between i and j:
    # If d_ij < target_dist, push apart.
    # We don't know the optimal target_dist yet.
    # Instead, we can use a "soft" potential that minimizes the sum of inverse distances?
    # Or minimize the sum of (1/d_ij)^2 for close pairs?
    # A common approach for packing is to minimize the sum of squared overlaps.
    
    # Let's try a gradient descent on a smooth surrogate for the min-distance.
    # Surrogate: E = sum_{i<j} max(0, r_target - d_ij/2)^2 + sum_i max(0, r_target - (1-|p_i|))^2
    # But we don't know r_target.
    
    # Alternative: Minimize the sum of inverse distances: E = sum_{i<j} 1/d_ij.
    # This tends to spread points out.
    # However, it doesn't explicitly enforce the boundary.
    # Add a boundary penalty: sum_i (1/|1-|p_i||) or similar.
    
    # Let's use a force-based approach with adaptive "temperature" or step size.
    # Forces:
    # 1. Pairwise repulsion: F_ij = k * (1/d_ij^2) * (p_i - p_j)/d_ij  (Coulomb-like)
    # 2. Boundary repulsion: F_boundary = k_b * (p_i / |p_i|) * (1/(1-|p_i|)^2)
    
    # To make it converge to a local optimum of the min-distance, we can use
    # a potential that is very steep near the current minimum distance.
    
    # Let's stick to a simpler, robust method:
    # Iterative local optimization:
    # 1. Find the pair (i,j) with minimum distance d_min.
    # 2. Find the point k with minimum boundary distance b_min.
    # 3. If d_min/2 < b_min:
    #    - Move i and j apart slightly along the vector (p_i - p_j).
    # 4. Else:
    #    - Move k towards the center slightly.
    # 5. Repeat.
    
    # This is essentially a specialized Nelder-Mead or gradient method.
    # Let's implement a vectorized version of this "push apart" logic.
    
    step_size = 0.01
    cooling = 0.999
    
    for step in range(100000):
        if time.time() > deadline:
            break
            
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        boundary_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists) / 2.0
        min_boundary = np.min(boundary_dists)
        
        # Determine which constraint is active
        if min_pair_dist < min_boundary:
            # Active pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-12:
                # Coincident points, separate randomly
                p[i] += np.random.randn(2) * 1e-4
                p[j] -= np.random.randn(2) * 1e-4
                continue
            
            # Direction from j to i
            direction = (p[i] - p[j]) / d
            
            # Move i and j apart
            # The amount to move should be small
            delta = step_size * 0.5
            p[i] += direction * delta
            p[j] -= direction * delta
        else:
            # Active boundary
            i = np.argmin(boundary_dists)
            norm = norms[i]
            if norm < 1e-12:
                # At center, move randomly
                p[i] += np.random.randn(2) * 1e-4
                continue
            
            # Direction from center to i
            direction = p[i] / norm
            
            # Move i towards center
            delta = step_size * 0.5
            p[i] -= direction * delta
        
        # Cool down
        step_size *= cooling
        
        # Optional: Check if we are stuck?
        # If step_size becomes very small, we might be at a local optimum.
        if step_size < 1e-6:
            # Try a larger random kick to escape?
            # Or just stop.
            # Let's try a random kick if we are stuck.
            if np.random.random() < 0.01:
                idx = np.random.randint(n)
                p[idx] += np.random.randn(2) * 0.01
                step_size = 0.01 # Reset cooling
                
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Refinement
    # We can run multiple rounds of physics refinement with different initial perturbations
    # or just one long run.
    
    # Run physics refinement
    refined_points = _physics_refine(points, time_limit=120.0)
    refined_score = _compute_score(refined_points)
    
    if refined_score > best_score:
        best_points = refined_points
        best_score = refined_score
        
    # 3. Final polish: Local search on the best points found
    # We can do a few steps of the "push apart" logic with a very small step size
    # to ensure we are at a local optimum.
    
    if time.time() < deadline - 5.0:
        p = best_points.copy()
        step_size = 1e-4
        for step in range(5000):
            if time.time() > deadline - 2.0:
                break
            diff = p[:, np.newaxis, :] - p[np.newaxis, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            norms = np.linalg.norm(p, axis=1)
            boundary_dists = 1.0 - norms
            
            min_pair_dist = np.min(dists) / 2.0
            min_boundary = np.min(boundary_dists)
            
            if min_pair_dist < min_boundary:
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                if i != j:
                    d = dists[i, j]
                    if d > 1e-12:
                        direction = (p[i] - p[j]) / d
                        delta = step_size * 0.5
                        p[i] += direction * delta
                        p[j] -= direction * delta
            else:
                i = np.argmin(boundary_dists)
                norm = norms[i]
                if norm > 1e-12:
                    direction = p[i] / norm
                    delta = step_size * 0.5
                    p[i] -= direction * delta
                    
        polished_score = _compute_score(p)
        if polished_score > best_score:
            best_points = p
            best_score = polished_score

    return best_points
# EVOLVE-BLOCK-END
