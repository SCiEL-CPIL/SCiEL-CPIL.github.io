import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _concentric_packing():
    """
    Generate an initial packing using concentric rings.
    This provides a structured start that is different from a simple hex grid.
    """
    # We will try to fit rings with increasing radius.
    # The number of circles in a ring k (radius r_k) is roughly proportional to r_k.
    # Let's assign circles to rings.
    
    # Heuristic: 
    # Ring 0: 1 circle at center? Or maybe skip center if it limits radius.
    # For N=50, let's try a configuration like:
    # Ring 0: 1 circle (at 0,0)
    # Ring 1: 6 circles
    # Ring 2: 12 circles
    # Ring 3: 18 circles
    # Ring 4: 13 circles
    # Total: 1+6+12+18+13 = 50
    
    ring_counts = [1, 6, 12, 18, 13]
    total = sum(ring_counts)
    if total != N:
        # Fallback to hex if counts don't match (shouldn't happen with hardcoded N=50)
        return _hexagonal_fallback()
        
    centers = []
    
    # Estimate optimal ring radii.
    # The spacing between rings should be approx 2*r.
    # The angular spacing in ring k should be approx 2*r / r_k.
    # Let's start with a guess for r, say 0.12.
    # Then ring radii are approx 0, 0.24, 0.48, 0.72, 0.96? 
    # But boundary constraint: r_k + r <= 1 => r_k <= 1-r.
    # If r=0.125, max r_k = 0.875.
    # Let's distribute radii more evenly or based on density.
    
    # A better heuristic for radii:
    # Area of circle = pi * r_container^2.
    # Area of N circles = N * pi * r^2.
    # Packing density ~ 0.9.
    # N * r^2 ~ 0.9 * 1^2 => r ~ sqrt(0.9/50) ~ sqrt(0.018) ~ 0.134.
    # Best known is ~0.1258.
    
    # Let's place rings at radii that maximize the minimum distance.
    # We'll use a simple optimization for the ring radii and angles later in annealing.
    # For initialization, let's use equal angular spacing and radii that fill the disk.
    
    # Radii guess:
    # We want to place 1, 6, 12, 18, 13 circles.
    # Let's try radii: 0, 0.3, 0.55, 0.75, 0.9
    # This might be too tight or too loose. 
    # Let's use a more systematic approach: 
    # r_k = k * spacing.
    # But the last ring must be <= 1 - r_est.
    
    # Let's just use a standard hexagonal-like radial placement.
    # Ring 0: (0,0)
    # Ring 1: 6 points at dist d1
    # Ring 2: 12 points at dist d2
    # Ring 3: 18 points at dist d3
    # Ring 4: 13 points at dist d4
    
    # Let's estimate d_i such that distances are roughly uniform.
    # d1 ~ 0.25
    # d2 ~ 0.5
    # d3 ~ 0.75
    # d4 ~ 0.9
    
    radii = [0.0, 0.25, 0.5, 0.75, 0.9]
    
    for k, count in enumerate(ring_counts):
        r_k = radii[k]
        if count == 1:
            centers.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Add a phase shift for odd rings to interleave
            if k % 2 == 1:
                angles += np.pi / count
            for a in angles:
                x = r_k * np.cos(a)
                y = r_k * np.sin(a)
                centers.append([x, y])
                
    return np.array(centers)

def _hexagonal_fallback():
    """Fallback hexagonal packing if concentric fails or for comparison."""
    spacing = 0.25
    points = []
    # Generate a grid of hexagonal points
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0 - spacing/2:
                points.append([x, y])
    points = np.array(points)
    # Sort by distance to center and take first N
    if len(points) >= N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        return points[idx]
    return np.array(points) # Should not happen

def _build_grid(centers, cell_size):
    """Build a spatial grid for neighbor lookup."""
    grid = {}
    for i, (x, y) in enumerate(centers):
        gx = int(np.floor(x / cell_size))
        gy = int(np.floor(y / cell_size))
        key = (gx, gy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid

def _get_neighbors(grid, x, y, cell_size, search_radius):
    """Get neighbors within search_radius using grid."""
    neighbors = []
    gx = int(np.floor(x / cell_size))
    gy = int(np.floor(y / cell_size))
    n_cells = int(np.ceil(search_radius / cell_size))
    for dx in range(-n_cells, n_cells + 1):
        for dy in range(-n_cells, n_cells + 1):
            key = (gx + dx, gy + dy)
            if key in grid:
                neighbors.extend(grid[key])
    return neighbors

def _calc_forces(centers, r_target):
    """
    Calculate repulsive forces for pairs that are closer than 2*r_target.
    Also apply inward force for circles too close to the boundary.
    """
    forces = np.zeros_like(centers)
    cell_size = 0.2
    grid = _build_grid(centers, cell_size)
    search_radius = 0.5 # Should cover most interactions
    
    for i in range(N):
        xi, yi = centers[i]
        neighbors = _get_neighbors(grid, xi, yi, cell_size, search_radius)
        
        for j in neighbors:
            if i == j:
                continue
            xj, yj = centers[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            
            if dist < 1e-6:
                # Random nudge if coincident
                forces[i, 0] += np.random.uniform(-0.01, 0.01)
                forces[i, 1] += np.random.uniform(-0.01, 0.01)
                continue
            
            # If dist < 2*r_target, they are overlapping or too close.
            # We want to push them apart.
            # Force magnitude proportional to overlap depth.
            min_dist = 2 * r_target
            if dist < min_dist:
                # Overlap
                overlap = min_dist - dist
                # Force direction is away from j
                f_mag = overlap / (dist + 1e-6)
                forces[i, 0] += dx * f_mag
                forces[i, 1] += dy * f_mag
            else:
                # Even if not overlapping, we might want a weak repulsion 
                # to spread them out if they are very close to the limit.
                # But for speed, let's only act on overlaps or very close pairs.
                # Let's add a weak repulsion if dist < min_dist * 1.2
                if dist < min_dist * 1.2:
                    f_mag = 0.1 / (dist + 1e-6)
                    forces[i, 0] += dx * f_mag
                    forces[i, 1] += dy * f_mag

        # Boundary constraint
        dist_center = np.sqrt(xi**2 + yi**2)
        if dist_center > 1e-6:
            margin = 1.0 - dist_center
            if margin < r_target:
                # Push inward
                f_mag = (r_target - margin) / (dist_center + 1e-6)
                forces[i, 0] -= xi * f_mag
                forces[i, 1] -= yi * f_mag
            elif margin < r_target * 1.2:
                # Weak inward push
                f_mag = 0.1 * (r_target * 1.2 - margin) / (dist_center + 1e-6)
                forces[i, 0] -= xi * f_mag
                forces[i, 1] -= yi * f_mag
                
    return forces

def _anneal(centers, deadline):
    """
    Simulated Annealing to optimize the packing.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Initial temperature
    T = 0.1
    # Cooling rate
    cooling = 0.995
    min_T = 1e-5
    
    # Step size for moves
    step_scale = 0.05
    
    while time.time() < deadline and T > min_T:
        # Generate a move
        # Pick a random subset of points to move (or all)
        # For speed, move all points with a small perturbation
        # Or move a random point.
        # Let's try moving all points with a small Gaussian noise, scaled by T.
        
        # Actually, a more effective move for packing is to move points in the direction of the force.
        # Let's use the force as the primary direction, plus noise.
        
        forces = _calc_forces(current, current_r)
        
        # Add noise
        noise = np.random.normal(0, T * step_scale, size=(N, 2))
        
        # Move
        new_centers = current + forces * 0.1 + noise
        
        # Check validity and compute new radius
        new_r = _compute_radius(new_centers)
        
        if new_r > 0:
            # Accept if better, or with probability exp(-delta_r / T)
            if new_r > current_r:
                current = new_centers
                current_r = new_r
                if current_r > best_r:
                    best = current.copy()
                    best_r = current_r
            else:
                delta_r = current_r - new_r
                # Probability of accepting a worse state
                # Note: delta_r is positive for worse states.
                # We want to accept if exp(-delta_r / T) > random
                # If delta_r is large, probability is small.
                # If T is small, probability is small.
                # We need to scale delta_r appropriately.
                # Radius changes are small, e.g. 1e-4. T is 0.1.
                # exp(-1e-4 / 0.1) = exp(-0.001) ~ 0.999.
                # This means we accept almost all worse moves initially.
                # As T drops, we become stricter.
                
                # Let's scale delta_r by 100 to make the effect more pronounced?
                # Or just use it as is.
                prob = np.exp(-delta_r / T)
                if np.random.rand() < prob:
                    current = new_centers
                    current_r = new_r
        
        # Cool down
        T *= cooling
        
    return best, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with concentric packing
    centers = _concentric_packing()
    
    # Ensure initial validity
    r_init = _compute_radius(centers)
    if r_init <= 0:
        # Fallback to hex if concentric is invalid
        centers = _hexagonal_fallback()
        r_init = _compute_radius(centers)
        
    # Run annealing
    best_centers, best_r = _anneal(centers, deadline)
    
    return best_centers
