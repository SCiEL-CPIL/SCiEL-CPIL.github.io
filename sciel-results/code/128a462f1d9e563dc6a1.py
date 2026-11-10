import numpy as np

def construct():
    N = 300
    
    # 1. Initialization: Hexagonal grid
    # Target density approx 300 points in unit circle.
    # Area = pi. Cell area for hex grid with spacing s is s^2 * sqrt(3)/2.
    # 300 = pi / (s^2 * sqrt(3)/2) => s^2 = 2*pi / (300*sqrt(3)) ~ 0.0121 => s ~ 0.11
    s = 0.11
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y < 0.95: # Keep well inside initially
                points.append((x, y))
    
    points = np.array(points)
    
    # If we have too many, subsample. If too few, pad.
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points
        while len(points) < N:
            p = np.random.uniform(-0.8, 0.8, 2)
            if np.linalg.norm(p) < 0.9:
                points = np.vstack([points, p])
                
    # Add small random jitter to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    # Ensure inside
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 0.95
    points[mask] = points[mask] * (0.95 / radii[mask])[:, np.newaxis]

    # 2. Simulated Annealing
    # Energy: Sum of inverse squared distances + boundary penalty
    # We want to MINIMIZE energy.
    
    def calculate_energy(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero
        
        # Pairwise energy: 1 / dist_sq
        # Add small epsilon to avoid inf
        pair_energy = np.sum(1.0 / (dist_sq + 1e-6))
        
        # Boundary energy: 1 / (1 - r)^2
        radii = np.linalg.norm(pts, axis=1)
        # If radius > 1, penalty is huge. 
        # We want to keep points inside. 
        # Let's use max(0, 1 - (1-radii))^2? 
        # Simpler: if radii > 0.99, penalize heavily.
        # Let's use a smooth potential: 1 / (1 - radii + eps)^2
        # But if radii > 1, this becomes complex. 
        # Let's constrain positions to be inside in the move step, or use a hard penalty.
        
        # Hard penalty for being outside
        outside_mask = radii > 1.0
        if np.any(outside_mask):
            # Large penalty proportional to violation squared
            violations = radii[outside_mask] - 1.0
            boundary_penalty = np.sum(violations**2) * 100.0
        else:
            # Soft penalty to push away from boundary
            # We want to maximize distance from boundary.
            # Minimize 1 / (1 - radii)^2
            # To avoid singularity at radii=1, we cap radii at 0.999 in calculation
            safe_radii = np.minimum(radii, 0.999)
            boundary_dist = 1.0 - safe_radii
            boundary_penalty = np.sum(1.0 / (boundary_dist**2 + 1e-6))
            
        return pair_energy + boundary_penalty

    # Initial Energy
    current_energy = calculate_energy(points)
    
    # SA Parameters
    T_start = 0.5
    T_end = 0.001
    alpha = 0.995 # Cooling factor
    max_iter = 5000 # Adjust based on time
    
    # Pre-allocate
    best_points = points.copy()
    best_energy = current_energy
    
    for it in range(max_iter):
        T = T_start * (alpha ** it)
        if T < T_end:
            break
            
        # Select a random point to move
        idx = np.random.randint(N)
        
        # Generate a random move
        # Step size scales with T
        step_size = 0.05 * np.sqrt(T)
        delta = np.random.normal(0, step_size, 2)
        
        new_points = points.copy()
        new_points[idx] += delta
        
        # Hard constraint: keep inside unit circle
        new_radius = np.linalg.norm(new_points[idx])
        if new_radius > 0.99:
            # Project back to boundary (slightly inside)
            new_points[idx] = new_points[idx] * (0.99 / new_radius)
            
        # Calculate new energy
        # Optimization: Only recalculate energy for the changed point?
        # For N=300, full recalc is O(N^2) = 90k ops. 5000 iters = 450M ops.
        # This might be slow in pure Python loops, but NumPy vectorization helps.
        # However, calling calculate_energy 5000 times with 300x300 matrix might be tight.
        # Let's try to optimize energy calc or reduce iters.
        
        new_energy = calculate_energy(new_points)
        
        # Metropolis Criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            points = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
                
    return best_points
