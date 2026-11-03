import numpy as np
import time

N = 300

def _hex_lattice_init(n):
    """Generate a hexagonal lattice initialization for n circles."""
    # Estimate spacing. Area of unit circle is pi. Area per circle is pi/n.
    # Hex packing density is pi/sqrt(12) ~ 0.9069.
    # Area per point ~ sqrt(3)/2 * d^2.
    # So d^2 ~ 2 * (pi/n) / (pi/sqrt(12)) * (1/0.9069) ... 
    # Simpler: d ~ sqrt(4 / (n * sqrt(3))) roughly.
    # Let's just try a range of spacings and pick the one that gives enough points.
    
    best_points = None
    best_count = 0
    
    # Try different spacings
    for d in np.linspace(0.05, 0.1, 10):
        k = int(1.0 / d) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points that can fit a circle of radius d/2 inside unit circle
        # Actually, we just need centers inside 1 - r. But r is unknown.
        # Let's just filter inside 0.95 and see how many we get.
        mask = radius <= 0.95
        points = points[mask]
        
        if len(points) >= n:
            # Sort by distance to origin to pick the most central ones
            indices = np.argsort(radius[mask], kind="stable")
            points = points[indices[:n]]
            return points
    # Fallback: simple grid
    return np.random.rand(n, 2) * 2 - 1

def _compute_radius(points):
    """Compute the max radius for the given centers."""
    # Pairwise distances
    # To save memory/time for N=300, we can do it in chunks or just full matrix.
    # 300x300 matrix is 90,000 floats. Very small.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Get upper triangle indices
    i, j = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[i, j])
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialize
    points = _hex_lattice_init(N)
    current_r = _compute_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    # 2. Simulated Annealing
    T = 1e-4
    T_min = 1e-8
    cooling = 0.995
    step_size = 0.05
    step_cooling = 0.999
    
    n_steps = 0
    while T > T_min and time.time() < deadline:
        # Move a random point
        i = np.random.randint(N)
        delta = np.random.normal(0, step_size, size=2)
        new_points = points.copy()
        new_points[i] += delta
        
        new_r = _compute_radius(new_points)
        
        # Acceptance criterion
        if new_r >= current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Boltzmann factor
            # We want to maximize r, so delta_r = new_r - current_r is negative.
            # prob = exp(delta_r / T)
            if np.random.rand() < np.exp((new_r - current_r) / T):
                points = new_points
                current_r = new_r
        
        T *= cooling
        step_size *= step_cooling
        n_steps += 1
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 300

def _hex_lattice_init(n):
    """Generate a hexagonal lattice initialization for n circles."""
    best_points = None
    
    # Try different spacings to find one that yields at least n points inside radius ~0.95
    # We want to pick the tightest valid packing initial guess.
    # Actually, starting slightly looser allows the optimizer to find the tightest.
    # But a good starting point is key.
    
    for d in np.linspace(0.06, 0.04, 20):
        k = int(1.0 / d) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        # Filter points that are safely inside. 
        # If we use d as spacing, min dist is d. So r=d/2.
        # We need 1 - |c| >= d/2 => |c| <= 1 - d/2.
        mask = radius <= 1.0 - d / 2.0
        points = points[mask]
        
        if len(points) >= n:
            # Sort by distance to origin to pick the most central ones
            indices = np.argsort(radius[mask], kind="stable")
            points = points[indices[:n]]
            return points
            
    # Fallback: Random
    return np.random.rand(n, 2) * 2 - 1

def _compute_radius(points):
    """Compute the max radius for the given centers."""
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[i, j])
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialize
    points = _hex_lattice_init(N)
    current_r = _compute_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    # 2. Simulated Annealing
    T = 1e-3
    T_min = 1e-9
    cooling = 0.998
    step_size = 0.1
    step_cooling = 0.999
    
    n_steps = 0
    while T > T_min and time.time() < deadline:
        # Move a random point
        i = np.random.randint(N)
        delta = np.random.normal(0, step_size, size=2)
        new_points = points.copy()
        new_points[i] += delta
        
        new_r = _compute_radius(new_points)
        
        # Acceptance criterion
        if new_r >= current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Boltzmann factor
            if np.random.rand() < np.exp((new_r - current_r) / T):
                points = new_points
                current_r = new_r
        
        T *= cooling
        step_size *= step_cooling
        n_steps += 1
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return best_points
# EVOLVE-BLOCK-END
