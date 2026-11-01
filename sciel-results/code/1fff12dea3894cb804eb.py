import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    # Area of disk = pi. Area per circle ~ pi * r^2.
    # r ~ 0.053. Area ~ 0.0088.
    # Total area 300 * 0.0088 = 2.64 < pi.
    # Spacing s ~ 2*r = 0.106.
    # Hex grid spacing in x: s, in y: s*sqrt(3)/2.
    
    def get_hex_lattice_points(n_points):
        # Try different spacings to find one that fits >= n_points
        best_points = None
        # Start with an estimate
        r_est = 0.05
        s = 2 * r_est
        
        # We want to generate points on a hex grid and filter those inside unit circle
        # Then pick the n_points closest to origin? Or just the first n_points?
        # Picking closest to origin usually gives a more compact cluster.
        
        for _ in range(10): # Try a few spacings
            k = int(2.0 / s) + 5
            rows = []
            for j in range(-k, k + 1):
                y = j * s * np.sqrt(3.0) / 2.0
                offset = 0.5 * s if j % 2 != 0 else 0.0
                for i in range(-k, k + 1):
                    x = i * s + offset
                    rows.append((x, y))
            points = np.array(rows)
            # Filter inside unit circle (with margin for radius)
            # We don't know exact radius yet, but points must be inside.
            # Let's just filter by distance < 1.0
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii < 1.0
            points = points[mask]
            
            if len(points) >= n_points:
                # Sort by distance to origin to get the most central cluster
                indices = np.argsort(radii[mask])
                return points[indices[:n_points]]
            s *= 0.95 # Tighten spacing
        
        # Fallback if loop didn't return (shouldn't happen)
        return np.random.rand(n_points, 2) * 2 - 1

    points = get_hex_lattice_points(N)
    
    # Ensure points are strictly inside
    radii = np.hypot(points[:, 0], points[:, 1])
    points[radii >= 1.0] = points[radii >= 1.0] * (0.99 / radii[radii >= 1.0, None])
    
    # 2. Optimization: Local Search / Repulsion
    # We want to maximize the minimum distance between points and to boundary.
    # Let's use a step-based approach.
    
    # Pre-calculate some constants
    # We will iterate and push points apart.
    
    # Helper to compute min pair distance and boundary distance
    def compute_metrics(pts):
        # Pairwise distances
        # N=300, N^2 = 90k. This is fast in numpy.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        # Find indices
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Boundary distances
        radii = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - radii
        min_bound_dist = np.min(bound_dists)
        min_bound_idx = np.argmin(bound_dists)
        
        return min_pair_dist, min_idx, min_bound_dist, min_bound_idx

    # Optimization Loop
    step_size = 0.01
    # We can adapt step size or use a fixed one.
    # Let's use a simple loop until time limit.
    
    # To avoid getting stuck, we can add a small random perturbation occasionally.
    
    for iteration in range(10000): # Cap iterations just in case
        if time.time() - start_time > time_limit:
            break
            
        min_pair_dist, (i, j), min_bound_dist, k = compute_metrics(points)
        
        # Current "radius" is min(min_pair_dist/2, min_bound_dist)
        # We want to increase the limiting factor.
        
        if min_pair_dist / 2.0 < min_bound_dist:
            # Internal collision is the bottleneck
            # Push i and j apart
            vec = points[i] - points[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
                # Push by a fraction of the violation? Or fixed step?
                # Fixed step is safer for stability.
                points[i] += vec * step_size
                points[j] -= vec * step_size
        else:
            # Boundary is the bottleneck
            # Push k towards origin
            vec = -points[k]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
                points[k] += vec * step_size
        
        # Optional: Small random jitter to escape local minima
        # Only do this if we've been stuck? 
        # For simplicity, let's stick to deterministic push first.
        # If we want to be more aggressive, we can reduce step size over time.
        step_size *= 0.999 # Slowly anneal step size
        
    # Final check and cleanup
    # Ensure no points are outside
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        points[mask] = points[mask] * (1.0 / radii[mask, None])
        
    return points
