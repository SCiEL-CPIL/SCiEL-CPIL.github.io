import numpy as np
import scipy.optimize as opt
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Normal vectors for the 6 directions of the outer hexagon
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.deg2rad(np.arange(0, 360, 60))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Pre-compute pair indices for overlap check
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    def calculate_L_and_penalty(params):
        # params: array of size 2N, [x1, y1, x2, y2, ...]
        centers = params.reshape(N, 2)
        
        # 1. Calculate required outer side length L
        # Projection of centers onto normals
        # projections shape: (6, N)
        projections = centers @ normals.T
        # Max projection in each direction
        max_projs = np.max(projections, axis=1)
        
        # L_k = (2/sqrt(3)) * max_proj_k + 1
        # We take the max over all k
        L_required = np.max((2.0 / np.sqrt(3.0)) * max_projs + 1.0)
        
        # 2. Calculate Overlap Penalty
        # Distance between all pairs
        diffs = centers[i_indices] - centers[j_indices]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Minimum distance for non-overlap is sqrt(3)
        min_dist = np.sqrt(3.0)
        overlaps = np.maximum(0.0, min_dist - dists)
        penalty = np.sum(overlaps**2)
        
        # We want to minimize L, but ensure no overlap.
        score = L_required + 1000.0 * penalty
        
        return score, L_required, penalty, centers

    best_L = float('inf')
    best_centers = None
    
    # Initial configurations
    starts = []
    
    # Start 1: Compact hexagonal cluster (12 closest points to origin in triangular lattice)
    # Triangular lattice spacing for non-overlapping unit hexagons is sqrt(3)
    s = np.sqrt(3.0)
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + j * 0.5 * s
            y = j * (np.sqrt(3.0) / 2.0) * s
            lattice_points.append((x, y))
    lattice_points = np.array(lattice_points)
    dists_to_origin = np.linalg.norm(lattice_points, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    init_cluster = lattice_points[sorted_indices[:12]]
    starts.append(init_cluster.flatten())
    
    # Start 2: Random
    np.random.seed(42)
    for _ in range(10):
        r = np.random.uniform(0, 5, N)
        theta = np.random.uniform(0, 2*np.pi, N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        starts.append(np.column_stack((x, y)).flatten())
        
    # Start 3: Grid 4x3
    x_grid = np.linspace(-3, 3, 4)
    y_grid = np.linspace(-2, 2, 3)
    xx, yy = np.meshgrid(x_grid, y_grid)
    init_grid = np.column_stack((xx.flatten(), yy.flatten()))
    starts.append(init_grid.flatten())
    
    for start_params in starts:
        if time.time() > deadline:
            break
            
        try:
            result = opt.minimize(
                lambda x: calculate_L_and_penalty(x)[0],
                start_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            score, L_req, penalty, centers = calculate_L_and_penalty(result.x)
            
            if penalty < 1e-6:
                if L_req < best_L:
                    best_L = L_req
                    best_centers = centers
                    
        except Exception:
            continue
            
    if best_centers is None:
        best_centers = starts[0].reshape(N, 2)
        _, best_L, _, _ = calculate_L_and_penalty(best_centers)
        
    # Center the configuration
    centroid = np.mean(best_centers, axis=0)
    best_centers = best_centers - centroid
    
    # Recalculate L for the centered configuration
    _, L_final, _, centers_centered = calculate_L_and_penalty(best_centers.flatten())
    
    inner = []
    for i in range(N):
        x, y = centers_centered[i]
        inner.append([float(x), float(y), 0.0])
        
    return inner, (0.0, 0.0), float(L_final), 0.0

# EVOLVE-BLOCK-END
