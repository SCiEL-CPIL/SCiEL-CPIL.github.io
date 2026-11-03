"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline for 120s limit safety

    # Initialize with random points on unit sphere
    def random_sphere(n):
        v = np.random.randn(n, 3)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / norms

    pts = random_sphere(N)

    # Build a fine spherical grid for covering radius estimation
    # Use icosahedron subdivision or simple lat-lon grid
    # For speed, use a moderate grid: ~100x200 = 20000 points
    n_lat = 60
    n_lon = 120
    lats = np.linspace(np.pi / (2 * n_lat), np.pi - np.pi / (2 * n_lat), n_lat)
    lons = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    # Convert to Cartesian
    sin_lat = np.sin(lat_grid)
    cos_lat = np.cos(lat_grid)
    sin_lon = np.sin(lon_grid)
    cos_lon = np.cos(lon_grid)
    grid_x = sin_lat * cos_lon
    grid_y = sin_lat * sin_lon
    grid_z = cos_lat
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    def covering_radius_deg(pts, grid_pts):
        # For each grid point, find nearest pts point
        # Use scipy if available, else chunked computation
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            dists, _ = tree.query(grid_pts, k=1)
            # dists are Euclidean; convert to angular
            # angle = arccos(dot) but we have Euclidean dist
            # For unit vectors: angle = 2*arcsin(d/2)
            angles = 2.0 * np.arcsin(np.minimum(1.0, dists / 2.0))
            return np.degrees(np.max(angles))
        except ImportError:
            # Fallback: chunked
            max_angle = 0.0
            chunk = 5000
            for i in range(0, len(grid_pts), chunk):
                g = grid_pts[i:i+chunk]
                # dot products
                dots = g @ pts.T  # (chunk, N)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.degrees(np.arccos(dots))
                max_angle = max(max_angle, np.max(angles))
            return max_angle

    def find_worst_index(pts, grid_pts):
        """Find which point has the largest 'influence' (worst coverage area)."""
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            dists, idxs = tree.query(grid_pts, k=1)
            angles = 2.0 * np.arcsin(np.minimum(1.0, dists / 2.0))
            # For each point, find max angle among grid points assigned to it
            worst_per_point = np.zeros(N)
            for j in range(N):
                mask = idxs == j
                if np.any(mask):
                    worst_per_point[j] = np.max(angles[mask])
            return np.argmax(worst_per_point)
        except ImportError:
            dots = grid_pts @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            nearest = np.argmin(angles, axis=1)
            worst_per_point = np.zeros(N)
            for j in range(N):
                mask = nearest == j
                if np.any(mask):
                    worst_per_point[j] = np.max(angles[mask])
            return np.argmax(worst_per_point)

    best_pts = pts.copy()
    best_cr = covering_radius_deg(pts, grid_pts)

    # Stochastic local search
    iters_per_sec_target = 50  # target ~50 iterations per second
    while time.time() < deadline:
        # Find worst point
        worst_idx = find_worst_index(pts, grid_pts)
        
        # Perturb: move worst point in random tangential direction
        p = pts[worst_idx]
        # Random direction
        rand_dir = np.random.randn(3)
        # Make tangential
        rand_dir = rand_dir - np.dot(rand_dir, p) * p
        rand_norm = np.linalg.norm(rand_dir)
        if rand_norm < 1e-10:
            continue
        rand_dir /= rand_norm
        
        # Try different step sizes
        improved = False
        for step_size in [0.05, 0.02, 0.01, 0.005, 0.002]:
            new_p = p + step_size * rand_dir
            new_p = new_p / np.linalg.norm(new_p)
            pts[worst_idx] = new_p
            
            cr = covering_radius_deg(pts, grid_pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
                improved = True
                break
            else:
                # Revert
                pts[worst_idx] = p
        
        if not improved:
            # Try random perturbation of a random point
            idx = np.random.randint(N)
            p = pts[idx]
            rand_dir = np.random.randn(3)
            rand_dir = rand_dir - np.dot(rand_dir, p) * p
            rand_norm = np.linalg.norm(rand_dir)
            if rand_norm < 1e-10:
                continue
            rand_dir /= rand_norm
            for step_size in [0.03, 0.01, 0.005]:
                new_p = p + step_size * rand_dir
                new_p = new_p / np.linalg.norm(new_p)
                pts[idx] = new_p
                cr = covering_radius_deg(pts, grid_pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = pts.copy()
                    break
                else:
                    pts[idx] = p

    return best_pts
# EVOLVE-BLOCK-END
