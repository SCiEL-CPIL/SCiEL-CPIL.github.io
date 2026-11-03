"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        n = len(centers)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(n_rows, n_cols, scale=1.0):
        """Create hexagonal packing layout."""
        d = scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def force_directed_relaxation(centers, max_iter=5000, dt=0.0001, cooling=0.999):
        """
        Force-directed relaxation using Lennard-Jones-style repulsion
        and wall confinement. Uses velocity Verlet integration.
        """
        centers = centers.copy()
        velocities = np.zeros_like(centers)
        
        # Lennard-Jones parameters
        sigma = 2 * compute_radius(centers)  # approximate diameter
        if sigma < 0.01:
            sigma = 0.02
        epsilon = 1.0
        wall_strength = 10.0  # strength of wall repulsion
        
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute forces
            forces = np.zeros_like(centers)
            n = len(centers)
            
            # Pairwise repulsion (Lennard-Jones)
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # LJ potential: V = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
            # Force: F = 24*epsilon*(2*(sigma/r)^12 - (sigma/r)^6)/r * r_hat
            # We use a simpler repulsive force to avoid attraction at long range
            # F = k / r^2 in direction away from neighbor
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    d = dist_matrix[i, j]
                    if d < 1e-10:
                        continue
                    # Repulsive force: k / r^3 (gradient of 1/r^2)
                    force_mag = 1.0 / (d * d + 1e-12)
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    forces[i, 0] += force_mag * dx
                    forces[i, 1] += force_mag * dy
            
            # Wall repulsion
            for i in range(n):
                x, y = centers[i]
                # Left wall
                if x < 0:
                    forces[i, 0] += wall_strength * (-x)
                # Right wall
                if x > 0:
                    forces[i, 0] -= wall_strength * x
                # Bottom wall
                if y < 0:
                    forces[i, 1] += wall_strength * (-y)
                # Top wall
                if y > 0:
                    forces[i, 1] -= wall_strength * y
            
            # Velocity Verlet integration
            velocities = velocities * cooling + forces * dt
            centers = centers + velocities * dt
            
            # Project back into square
            centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Compute radius
            current_r = compute_radius(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
            
            # Cool down
            dt *= cooling
        
        return best_centers, best_r

    def multi_restart_force_directed():
        """Run force-directed relaxation from multiple initial configurations."""
        best_centers = None
        best_radius = 0.0
        
        # Generate diverse initial configurations
        initializations = []
        
        # Hexagonal layouts with different row/col counts
        for n_rows in range(12, 18):
            n_cols = math.ceil(N / n_rows)
            for scale in [0.90, 0.95, 1.00, 1.05, 1.10]:
                pts = hexagonal_init(n_rows, n_cols, scale)
                if pts is not None:
                    initializations.append(pts)
        
        # Square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) >= N:
            initializations.append(grid[:N])
        
        # Random initialization (Jittered grid)
        np.random.seed(42)
        for _ in range(3):
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
            jitter = np.random.randn(N, 2) * 0.005
            grid = grid + jitter
            grid[:, 0] = np.clip(grid[:, 0], -0.5 + 0.01, 0.5 - 0.01)
            grid[:, 1] = np.clip(grid[:, 1], -0.5 + 0.01, 0.5 - 0.01)
            initializations.append(grid)
        
        # Run force-directed relaxation on each initialization
        for init_idx, init_centers in enumerate(initializations):
            if time.time() > deadline - 15:
                break
            
            # Compute initial radius
            init_r = compute_radius(init_centers)
            
            # Run force-directed relaxation
            refined_centers, refined_r = force_directed_relaxation(
                init_centers, 
                max_iter=3000, 
                dt=0.00005, 
                cooling=0.9995
            )
            
            if refined_r > best_radius:
                best_radius = refined_r
                best_centers = refined_centers.copy()
        
        return best_centers, best_radius

    # Run multi-restart force-directed relaxation
    best_centers, best_radius = multi_restart_force_directed()
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
