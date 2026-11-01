"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    def compute_radius(pts):
        """Compute the maximum radius for a given set of points."""
        d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        r_boundary = np.min(d_boundary)
        
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        return min(r_boundary, r_pair)

    def initialize_points():
        """Initialize points on a jittered grid inside the unit circle."""
        grid_size = 12
        x = np.linspace(-0.95, 0.95, grid_size)
        y = np.linspace(-0.95, 0.95, grid_size)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        radii = np.linalg.norm(points, axis=1)
        points = points[radii < 0.95]
        
        jitter = 0.05
        points += np.random.uniform(-jitter, jitter, size=points.shape)
        
        radii = np.linalg.norm(points, axis=1)
        points = points[radii < 0.95]
        
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            points = points[indices]
        elif len(points) < N:
            while len(points) < N:
                p = np.random.uniform(-0.9, 0.9, 2)
                if np.linalg.norm(p) < 0.95:
                    points = np.vstack([points, p])
        
        return points

    T_start = 0.05
    T_end = 1e-5
    max_iters = 10000
    
    current_points = initialize_points()
    current_r = compute_radius(current_points)
    
    best_points = current_points.copy()
    best_r = current_r
    
    for i in range(max_iters):
        T = T_start * (T_end / T_start) ** (i / max_iters)
        
        idx = np.random.randint(N)
        
        current_step = 0.01 * (1 + T * 10) 
        delta = np.random.uniform(-current_step, current_step, 2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        norm_new = np.linalg.norm(new_points[idx])
        if norm_new > 1.0:
            new_points[idx] = new_points[idx] / norm_new * 0.999
        
        new_r = compute_radius(new_points)
        
        if new_r > current_r:
            current_points = new_points
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_points = current_points.copy()
        else:
            delta_r = current_r - new_r
            if delta_r < 1e-9:
                prob = 1.0
            else:
                prob = np.exp(-delta_r / T)
            
            if np.random.random() < prob:
                current_points = new_points
                current_r = new_r
        
    return best_points
# EVOLVE-BLOCK-END
