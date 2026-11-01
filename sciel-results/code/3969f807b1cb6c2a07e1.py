"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialization: Hexagonal lattice with jitter
    # We generate a hexagonal lattice and pick the N points closest to the origin
    # that fit inside the circle.
    
    def initialize_hex():
        # Estimate spacing to ensure N points fit
        # r ~ 0.125, spacing ~ 0.25
        spacing = 0.25
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        # Filter points inside unit circle
        mask = radius < 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Sort by distance to center, take closest N
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx].copy()
            # Add small random jitter to break symmetry
            pts += np.random.normal(0, 0.01, pts.shape)
            # Ensure jittered points are still inside
            norms = np.linalg.norm(pts, axis=1)
            for i in range(N):
                if norms[i] > 0.99:
                    pts[i] = pts[i] / norms[i] * 0.99
            return pts
        else:
            # Fallback: random points
            return np.random.rand(N, 2) * 1.8 - 0.9

    points = initialize_hex()
    
    # 2. Simulated Annealing
    
    def get_radius(pts):
        # Distance to boundary
        d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        r_boundary = np.min(d_boundary)
        
        # Pairwise distances
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        return min(r_boundary, r_pair)

    best_points = points.copy()
    best_r = get_radius(best_points)
    current_r = best_r
    
    # SA Parameters
    T_start = 0.01
    T_end = 1e-5
    n_steps = 100000
    # Move size decay
    move_scale_start = 0.1
    move_scale_end = 0.001
    
    for step in range(n_steps):
        # Cooling schedule
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        move_scale = move_scale_start * (move_scale_end / move_scale_start) ** progress
        
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate a random move
        # Move in a random direction with magnitude scaled by move_scale
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(0, move_scale)
        move = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        
        new_points = points.copy()
        new_points[i] += move
        
        # Check if new point is inside the unit circle
        norm_i = np.linalg.norm(new_points[i])
        if norm_i > 1.0:
            # Project back to circle boundary minus epsilon
            new_points[i] = new_points[i] / norm_i * 0.999
            
        new_r = get_radius(new_points)
        
        # Acceptance criterion
        if new_r > current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Calculate probability of accepting worse solution
            delta_r = new_r - current_r
            if T > 1e-9:
                prob = np.exp(delta_r / T)
                if np.random.rand() < prob:
                    points = new_points
                    current_r = new_r
    
    return best_points
# EVOLVE-BLOCK-END
