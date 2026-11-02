"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200

def _compute_radius(centers):
    """Compute the maximum possible radius for the given centers."""
    if len(centers) < 2:
        return 0.5 - np.max(np.abs(centers))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    
    bound_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    
    return min(min_pair, min_bound)

def construct():
    # Initial configuration: hexagonal lattice
    def hexagonal_init(n_circles):
        best_points = None
        best_r = 0
        
        for rows in range(10, 25):
            for cols in range(10, 25):
                if rows * cols < n_circles:
                    continue
                
                r_max = min(1.0 / (2*cols - 1), 1.0 / ((rows - 1) * np.sqrt(3)))
                
                if r_max > best_r:
                    best_r = r_max
                    pts = []
                    for j in range(rows):
                        for i in range(cols):
                            x = (i * 2 - (cols - 1)) * r_max
                            if j % 2 == 1:
                                x += r_max
                            y = (j - (rows - 1) / 2.0) * r_max * np.sqrt(3)
                            pts.append([x, y])
                    pts = np.array(pts[:n_circles])
                    if np.all(np.abs(pts) <= 0.5 - best_r * 0.01):
                        best_points = pts
                        best_r = r_max
        
        if best_points is not None:
            return best_points
        
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]

    centers = hexagonal_init(N)
    velocities = np.zeros_like(centers)
    
    # Physics parameters
    mass = 1.0
    k_stiffness = 1000.0
    damping = 0.99
    dt_initial = 0.001
    
    # Target radius strategy
    r_target = 0.030
    r_step = 0.0001
    max_r = 0.040
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Simulation loop
    expansion_steps = 40
    max_steps = 3000
    
    for _ in range(expansion_steps):
        if r_target >= max_r:
            break
            
        velocities[:] = 0.0
        
        dt = dt_initial
        
        for step in range(max_steps):
            forces = np.zeros_like(centers)
            
            # Pairwise forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            overlap = 2*r_target - dists
            mask = overlap > 0
            np.fill_diagonal(mask, False)
            
            force_mag = np.zeros_like(dists)
            force_mag[mask] = k_stiffness * (overlap[mask] ** 3)
            
            dir_vec = diff / dists[:, :, np.newaxis]
            
            forces += np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
            
            # Boundary forces
            dist_left = centers[:, 0] - (-0.5 + r_target)
            mask_left = dist_left < 0
            if np.any(mask_left):
                overlap_left = -dist_left[mask_left]
                f_left = k_stiffness * (overlap_left ** 3)
                forces[mask_left, 0] += f_left
            
            dist_right = (0.5 - r_target) - centers[:, 0]
            mask_right = dist_right < 0
            if np.any(mask_right):
                overlap_right = -dist_right[mask_right]
                f_right = k_stiffness * (overlap_right ** 3)
                forces[mask_right, 0] -= f_right
            
            dist_bottom = centers[:, 1] - (-0.5 + r_target)
            mask_bottom = dist_bottom < 0
            if np.any(mask_bottom):
                overlap_bottom = -dist_bottom[mask_bottom]
                f_bottom = k_stiffness * (overlap_bottom ** 3)
                forces[mask_bottom, 1] += f_bottom
            
            dist_top = (0.5 - r_target) - centers[:, 1]
            mask_top = dist_top < 0
            if np.any(mask_top):
                overlap_top = -dist_top[mask_top]
                f_top = k_stiffness * (overlap_top ** 3)
                forces[mask_top, 1] -= f_top
            
            # Symplectic Euler
            velocities += (forces / mass) * dt
            velocities *= damping
            centers += velocities * dt
            
            # Adaptive time step
            max_v = np.max(np.linalg.norm(velocities, axis=1))
            if max_v > 0.1:
                dt *= 0.5
            elif max_v < 0.001 and step > 100:
                break
        
        current_r = _compute_radius(centers)
        
        if current_r > best_radius:
            best_radius = current_r
            best_centers = centers.copy()
            
        r_target += r_step
        
    return best_centers

# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    print(f"Radius: {_compute_radius(centers):.10f}")
