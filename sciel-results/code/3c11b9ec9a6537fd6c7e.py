"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200


def construct():
    """
    Uses a physics-based approach with symplectic Euler integration 
    to pack N circles in a unit square [-0.5, 0.5]^2.
    """
    
    def generate_init(n_circles):
        """Generate a jittered square grid initialization."""
        m = int(np.ceil(np.sqrt(n_circles)))
        # Create grid
        # We want points in [-0.5, 0.5]
        # Spacing 1/m
        # Center of cell: (i + 0.5)/m - 0.5
        xs = (np.arange(m) + 0.5) / m - 0.5
        ys = (np.arange(m) + 0.5) / m - 0.5
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Shuffle and take first N
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        idx = np.random.permutation(len(pts))
        pts = pts[idx[:n_circles]]
        
        return pts

    def simulate(centers, r_target, steps=1500, k=50.0, damping=0.98, dt=0.01):
        """
        Run symplectic Euler simulation.
        """
        c = centers.copy()
        v = np.zeros_like(c)
        m = 1.0 # mass
        
        for step in range(steps):
            # Calculate forces
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            mask = dists < r_target
            safe_dists = np.where(dists < 1e-8, 1e-8, dists)
            
            # Force magnitude
            # F = k * (r_target - d)
            f_mag = k * (r_target - dists)
            f_mag[~mask] = 0.0
            
            # Force vector
            # F_ij = f_mag * (c_i - c_j) / d
            # diff[i,j] = c_i - c_j
            f_vec = f_mag[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
            
            forces = np.sum(f_vec, axis=1)
            
            # Wall forces
            # Right
            mask_rx = c[:, 0] > 0.5 - r_target
            if np.any(mask_rx):
                forces[mask_rx, 0] -= k * (c[mask_rx, 0] - (0.5 - r_target))
            
            # Left
            mask_lx = c[:, 0] < -0.5 + r_target
            if np.any(mask_lx):
                forces[mask_lx, 0] += k * ((-0.5 + r_target) - c[mask_lx, 0])
            
            # Top
            mask_ty = c[:, 1] > 0.5 - r_target
            if np.any(mask_ty):
                forces[mask_ty, 1] -= k * (c[mask_ty, 1] - (0.5 - r_target))
            
            # Bottom
            mask_by = c[:, 1] < -0.5 + r_target
            if np.any(mask_by):
                forces[mask_by, 1] += k * ((-0.5 + r_target) - c[mask_by, 1])
            
            # Update velocity
            v = v + (forces / m) * dt
            v = v * damping
            
            # Update position
            c = c + v * dt
            
            # Check convergence
            # If max force is small, break
            # Note: forces is the force on each particle
            if np.max(np.abs(forces)) < 1e-4:
                break
        
        # Check success
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        bound_dists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_bound = np.min(bound_dists)
        
        # Allow small tolerance
        success = (min_pair >= r_target - 1e-5) and (min_bound >= r_target - 1e-5)
        
        return c, success

    # Initial configuration
    centers_init = generate_init(N)
    
    # Binary search for the maximum radius
    r_low = 0.030
    r_high = 0.040
    
    best_centers = centers_init
    best_r = 0.0
    
    # Run binary search
    # 25 iterations for good precision
    for _ in range(25):
        r_mid = (r_low + r_high) / 2.0
        
        # Use best_centers as init
        init_c = best_centers
        
        c_new, success = simulate(init_c, r_mid)
        
        if success:
            r_low = r_mid
            best_centers = c_new
            best_r = r_mid
        else:
            r_high = r_mid
            
    return best_centers


# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    
    # Compute radius for printing
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    bound_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    print(f"Radius: {min(min_pair, min_bound):.10f}")
