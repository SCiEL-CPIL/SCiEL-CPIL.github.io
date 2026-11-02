"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200


def construct():
    # Hexagonal initialization
    def hexagonal_init(n_circles, r):
        """Generate hexagonal lattice points centered in unit square, scaled by r."""
        # We want to fit a hex grid of spacing 2r into the square.
        # Approximate number of rows/cols
        # Width ~ (cols-1)*2r + r, Height ~ (rows-1)*r*sqrt(3)
        # Let's just generate a large hex grid and crop/center it.
        
        # Estimate grid size
        # r is the radius, so diameter is 2r.
        # Hex spacing horizontal: 2r, vertical: r*sqrt(3)
        
        # Try to fit as many as possible
        # Let's use a fixed number of rows/cols based on aspect ratio
        # For a square, roughly equal rows and cols in terms of extent.
        
        # Let's just generate points on a hex grid with spacing 2r (x) and r*sqrt(3) (y)
        # and center them.
        
        # Determine range
        # We need N points.
        # Let's assume a square-ish block of hex points.
        # cols * rows >= N
        # Let cols = ceil(sqrt(N)), rows = ceil(N/cols)
        
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
        
        pts = []
        for j in range(rows):
            for i in range(cols):
                x = (i * 2 - (cols - 1)) * r
                if j % 2 == 1:
                    x += r
                y = (j - (rows - 1) / 2.0) * r * np.sqrt(3)
                pts.append([x, y])
        
        pts = np.array(pts[:n_circles])
        
        # Center the points
        pts -= pts.mean(axis=0)
        
        # Scale/translate to fit? 
        # The hex init might be too big or too small for the given r.
        # We will rely on the dynamics to push them into the square.
        # But if they start outside, forces will push them in.
        
        return pts

    def simulate(centers, r, max_steps=2000, dt_init=0.01):
        """
        Run symplectic Euler simulation with soft-contact potential.
        Returns final centers and success flag.
        """
        centers = centers.copy()
        velocities = np.zeros_like(centers)
        
        # Parameters
        k_repel = 500.0  # Stiffness for circle-circle
        k_bound = 500.0  # Stiffness for boundary
        damping = 0.98
        dt = dt_init
        
        # Potential: U = sum (2r - d)^4 for d < 2r
        # Force: F = -dU/dc = 4*k*(2r-d)^3 * (d/|d|) ... wait, gradient of (2r-d)^4 is -4(2r-d)^3 * grad(d)
        # grad(d_ij) = (c_i - c_j) / d_ij
        # So F_ij on i is k * 4 * (2r - d_ij)^3 * (c_i - c_j) / d_ij
        
        for step in range(max_steps):
            # Adaptive time step: ensure dt * max_force is small
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Circle-Circle
            # Use broadcasting for efficiency, but N=200 is small enough for O(N^2) loops if optimized or vectorized carefully.
            # Vectorized O(N^2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, 1.0) # Avoid div by 0
            
            # Only consider pairs where dists < 2r
            # Mask
            mask = dists < 2 * r
            # Set diagonal to False
            np.fill_diagonal(mask, False)
            
            # Compute repulsion magnitude
            # overlap = 2r - dists
            # Force magnitude factor = k * 4 * overlap^3 / dists
            # We only compute for masked entries to save time? 
            # N=200, 40000 pairs. It's fast.
            
            overlaps = 2 * r - dists
            overlaps[~mask] = 0.0
            
            # Force vector contribution
            # F_ij = k * 4 * overlap^3 * (c_i - c_j) / dists
            # Note: diff[i,j] = c_i - c_j
            # We need to sum over j for each i.
            
            # Term: overlap^3 / dists
            factor = np.zeros_like(dists)
            factor[mask] = (overlaps[mask] ** 3) / dists[mask]
            
            # F_i = k * 4 * sum_j ( factor[i,j] * diff[i,j] )
            forces += k_repel * 4.0 * np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary Forces
            # Constraint: |c_x| <= 0.5 - r
            # If c_x > 0.5 - r, push left. Force = -k_b * (c_x - (0.5-r))^3
            # If c_x < -0.5 + r, push right. Force = k_b * ((-0.5+r) - c_x)^3
            
            # X-axis
            limit_x = 0.5 - r
            # Right wall
            over_x_right = np.maximum(0, centers[:, 0] - limit_x)
            forces[:, 0] -= k_bound * 4.0 * (over_x_right ** 3)
            # Left wall
            over_x_left = np.maximum(0, -limit_x - centers[:, 0])
            forces[:, 0] += k_bound * 4.0 * (over_x_left ** 3)
            
            # Y-axis
            limit_y = 0.5 - r
            # Top wall
            over_y_top = np.maximum(0, centers[:, 1] - limit_y)
            forces[:, 1] -= k_bound * 4.0 * (over_y_top ** 3)
            # Bottom wall
            over_y_bot = np.maximum(0, -limit_y - centers[:, 1])
            forces[:, 1] += k_bound * 4.0 * (over_y_bot ** 3)
            
            # Update velocity
            velocities += dt * forces
            velocities *= damping
            
            # Update position
            centers += dt * velocities
            
            # Check for stability / max force
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force > 1e6:
                dt *= 0.5
                # Revert step if it exploded? Simple approach: just reduce dt for next step
            elif max_force < 1e-6:
                # Converged
                break
            else:
                # Adaptive dt: keep dt * max_force ~ 0.1
                target_dt = 0.1 / max_force
                if target_dt > dt * 1.5:
                    dt = min(dt * 1.1, target_dt)
                elif target_dt < dt * 0.5:
                    dt = max(dt * 0.9, target_dt)
        
        # Final check: are all constraints satisfied?
        # Min distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        min_dist = np.min(dists)
        
        # Boundary
        min_bound = np.min(np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        ))
        
        # Success if min_dist >= 2r - 1e-5 and min_bound >= r - 1e-5
        # Note: min_bound is distance to edge. We need distance to edge >= r.
        # So min_bound >= r.
        
        success = (min_dist >= 2 * r - 1e-5) and (min_bound >= r - 1e-5)
        
        return centers, success

    # Binary search for max radius
    # Lower bound: square grid
    r_low = 0.5 / (2 * np.ceil(np.sqrt(N))) # ~0.025
    # Upper bound: slightly above known best
    r_high = 0.0375
    
    best_centers = None
    best_r = r_low
    
    # Initial guess for centers
    centers_init = hexagonal_init(N, 0.03)
    
    # Binary search
    for _ in range(20):
        r_mid = (r_low + r_high) / 2
        
        # Generate init based on r_mid? 
        # Better to use a generic init and let dynamics do the work.
        # Or scale the init to be roughly the right size.
        # Let's use the hex init scaled to fit r_mid roughly.
        # The hex init function takes r.
        c_init = hexagonal_init(N, r_mid)
        
        c_final, success = simulate(c_init, r_mid, max_steps=3000)
        
        if success:
            best_r = r_mid
            best_centers = c_final
            r_low = r_mid
        else:
            r_high = r_mid
        
        # If we found a good one, we can try to refine
        # But binary search handles it.
        
    # If best_centers is None, fallback to a simple grid
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_r = 0.5 / m

    # Final verification and slight polish
    # Compute actual radius
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    min_pair = np.min(dists) / 2
    
    bound_dists = np.minimum(
        np.minimum(0.5 - best_centers[:, 0], 0.5 + best_centers[:, 0]),
        np.minimum(0.5 - best_centers[:, 1], 0.5 + best_centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    
    r_final = min(min_pair, min_bound)
    
    # Clip to be safe
    best_centers = np.clip(best_centers, -0.5 + r_final, 0.5 - r_final)
    
    return best_centers


# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    # Recompute radius for print
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    min_pair = np.min(dists) / 2
    bound_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    print(f"Radius: {min(min_pair, min_bound):.10f}")
