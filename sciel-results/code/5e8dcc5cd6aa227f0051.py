"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200


def construct():
    """
    Uses a physics-inspired approach:
    1. Start with a good initial configuration (hexagonal-like packing).
    2. Simulate a system of repelling particles in a box.
    3. The particles push against each other and the walls, settling into
       a configuration that maximizes the minimum distance (locally).
    """

    def hexagonal_init(n_circles):
        """Generate hexagonal lattice points centered in unit square."""
        best_points = None
        best_r = 0

        # Try different numbers of rows
        for rows in range(10, 25):
            for cols in range(10, 25):
                total = rows * cols
                if total < n_circles:
                    continue

                # Hexagonal grid dimensions
                r_max = min(1.0 / (2 * cols - 1), 1.0 / ((rows - 1) * np.sqrt(3)))

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
                    if np.all(np.abs(pts) <= 0.5 - r_max * 0.01):
                        best_points = pts
                        best_r = r_max

        if best_points is not None:
            return best_points

        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]

    centers = hexagonal_init(N)
    velocities = np.zeros_like(centers)

    # Simulation parameters
    # We want to maximize the minimum distance.
    # We use a repulsive force that is strong at short distances.
    # F ~ 1/d^2 is a common choice.
    # We also add repulsion from the walls to ensure the "radius"
    # (distance to wall) is comparable to the pairwise distances.
    
    k_pair = 1.0       # Pairwise repulsion strength
    k_wall = 1.0       # Wall repulsion strength
    cutoff = 0.2       # Cutoff distance for forces
    dt = 0.001         # Time step
    damping = 0.95     # Velocity damping
    steps = 2000       # Number of simulation steps

    # Adaptive time stepping: if the system becomes unstable (NaNs or huge jumps), reduce dt.
    # For simplicity, we use a fixed small dt and damping which is usually stable enough.
    
    for step in range(steps):
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        # To avoid O(N^2) in Python, we use vectorized NumPy operations.
        # For N=200, N^2 = 40,000, which is manageable.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Mask diagonal and distances > cutoff
        mask = (dists < cutoff) & (dists > 1e-6)
        np.fill_diagonal(mask, False)
        
        # Force magnitude: k / d^2
        # We compute the vector force directly: F_ij = k * diff_ij / d^3
        # Because diff / d^2 * (1/d) = diff / d^3? No.
        # Unit vector u = diff / d.
        # F = k * u / d^2 = k * diff / d^3.
        
        # To avoid division by zero, we only compute for masked elements.
        # However, creating a full N x N x 2 array of forces is memory intensive but N=200 is fine (200*200*2*8 bytes = 640KB).
        
        # We compute the force contribution for all pairs.
        # inv_d3 = 1 / d^3 where d < cutoff, else 0.
        inv_d3 = np.zeros_like(dists)
        inv_d3[mask] = 1.0 / (dists[mask] ** 3)
        
        # F_ij = k_pair * diff_ij * inv_d3_ij
        # Note: diff[i,j] = c_i - c_j. Force on i is in direction of c_i - c_j (repulsion).
        # So F_i = sum_j k * (c_i - c_j) / d^3.
        
        forces += k_pair * np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall forces
        # For each particle, check distance to 4 walls.
        # Left wall: x = -0.5. d = x - (-0.5) = x + 0.5.
        # If d < cutoff, force is to the right (+x). F = k_wall / d^2.
        
        d_left = centers[:, 0] + 0.5
        d_right = 0.5 - centers[:, 0]
        d_bottom = centers[:, 1] + 0.5
        d_top = 0.5 - centers[:, 1]
        
        # Apply force if distance < cutoff
        # Use np.where to avoid division by zero or applying force when d >= cutoff
        
        # Left wall
        mask_l = d_left < cutoff
        forces[mask_l, 0] += k_wall / (d_left[mask_l] ** 2)
        
        # Right wall
        mask_r = d_right < cutoff
        forces[mask_r, 0] -= k_wall / (d_right[mask_r] ** 2)
        
        # Bottom wall
        mask_b = d_bottom < cutoff
        forces[mask_b, 1] += k_wall / (d_bottom[mask_b] ** 2)
        
        # Top wall
        mask_t = d_top < cutoff
        forces[mask_t, 1] -= k_wall / (d_top[mask_t] ** 2)
        
        # Update velocities (Symplectic Euler: v_new = v_old + F*dt, then x_new = x_old + v_new*dt)
        velocities += forces * dt
        velocities *= damping
        
        # Update positions
        centers += velocities * dt
        
        # Hard constraints: keep centers within [-0.5, 0.5]
        # If a particle goes outside, clip it and zero its velocity component.
        # This acts as a hard wall, but the soft force should prevent this mostly.
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Zero out velocities if hitting the wall to prevent energy gain
        # This is a simple approximation.
        velocities[centers[:, 0] == -0.5, 0] = 0
        velocities[centers[:, 0] == 0.5, 0] = 0
        velocities[centers[:, 1] == -0.5, 1] = 0
        velocities[centers[:, 1] == 0.5, 1] = 0

    return centers

# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    # Compute radius for verification
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    
    bound_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    print(f"Radius: {min(min_pair, min_bound):.10f}")
