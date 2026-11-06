"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def generate_hex_lattice(r):
        """
        Generate a hexagonal lattice of points with circle radius r.
        Returns an array of points that fit inside the unit circle.
        """
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        # Range of indices to cover the unit circle
        k = int(1.0 / d) + 2
        
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            # Offset every other row by half a spacing
            offset = 0.5 * d if (j % 2 != 0) else 0.0
            for i in range(-k, k + 1):
                x = i * d + offset
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure the circle fits inside the unit circle
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        
        # If we have more than N points, select N points closest to the center
        # This ensures a compact, valid initial configuration
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        r = min( min_{i<j} dist(i,j)/2, min_i (1 - dist(i, origin)) )
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Distance to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Pairwise distances
        # Vectorized calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    # 1. Initialization: Find best hexagonal lattice
    # We want to find the largest r such that the hex lattice yields at least N points.
    # Smaller r -> larger spacing? No.
    # r is the radius of the circles. Spacing d = 2r.
    # If r is small, d is small. More points fit in the circle.
    # If r is large, d is large. Fewer points fit.
    # So we binary search for the largest r that still fits N points.
    
    r_low = 0.01
    r_high = 0.08
    
    # Ensure r_low gives enough points
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    # Binary search for max r in hex lattice
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    # r_low is the largest radius for which hex lattice works directly.
    points = generate_hex_lattice(r_low)
    
    # Safety net: if something went wrong, use random points
    if len(points) < N:
         angles = np.random.rand(N) * 2 * np.pi
         radii = np.sqrt(np.random.rand(N)) * 0.5
         points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Ensure we have exactly N points
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Vectorized Force-Based Relaxation
    # Strategy:
    # - Calculate forces to push overlapping circles apart and boundary-violating circles in.
    # - Use a cutoff distance to limit interactions to nearby pairs for efficiency (though N=300 is small enough for full matrix).
    # - Update positions with a decaying step size.
    
    # Parameters
    step_size = 0.005
    decay_rate = 0.9995
    cutoff_factor = 1.5  # Consider pairs within 1.5 * current_r for repulsion
    
    # Main optimization loop
    for it in range(20000):
        if time.time() > deadline:
            break
            
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # If we've reached a very high radius, we can stop or continue to polish
        # Theoretical max ~0.0577. Best known ~0.0531.
        
        # Calculate Forces
        forces = np.zeros_like(points)
        
        # 1. Pairwise Repulsion
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # We want to push apart pairs that are closer than 2 * target_r.
        # Let target_r be slightly larger than current_r to encourage expansion.
        # However, if we push too hard, we might break the structure.
        # A standard approach: Force is proportional to (1/d^2) or similar, active only if d < 2*r_target.
        
        # Let's use a soft repulsion for all pairs, but stronger for close ones.
        # F_ij = k * (1/d^2) * (p_i - p_j) / d  -- this is a bit complex to vectorize with norms.
        # Simpler: F_ij = k * (p_i - p_j) / d^3  (unit vector / d^2)
        # To avoid division by zero, we already set diagonal to inf, so 1/inf = 0.
        
        # We only apply force if dist < cutoff.
        # cutoff = 2.0 * max(current_r, 0.05) * cutoff_factor
        cutoff = 2.0 * max(current_r, 0.05) * cutoff_factor
        
        mask = dists < cutoff
        # Zero out diagonal in mask just in case, though dists diag is inf
        np.fill_diagonal(mask, False)
        
        # Compute force vectors
        # We want sum over j of (p_i - p_j) / d_ij^2 for d_ij < cutoff
        # This is equivalent to: sum over j of diff[i,j] / dists[i,j]^2 * mask[i,j]
        
        # To avoid division by zero for non-masked elements, we can create a safe distance matrix
        safe_dists = np.where(mask, dists, 1.0)
        inv_d2 = np.where(mask, 1.0 / (safe_dists**2), 0.0)
        
        # force_vec[i, j] = diff[i, j] * inv_d2[i, j]
        force_vecs = diff * inv_d2[:, :, np.newaxis]
        
        # Sum forces for each point i
        # forces[i] = sum_j force_vecs[i, j]
        forces += np.sum(force_vecs, axis=1)
        
        # Scale forces to prevent explosion
        forces *= 0.1
        
        # 2. Boundary Attraction
        # If a point is too close to the boundary (dist > 1 - r), push it towards center.
        # We want to maximize the minimum distance to boundary.
        # So if dist_i > 1 - current_r, it is "active".
        # Force direction: -points[i] / |points[i]|
        # Strength: proportional to violation.
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_target = 1.0 - current_r
        
        # Mask for points too close to boundary
        # Actually, we want to push ALL points slightly in if they are near the boundary,
        # but specifically those that define the minimum boundary distance.
        # Let's apply a force to any point where radii > 1 - current_r * 1.1
        boundary_mask = radii > (1.0 - current_r * 1.1)
        
        if np.any(boundary_mask):
            # Normalized position vectors
            # Avoid division by zero
            safe_radii = np.where(radii > 1e-6, radii, 1.0)
            unit_vecs = points / safe_radii[:, np.newaxis]
            
            # Violation amount
            violations = np.maximum(radii - (1.0 - current_r), 0.0)
            
            # Force is -unit_vec * violation * strength
            boundary_forces = -unit_vecs * violations[:, np.newaxis] * 10.0
            forces[boundary_mask] += boundary_forces[boundary_mask]
            
            # Hard constraint: ensure points stay inside unit circle
            # If a point moves outside, project it back.
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
        
        # Update positions
        points += forces * step_size
        
        # Decay step size
        step_size *= decay_rate
        
        # Occasionally, do a small random perturbation to escape local minima
        # This is crucial for finding better packings.
        if it % 50 == 0:
            # Perturb a random subset of points
            num_perturb = N // 10
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * (step_size * 2)
            points[idx] += perturbations
            # Re-project to ensure validity
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]

    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
