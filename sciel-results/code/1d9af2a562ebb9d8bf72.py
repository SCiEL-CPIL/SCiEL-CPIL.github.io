"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 120s limit

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
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    def compute_forces(pts, target_r, softness=50.0):
        """
        Compute force vectors for each point using a soft-min surrogate.
        Returns an array of shape (N, 2) with force vectors.
        """
        n = len(pts)
        forces = np.zeros((n, 2))
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - radii
        
        # Use soft-min to get smooth gradients
        # For pairs: we want to maximize the minimum distance
        # Use a soft-min of (dist - 2*target_r) for pairs
        # and (boundary_dist - target_r) for boundaries
        
        # Pair forces: for each pair, if dist < 2*target_r, push apart
        # Using a smooth approximation
        min_dist_val = np.min(dists)
        min_boundary_val = np.min(boundary_dists)
        
        # Current effective radius
        current_r = min(min_dist_val / 2.0, min_boundary_val)
        
        # For the soft approach, we weight constraints by how close they are to the minimum
        # Use exponential weighting based on distance from the minimum
        
        # Pair contribution
        # For each point i, sum over all j != i of force from pair (i,j)
        # Force direction: from j to i (push i away from j)
        # Weight: exp(-(dist_ij - 2*target_r) / softness) if dist_ij < 2*target_r + margin
        
        margin = 0.02
        pair_weights = np.zeros_like(dists)
        active_mask = dists < (2.0 * target_r + margin)
        active_mask = active_mask & (dists > 1e-10)
        
        # Smooth weight: larger for constraints closer to the binding constraint
        # Use exp(-(dist - 2*r_target) / softness) for active pairs
        pair_weights[active_mask] = np.exp(-(dists[active_mask] - 2.0 * target_r) / softness)
        np.fill_diagonal(pair_weights, 0.0)
        
        # Normalize weights so they sum to something reasonable
        # Scale by 1/N to keep forces bounded
        pair_scale = 1.0 / N
        
        for i in range(n):
            # Sum of forces from all other points
            f = np.zeros(2)
            for j in range(n):
                if i == j or pair_weights[i, j] == 0:
                    continue
                d = pts[i] - pts[j]
                dist = dists[i, j]
                if dist > 1e-10:
                    f += pair_weights[i, j] * d / dist
            forces[i] += pair_scale * f
        
        # Boundary forces: push inward if close to boundary
        boundary_active = boundary_dists < (target_r + margin)
        for i in range(n):
            if boundary_active[i]:
                w = np.exp(-(boundary_dists[i] - target_r) / softness)
                d = -pts[i]  # Direction toward center
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    forces[i] += pair_scale * w * d / dist
        
        return forces, current_r

    def generate_hex_lattice(r):
        """Generate points on a hexagonal lattice with spacing 2r that fit inside the unit circle."""
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        # Range of indices to cover the unit circle
        k = int(1.0 / d) + 2
        
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure the circle fits inside the unit circle
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # 1. Initialization: Find a good hexagonal lattice starting point
    r_low = 0.01
    r_high = 0.08
    
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    points = generate_hex_lattice(r_low)
    
    if len(points) < N:
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.5
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points[:N]

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Force-based relaxation with smooth surrogate
    # Phase 1: Coarse optimization with larger steps
    # Phase 2: Fine polish with smaller steps
    
    def optimize_phase(pts, start_r, max_time, step_size, softness):
        """Run force-based optimization for a given phase."""
        points = pts.copy()
        current_r = compute_radius(points)
        best_local_r = current_r
        best_local_points = points.copy()
        target_r = current_r * 1.01  # Slightly above current to drive improvement
        
        while time.time() < max_time:
            # Compute forces
            forces, current_r = compute_forces(points, target_r, softness)
            
            # Update best if improved
            if current_r > best_local_r:
                best_local_r = current_r
                best_local_points = points.copy()
                # Adjust target_r to be slightly above current
                target_r = current_r * 1.005
            
            # Apply forces with step size
            # Normalize forces to prevent huge steps
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms) if len(force_norms) > 0 else 0
            if max_force > 0:
                forces = forces / max_force
            
            new_points = points + forces * step_size
            
            # Ensure validity
            if not np.all(np.isfinite(new_points)):
                # Reduce step size and try again
                step_size *= 0.5
                continue
            
            # Check if new points are valid (all inside unit circle)
            new_radii = np.hypot(new_points[:, 0], new_points[:, 1])
            if np.any(new_radii > 1.0):
                # Project back inside
                mask = new_radii > 1.0
                new_points[mask] = new_points[mask] / new_radii[mask]
            
            new_r = compute_radius(new_points)
            
            if new_r >= current_r:
                points = new_points
            else:
                # Try smaller step
                step_size *= 0.8
                if step_size < 1e-6:
                    break
        
        return best_local_points, best_local_r, step_size

    # Phase 1: Coarse optimization
    phase1_deadline = start_time + 90.0
    points, best_r, step_size = optimize_phase(points, best_r, phase1_deadline, 0.01, 30.0)
    best_points = points.copy()
    
    # Phase 2: Fine polish
    phase2_deadline = deadline - 5.0
    points, best_r, step_size = optimize_phase(points, best_r, phase2_deadline, 0.002, 100.0)
    best_points = points.copy()
    
    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
    
    return best_points
# EVOLVE-BLOCK-END
