"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialize with a hexagonal lattice for a good starting point
    # This mimics the seed but prepares for optimization
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98

    # 2. Force-Directed Optimization
    # We want to maximize r = min(min_i(1-|c_i|), min_{i<j} |c_i-c_j|/2)
    
    def get_radius(pts):
        # Distance to boundary
        d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        # Min distance to boundary
        r_boundary = np.min(d_boundary)
        
        # Pairwise distances
        # Calculate all pairwise distances
        # To save memory, we can do it in chunks or just use broadcasting for N=50
        # N=50 is small enough for full matrix
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        # Min pairwise distance
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        return min(r_boundary, r_pair)

    current_r = get_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Optimization parameters
    max_iters = 2000
    step_size = 0.05
    # We will iteratively try to push points apart to increase the minimum radius
    
    for i in range(max_iters):
        # Calculate forces
        forces = np.zeros_like(points)
        
        # 1. Pairwise repulsion
        # We want to increase the minimum distance.
        # A simple heuristic: push apart any pair that is closer than the current target.
        # Target is roughly current_r * 2.
        # But to be effective, we should push based on the "tightness" of the constraint.
        
        # Let's use a force model that pushes apart all pairs proportional to 1/d^2 or similar,
        # but weighted to focus on the closest pairs.
        
        # Calculate pairwise distances
        dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Force magnitude: We want to maximize the minimum distance.
        # Let's push apart pairs with distance < 2 * current_r * 1.1 (some margin)
        # Actually, a standard approach is to use a repulsion force for all pairs,
        # which naturally pushes the system to a "Voronoi" like optimal packing.
        
        # F_ij = k * (1/d^2) * (r_j - r_i) ? No, simple repulsion.
        # Let's use: F_ij = (1 / d^2) * (unit_vector_ij)
        # This is expensive for N=50? No, 50x50 is tiny.
        
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                d = dists[i_idx, j_idx]
                if d < 1e-9:
                    continue
                # Repulsion force
                # We want to push them apart.
                # Scale factor: stronger when closer.
                # We also want to ensure they don't overlap with the boundary.
                
                # Vector from i to j
                vec = points[j_idx] - points[i_idx]
                unit_vec = vec / d
                
                # Force magnitude
                # Use a soft repulsion
                f_mag = 1.0 / (d * d)
                
                forces[i_idx] -= unit_vec * f_mag
                forces[j_idx] += unit_vec * f_mag

        # 2. Boundary attraction/repulsion
        # We want to keep points inside.
        # If a point is close to the boundary, push it in.
        # If we just use repulsion from pairs, points will fly out.
        # So we need an inward force or a constraint.
        # Let's add an inward force proportional to distance from center?
        # Or better: if distance > 1 - r_current, push in.
        
        for i_idx in range(N):
            d_center = np.linalg.norm(points[i_idx])
            # Target boundary distance is 1 - current_r
            # But we want to maximize r, so we want to allow points to go further out
            # if the packing allows it.
            # However, to maintain validity, we must respect the current best r or a target r.
            
            # Let's use a "soft" boundary: 
            # Force = -k * (d_center - (1 - current_r)) * (points[i_idx] / d_center)
            # If d_center < 1 - current_r, force is 0 (or weakly outward to fill space?)
            # If d_center > 1 - current_r, force is inward.
            
            # Actually, if we just use pairwise repulsion, the system will expand.
            # We need to constrain the expansion.
            # Let's project points back to the unit circle after each step?
            # No, that distorts the packing.
            
            # Alternative: The objective is max r.
            # r = min(min_boundary, min_pair/2).
            # If we maximize min_pair/2, min_boundary will also increase if points move out.
            # But if points move out, min_boundary decreases.
            # The optimal solution balances these.
            
            # Let's add a force that pulls points towards the center if they are too close to the boundary relative to the current r.
            # But we don't know the "ideal" boundary distance.
            
            # Let's try: Force_inward = k * max(0, d_center - (1 - current_r * 0.9)) * unit_vec_center
            # This keeps them safely inside while allowing expansion.
            
            if d_center > 1e-9:
                unit_center = points[i_idx] / d_center
                # If they are too close to the edge, push in
                limit = 1.0 - current_r
                if d_center > limit:
                    f_in = (d_center - limit) * 10.0 # Strong inward force
                    forces[i_idx] -= unit_center * f_in
                else:
                    # Weak outward force to fill space?
                    # No, let the pairwise repulsion push them out.
                    pass

        # Update positions
        points += forces * step_size
        
        # Ensure points are strictly inside the unit circle to avoid invalid states
        # If a point goes outside, pull it back
        norms = np.linalg.norm(points, axis=1)
        for i_idx in range(N):
            if norms[i_idx] > 1.0:
                points[i_idx] = points[i_idx] / norms[i_idx] * 0.999

        # Calculate new radius
        new_r = get_radius(points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = points.copy()
            # Adaptive step size?
            # step_size = 0.9 * step_size # Slow down when improving
        else:
            # If we didn't improve, maybe the step was too big or we are stuck.
            # Revert to best points and try smaller step?
            # Or just continue, as forces might still be pushing in a useful direction.
            # Let's decay step size slightly to allow finer tuning
            step_size *= 0.99

    return best_points
# EVOLVE-BLOCK-END
