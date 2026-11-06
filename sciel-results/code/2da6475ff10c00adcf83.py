"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    def generate_hex_lattice(n_target):
        """
        Generate a hexagonal lattice of points that fits inside the unit circle.
        We binary search for the spacing d such that the number of points is close to n_target.
        """
        def count_points(d):
            if d <= 0:
                return 0
            k = int(1.0 / d) + 2
            sqrt3 = np.sqrt(3.0)
            count = 0
            for j in range(-k, k + 1):
                offset = 0.5 * d if (j % 2 != 0) else 0.0
                y = j * d * sqrt3 / 2.0
                if abs(y) > 1.0:
                    continue
                for i in range(-k, k + 1):
                    x = i * d + offset
                    if x*x + y*y <= 1.0:
                        count += 1
            return count

        # Binary search for d
        d_low = 0.01
        d_high = 0.5
        # We want count >= n_target, so we can find the largest d such that count >= n_target
        # Actually, we want count close to n_target.
        # Let's find d such that count is as close to n_target as possible.
        
        best_d = None
        best_diff = float('inf')
        
        # Coarse search
        for _ in range(50):
            d_mid = (d_low + d_high) / 2.0
            c = count_points(d_mid)
            if c > n_target:
                # d is too small (too many points), increase d
                d_low = d_mid
            else:
                # d is too large (too few points), decrease d
                d_high = d_mid
            
            # Check if we are close
            diff = abs(c - n_target)
            if diff < best_diff:
                best_diff = diff
                best_d = d_mid

        # Generate points with best_d
        d = best_d
        k = int(1.0 / d) + 2
        sqrt3 = np.sqrt(3.0)
        points = []
        for j in range(-k, k + 1):
            offset = 0.5 * d if (j % 2 != 0) else 0.0
            y = j * d * sqrt3 / 2.0
            if abs(y) > 1.0:
                continue
            for i in range(-k, k + 1):
                x = i * d + offset
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than N points, keep the N closest to the center
        if len(points) > N:
            dists_to_origin = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists_to_origin)
            points = points[indices[:N]]
        elif len(points) < N:
            # If we have fewer, add random points near the boundary
            # This is a fallback, shouldn't happen with good binary search
            needed = N - len(points)
            angles = np.random.rand(needed) * 2 * np.pi
            radii = np.sqrt(np.random.rand(needed)) * 0.9 + 0.1
            new_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            points = np.vstack([points, new_points])

        return points

    # 1. Initialization
    points = generate_hex_lattice(N)
    
    # Ensure validity
    radii = np.hypot(points[:, 0], points[:, 1])
    outside = radii > 1.0
    if np.any(outside):
        points[outside] = points[outside] / radii[outside, np.newaxis]

    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Force-Based Relaxation with Soft-Min Surrogate
    beta = 500.0  # Higher beta for sharper minimum
    step_size = 0.005
    decay_rate = 0.99995
    
    # Main optimization loop
    for it in range(20000):
        if time.time() > deadline - 10.0: # Leave 10s for final checks
            break
            
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Compute soft-min weights
        x_pair = dists / 2.0
        x_bound = boundary_dists
        
        min_x_pair = np.min(x_pair)
        min_x_bound = np.min(x_bound)
        min_x = min(min_x_pair, min_x_bound)
        
        # Weights
        # To avoid underflow, we can shift, but np.exp handles small values well
        # If beta * (x - min_x) is large, exp is 0.
        
        w_pair = np.exp(-beta * (x_pair - min_x))
        np.fill_diagonal(w_pair, 0.0)
        
        w_bound = np.exp(-beta * (x_bound - min_x))
        
        Z = np.sum(w_pair) + np.sum(w_bound)
        
        if Z == 0 or not np.isfinite(Z):
            break
        
        w_pair /= Z
        w_bound /= Z
        
        # Forces
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        factor = w_pair / (2.0 * safe_dists)
        np.fill_diagonal(factor, 0.0)
        
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        forces = beta * (F_pair + F_bound)
        
        # Update
        points += forces * step_size
        
        # Project to unit disk
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / radii[outside, np.newaxis]
            
        # Decay step size
        step_size *= decay_rate
        
        # Occasional perturbation
        if it % 50 == 0 and it > 0:
            num_perturb = N // 30
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * 0.005
            points[idx] += perturbations
            radii = np.hypot(points[:, 0], points[:, 1])
            outside = radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / radii[outside, np.newaxis]

    # Final Polish: High precision local search
    # Try to improve best_points with very small steps
    polish_points = best_points.copy()
    polish_step = 0.0001
    for it in range(1000):
        if time.time() > deadline:
            break
        current_r = compute_radius(polish_points)
        if current_r > best_r:
            best_r = current_r
            best_points = polish_points.copy()
        
        # Compute gradient for polish
        diff = polish_points[:, np.newaxis, :] - polish_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        radii = np.hypot(polish_points[:, 0], polish_points[:, 1])
        boundary_dists = 1.0 - radii
        
        x_pair = dists / 2.0
        x_bound = boundary_dists
        min_x = min(np.min(x_pair), np.min(x_bound))
        
        w_pair = np.exp(-1000.0 * (x_pair - min_x))
        np.fill_diagonal(w_pair, 0.0)
        w_bound = np.exp(-1000.0 * (x_bound - min_x))
        Z = np.sum(w_pair) + np.sum(w_bound)
        if Z == 0:
            break
        w_pair /= Z
        w_bound /= Z
        
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        factor = w_pair / (2.0 * safe_dists)
        np.fill_diagonal(factor, 0.0)
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = polish_points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        forces = 1000.0 * (F_pair + F_bound)
        polish_points += forces * polish_step
        
        radii = np.hypot(polish_points[:, 0], polish_points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            polish_points[outside] = polish_points[outside] / radii[outside, np.newaxis]

    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
