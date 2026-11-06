import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair = np.sqrt(np.min(dists_sq)) / 2.0
        return min(min_pair, min_boundary)

    def generate_hex_init():
        """Generate a hexagonal lattice configuration scaled to fit."""
        # Estimate spacing
        # Area of hex grid ~ N * spacing^2 * sqrt(3)/2
        # Area of circle ~ pi
        # spacing ~ sqrt(2 * pi / (N * sqrt(3)/2)) = sqrt(4 * pi / (N * sqrt(3)))
        spacing = np.sqrt(4.0 * np.pi / (N * np.sqrt(3.0)))
        
        # Generate grid
        dx = spacing
        dy = spacing * np.sqrt(3.0) / 2.0
        
        # Determine range
        range_x = int(np.ceil(1.0 / dx)) + 1
        range_y = int(np.ceil(1.0 / dy)) + 1
        
        pts = []
        for j in range(-range_y, range_y + 1):
            for i in range(-range_x, range_x + 1):
                x = i * dx + (j % 2) * dx / 2.0
                y = j * dy
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, randomly select N
        if len(pts) > N:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            # Add random points if too few
            while len(pts) < N:
                p = np.random.randn(2)
                p /= np.linalg.norm(p) * 0.9
                pts = np.vstack([pts, p])
        
        return pts[:N]

    def generate_spiral_init():
        """Generate a golden-angle spiral configuration."""
        ga = np.pi * (3.0 - np.sqrt(5.0))
        indices = np.arange(N)
        R_max = 0.95
        radii = np.sqrt(indices / N) * R_max
        angles = indices * ga
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    def optimize_for_r(pts, r_target, max_iters=500, deadline_check=None):
        """
        Optimize points to satisfy constraints for a given radius r_target.
        Minimizes violation of constraints.
        Returns: (best_pts, min_violation)
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_violation = np.inf
        lr = 0.005
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 1e-10) # Avoid div by zero
            dists = np.sqrt(dists_sq)
            
            # Constraint 1: d_ij >= 2 * r_target
            # Violation: max(0, 2*r - d_ij)
            viol_pair = np.maximum(0, 2 * r_target - dists)
            
            # Gradient for pair term:
            # d/dc_i sum_{j!=i} max(0, 2r - d_ij)^2
            # = sum_{j!=i} 2 * max(0, 2r - d_ij) * (c_i - c_j) / d_ij * I(d_ij < 2r)
            # Note: viol_pair is already max(0, ...).
            # grad_pair[i] = 2 * sum_j viol_pair[i,j] * (c_i - c_j) / d_ij * I(d_ij < 2r)
            # We can use viol_pair / dists as weight, but need to mask diagonal and non-violating pairs.
            # Actually, viol_pair is 0 for non-violating pairs, so multiplication handles it.
            # But we divide by dists.
            weights = viol_pair / dists
            np.fill_diagonal(weights, 0.0)
            
            grad_pair = 2.0 * np.sum(diff * weights[:, :, np.newaxis], axis=1)
            
            # Constraint 2: |c_i| <= 1 - r_target
            radii = np.hypot(pts[:, 0], pts[:, 1])
            limit = 1.0 - r_target
            viol_bound = np.maximum(0, radii - limit)
            
            # Gradient for bound term:
            # d/dc_i max(0, |c_i| - limit)^2
            # = 2 * max(0, |c_i| - limit) * c_i / |c_i| * I(|c_i| > limit)
            # viol_bound is 0 if not violating.
            # weights_bound[i] = 2 * viol_bound[i] / radii[i]
            radii_safe = np.where(radii > 1e-10, radii, 1e-10)
            weights_bound = 2.0 * viol_bound / radii_safe
            grad_bound = pts * weights_bound[:, np.newaxis]
            
            grad = grad_pair + grad_bound
            
            # Update
            pts -= grad * lr
            
            # Project inside unit circle (hard constraint for validity, though r_target < 1)
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
            
            # Check violation
            # Recompute min violation for tracking
            # min_pair_dist = min(dists)
            # min_bound_margin = min(1 - radii)
            # We want to know if constraints are satisfied.
            # Simplified: compute score for this r? No, just check max violation.
            # Actually, we just need to know if we can increase r.
            # Let's compute the actual minimum distance and boundary margin.
            min_d = np.min(dists)
            min_m = np.min(1.0 - radii)
            current_r = min(min_d / 2.0, min_m)
            
            if current_r > best_violation: # Here best_violation is actually best_r achieved so far in this inner loop?
                # Wait, the function is called to optimize FOR a specific r_target.
                # If current_r >= r_target, we are good.
                # Let's track the best r achieved.
                pass
        
        # After loop, return the best points found (maximizing r)
        # We need to re-evaluate the best pts.
        # Let's just return the final pts and its score.
        return pts, compute_score(pts)

    def optimize_adaptive(pts, deadline_check=None):
        """
        Iteratively increase r and optimize.
        """
        best_pts = pts.copy()
        best_r = compute_score(pts)
        
        # Start with a small r
        r = 0.04
        max_r = 0.06
        step_r = 0.001
        
        while r < max_r:
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Optimize for current r
            # Use a simple gradient descent loop here
            pts_curr = best_pts.copy()
            lr = 0.002
            max_iters = 200
            
            for it in range(max_iters):
                if deadline_check is not None and time.time() > deadline_check:
                    break
                
                diff = pts_curr[:, np.newaxis, :] - pts_curr[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                np.fill_diagonal(dists_sq, 1e-10)
                dists = np.sqrt(dists_sq)
                
                viol_pair = np.maximum(0, 2 * r - dists)
                weights = viol_pair / dists
                np.fill_diagonal(weights, 0.0)
                grad_pair = 2.0 * np.sum(diff * weights[:, :, np.newaxis], axis=1)
                
                radii = np.hypot(pts_curr[:, 0], pts_curr[:, 1])
                limit = 1.0 - r
                viol_bound = np.maximum(0, radii - limit)
                radii_safe = np.where(radii > 1e-10, radii, 1e-10)
                weights_bound = 2.0 * viol_bound / radii_safe
                grad_bound = pts_curr * weights_bound[:, np.newaxis]
                
                grad = grad_pair + grad_bound
                pts_curr -= grad * lr
                
                # Project
                new_radii = np.hypot(pts_curr[:, 0], pts_curr[:, 1])
                outside = new_radii > 1.0
                if np.any(outside):
                    pts_curr[outside] = pts_curr[outside] / new_radii[outside, np.newaxis]
            
            # Check if we satisfied r
            curr_score = compute_score(pts_curr)
            if curr_score >= r - 1e-6:
                best_pts = pts_curr
                best_r = curr_score
                r += step_r
            else:
                # Failed to satisfy r, break
                break
        
        return best_pts, best_r

    # Main execution
    # Try multiple initializations
    inits = [
        generate_hex_init(),
        generate_spiral_init(),
        np.random.randn(N, 2) * 0.5 # Random
    ]
    
    best_pts_overall = None
    best_score_overall = 0.0
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Optimize
        pts, score = optimize_adaptive(init, deadline_check=deadline - 5)
        
        if score > best_score_overall:
            best_score_overall = score
            best_pts_overall = pts
    
    # Final validation
    if best_pts_overall is None:
        best_pts_overall = generate_spiral_init()
    
    # Ensure validity
    radii = np.hypot(best_pts_overall[:, 0], best_pts_overall[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts_overall)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts_overall = best_pts_overall * scale
    
    return best_pts_overall
