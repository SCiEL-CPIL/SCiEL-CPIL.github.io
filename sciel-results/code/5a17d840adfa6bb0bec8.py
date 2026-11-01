import numpy as np
import time
import random

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(pts):
        if len(pts) < 2:
            return 0.0
        # Efficient distance calculation for N=300
        # Using broadcasting might be memory heavy (300x300x2 is fine, but let's use scipy if available or manual)
        # For speed in tight loops, we can use a vectorized pairwise distance
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # min_pair = np.min(dist_sq) / 2.0
        # min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        # return min(min_pair, min_boundary)
        
        # Faster: use scipy.spatial.distance.pdist if available, but manual is safer for interface
        # Let's use a compact manual version for N=300 which is ~45k pairs.
        # To save time in the inner loop of SA, we will calculate the radius less frequently 
        # or use a fast approximation? No, we need exact score.
        # However, 45k distance calcs in Python loops is slow. Vectorized numpy is fast.
        
        # Vectorized pairwise distance
        # This creates a 300x300 matrix. Memory is fine.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Exclude diagonal
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.min(dist_sq) / 2.0
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        return min(min_pair, min_boundary)

    def init_ring_structure():
        """
        Initialize with a hexagonal-like lattice but structured into rings.
        We estimate the number of points per ring based on the area.
        """
        # Approximate optimal packing density for 300 circles
        # Area of 300 circles = 300 * pi * r^2
        # Area of unit circle = pi
        # 300 * r^2 approx 1 => r approx 1/sqrt(300) = 0.0577
        # Best known r = 0.05315
        # Let's generate points in a hex lattice first, then bin them by radius.
        
        # Generate hex lattice
        r_est = 0.053
        spacing = 2 * r_est
        pts = []
        # Generate enough points
        k = int(1.0 / spacing) + 5
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing / 2.0
                y = j * spacing * np.sqrt(3) / 2.0
                if np.hypot(x, y) <= 1.0 - r_est:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # Sort by radius
        radii = np.hypot(pts[:, 0], pts[:, 1])
        indices = np.argsort(radii)
        pts = pts[indices]
        
        # Select N points
        if len(pts) >= N:
            pts = pts[:N]
        else:
            # Fallback: random
            pts = np.random.uniform(-1, 1, (N, 2))
            radii = np.hypot(pts[:, 0], pts[:, 1])
            mask = radii > 1
            pts[mask] /= radii[mask, np.newaxis]
            pts *= 0.9
            
        # Group into rings
        # We will define rings by quantizing the radius.
        # Let's define a function to assign rings based on distance from center.
        # For the search, we need a fixed set of "slots" (r, theta) where we can swap thetas.
        
        # Let's identify natural rings.
        # We'll just use the current radii as the fixed radii for the slots, 
        # but we need to group points into rings to allow swapping within/between rings.
        
        # A simpler discrete search: 
        # 1. Fix the radial coordinates (r_i) of all N points.
        # 2. Randomize the angular coordinates (theta_i).
        # 3. Use Simulated Annealing to swap angles between points.
        # This preserves the radial structure (which is often good in hex-like packings) 
        # but explores angular permutations.
        
        return pts

    # 1. Initialize
    pts = init_ring_structure()
    
    # Ensure validity
    radii = np.hypot(pts[:, 0], pts[:, 1])
    mask_out = radii > 1.0
    if np.any(mask_out):
        pts[mask_out] = pts[mask_out] / radii[mask_out, np.newaxis] * 0.9999

    best_pts = pts.copy()
    best_score = compute_radius(best_pts)
    
    # 2. Discrete Angular Search (Simulated Annealing)
    # We fix the radial distance of each point. We only permute the angles.
    # Actually, fixing radii strictly might be too restrictive if the initial lattice is imperfect.
    # Better: Fix the *set* of radii, but allow points to swap their (r, theta) assignments?
    # No, that's just permuting points.
    # The key is to change the *geometry*.
    
    # Let's try a different discrete move: 
    # "Local Angular Jitter" + "Radial Relaxation"
    # But the directive asks for a STRUCTURAL change.
    # "Radial-ring decomposition with smooth soft-min" was inspiration 1.
    # My proposed strategy: "Discrete angular reordering search on structured radial-ring initialization".
    
    # Implementation:
    # 1. Convert to polar (r, theta).
    # 2. Fix r_i.
    # 3. Perform SA on theta_i. Moves: swap theta_i and theta_j for random i, j.
    #    Or: rotate a subset of points by a small angle? No, that's continuous.
    #    Swap is discrete and explores the combinatorial space of which angle goes with which radius.
    #    Wait, if I swap angles, I am effectively permuting the points. 
    #    Permuting points doesn't change the set of distances. It changes the *labels*.
    #    The score depends on the set of distances, not labels.
    #    So swapping angles between two points with the SAME radius does nothing.
    #    Swapping angles between two points with DIFFERENT radii changes the positions.
    #    Point i was at (r_i, theta_i), Point j at (r_j, theta_j).
    #    After swap: i at (r_i, theta_j), j at (r_j, theta_i).
    #    This IS a valid move that changes the geometry.
    
    r_polar = np.hypot(pts[:, 0], pts[:, 1])
    theta_polar = np.arctan2(pts[:, 1], pts[:, 0])
    
    # Sort by radius to make "rings" explicit? 
    # Actually, we just keep the array of r and theta.
    # Move: pick two distinct indices i, j. Swap theta[i] and theta[j].
    
    current_theta = theta_polar.copy()
    current_score = best_score
    
    # SA Parameters
    T = 1e-4
    T_min = 1e-8
    alpha = 0.9995 # Cooling rate
    
    steps = 0
    max_steps = 50000 # Adjust based on speed
    
    # Precompute r for speed
    r_fixed = r_polar
    
    def get_score_from_theta(theta):
        # Reconstruct points
        x = r_fixed * np.cos(theta)
        y = r_fixed * np.sin(theta)
        pts_curr = np.column_stack((x, y))
        return compute_radius(pts_curr)
    
    # To speed up compute_radius in the inner loop, we can optimize it.
    # But N=300 is small. 50k steps * (300^2 ops) = 4.5e9 ops. 
    # Vectorized numpy handles 300x300 in ~1ms? Maybe 5ms.
    # 50k * 5ms = 250s. Too slow.
    # We need to reduce steps or speed up score.
    
    # Optimization: Only compute score if the move is "promising"? No.
    # We can subsample? No.
    # Let's reduce max_steps to 5000 and see.
    # Or use a faster distance metric? No, must be exact.
    
    # Let's check speed of compute_radius.
    # 300x300 matrix creation is fast.
    # Let's aim for 20000 steps.
    
    max_steps = 20000
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
            
        # Random move: Swap angles of two random points
        i, j = random.sample(range(N), 2)
        
        # Save old
        old_ti = current_theta[i]
        old_tj = current_theta[j]
        
        # Apply
        current_theta[i] = old_tj
        current_theta[j] = old_ti
        
        new_score = get_score_from_theta(current_theta)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                # Update best points
                x_best = r_fixed * np.cos(current_theta)
                y_best = r_fixed * np.sin(current_theta)
                best_pts = np.column_stack((x_best, y_best))
        else:
            # Revert
            current_theta[i] = old_ti
            current_theta[j] = old_tj
            
        T *= alpha
        steps += 1
        
    # 3. Final Polish: Local Gradient Descent on the best found topology
    # The discrete search finds a good topology (set of radii and angles).
    # Now we can use the original gradient method to refine the exact positions,
    # allowing radii to change slightly. This combines the benefits of both.
    
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        def obj_grad(x_flat):
            X = x_flat.reshape(N, 2)
            # Compute radius
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_pair = np.min(dist_sq) / 2.0
            radii = np.hypot(X[:, 0], X[:, 1])
            min_boundary = np.min(1.0 - radii)
            val = min(min_pair, min_boundary)
            
            # Gradient of min is tricky. Use soft min or just subgradient?
            # For polish, L-BFGS-B with a smooth approximation is better.
            # But we are short on time. Let's just return -val and use finite diff? No, too slow.
            # Let's use the soft min gradient from the original code, but start from the SA result.
            
            # Re-implement soft min gradient briefly
            alpha_sm = 200.0
            iu = np.triu_indices(N, k=1)
            pair_vals = dist_sq[iu] / 2.0 # Wait, dist_sq is squared. Need sqrt for distance.
            # My compute_radius used dist_sq/2 for pair min? 
            # Score is min(dist/2). dist = sqrt(dist_sq).
            # So pair_vals = sqrt(dist_sq)/2.
            
            dist = np.sqrt(np.maximum(dist_sq, 1e-15))
            pair_vals = dist[iu] / 2.0
            boundary_vals = 1.0 - radii
            all_vals = np.concatenate([pair_vals, boundary_vals])
            
            min_val = np.min(all_vals)
            all_vals_clipped = np.clip(all_vals, min_val - 10, min_val + 10)
            z = -alpha_sm * all_vals_clipped
            z_max = np.max(z)
            sum_exp = np.sum(np.exp(z - z_max))
            soft_min = -1.0/alpha_sm * (np.log(sum_exp) + z_max)
            obj = -soft_min
            
            weights = np.exp(z - z_max) / sum_exp
            
            grad_X = np.zeros_like(X)
            
            # Pair grads
            pair_weights = weights[:len(pair_vals)]
            W = np.zeros((N, N))
            W[iu] = pair_weights
            W = W + W.T
            
            dist_safe = np.where(dist > 1e-10, dist, 1.0)
            inv_dist = 1.0 / dist_safe
            C = W * inv_dist / 2.0
            np.fill_diagonal(C, 0)
            
            row_sums = np.sum(C, axis=1)
            term1 = X * row_sums[:, np.newaxis]
            term2 = C @ X
            grad_X = term1 - term2
            grad_X = -grad_X # Because obj = -soft_min
            
            # Boundary grads
            boundary_weights = weights[len(pair_vals):]
            radii_safe = np.where(radii > 1e-10, radii, 1.0)
            boundary_grad = (boundary_weights / radii_safe)[:, np.newaxis] * X
            grad_X += boundary_grad
            
            return obj, grad_X.flatten()
        
        x0 = best_pts.flatten()
        result = minimize(
            obj_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.fun < 0: # It improved
            pts_new = result.x.reshape(N, 2)
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask_out = r_new > 1.0
            if np.any(mask_out):
                pts_new[mask_out] = pts_new[mask_out] / r_new[mask_out, np.newaxis] * 0.9999
            
            score_new = compute_radius(pts_new)
            if score_new > best_score:
                best_score = score_new
                best_pts = pts_new

    # Final validation
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask_out = r_final >= 1.0
    if np.any(mask_out):
        best_pts[mask_out] = best_pts[mask_out] / r_final[mask_out, np.newaxis] * 0.9999
        
    # Check coincident
    # For N=300, pdist is fast enough for final check
    from scipy.spatial.distance import pdist
    min_dist = np.min(pdist(best_pts))
    if min_dist < 1e-6:
         # Perturb
         np.random.seed(42)
         for i in range(N):
             for j in range(i+1, N):
                 d = np.linalg.norm(best_pts[i] - best_pts[j])
                 if d < 1e-6:
                     best_pts[j] += np.random.normal(0, 1e-4, size=2)
                     # Re-check boundary
                     r_j = np.hypot(best_pts[j, 0], best_pts[j, 1])
                     if r_j > 1.0:
                         best_pts[j] /= r_j * 1.0001

    return best_pts
