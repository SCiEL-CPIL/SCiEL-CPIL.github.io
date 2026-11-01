import numpy as np
from scipy.optimize import minimize
import time

N = 300

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal lattice
    # Approximate density: N circles in unit circle area pi.
    # Area per circle ~ pi/N. Radius ~ sqrt(pi/N). 
    # Spacing in hex grid ~ 2*r.
    # Let's just generate a large hex grid and pick the best N.
    
    def generate_hex_points(n_points):
        # Estimate spacing
        # Area of unit circle is pi. N circles of radius r.
        # Packing density ~ 0.9.
        # N * pi * r^2 ~ 0.9 * pi  => r ~ sqrt(0.9/N)
        r_est = np.sqrt(0.9 / n_points)
        spacing = 2.0 * r_est * 1.05 # slightly larger to ensure fit initially
        
        # Generate points
        points = []
        # Range of i, j
        # Max radius 1.
        # x = (i + j/2) * s
        # y = j * s * sqrt(3)/2
        # We need to cover radius 1.
        # Max j ~ 1 / (s*sqrt(3)/2)
        max_j = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 2
        max_i = int(2.0 / spacing) + 2
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 1.0:
                    points.append((x, y))
        
        points = np.array(points)
        
        if len(points) < n_points:
            # If not enough points, reduce spacing
            return generate_hex_points(n_points, spacing * 0.9)
            
        # Sort by distance to origin? Or just take first N?
        # Taking points closest to origin usually gives a more compact cluster, 
        # but for packing in a circle, we want them distributed.
        # Actually, the hex lattice cut by the circle is already well distributed.
        # We just need N points. 
        # To be safe, we can select the N points that are "most central" 
        # or just the first N. The first N from the loop might be biased.
        # Let's sort by radius and take the N points with smallest radius?
        # No, that creates a dense center and sparse edge.
        # Better: Take all points, if > N, we need to remove some.
        # Removing points at the edge is better.
        # So sort by radius descending, and keep the first N?
        # Or sort by radius ascending and keep the first N?
        # If we keep the innermost N, the outer ones are cut off.
        # This is a valid packing.
        
        radii = np.hypot(points[:, 0], points[:, 1])
        # Keep the N points closest to the origin?
        # This ensures they are well inside.
        # However, the optimal packing uses the full area.
        # A better initial guess is to take a random subset or the first N from a shuffled list?
        # Let's try sorting by radius and taking the N points with the SMALLEST radii.
        # This might be too compact.
        
        # Alternative: Take the points that are within the unit circle.
        # If we have more than N, we can try to optimize.
        # But we need exactly N.
        # Let's just take the first N from the generated list, but shuffle first to avoid bias?
        # Or, more simply: generate a grid, filter to unit circle, if len > N, 
        # we can just take a random subset? No, that might be bad.
        
        # Let's stick to the hex grid points inside the circle.
        # If we have more than N, we can remove the ones furthest from the center?
        # That leaves a compact blob.
        # If we remove the ones closest to the center? That leaves a ring.
        
        # Actually, for N=300, the hex grid will have roughly 300-400 points in the unit circle.
        # Let's just take the N points with the smallest distance to the origin.
        # This is a safe, valid packing.
        
        idx = np.argsort(radii)[:n_points]
        return points[idx]

    # Recursive helper with default spacing
    def gen_hex(n):
        # Estimate spacing based on N
        r_est = np.sqrt(0.9 / n)
        spacing = 2.0 * r_est * 1.02
        max_j = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 2
        max_i = int(2.0 / spacing) + 2
        points = []
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) < n:
            return gen_hex(n, spacing * 0.95)
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii)[:n]
        return points[idx]

    try:
        x0 = gen_hex(N)
    except Exception:
        # Fallback: random points
        x0 = np.random.uniform(-0.9, 0.9, (N, 2))
        # Ensure inside
        norms = np.linalg.norm(x0, axis=1)
        x0 = x0 / norms[:, np.newaxis] * 0.9

    # Define objective function
    def objective(x, beta):
        x = x.reshape(N, 2)
        
        # Pairwise distances
        # Use broadcasting for efficiency, but N=300 -> 300x300 matrix is fine.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists = dists[mask]
        
        # Boundary distances
        norms = np.linalg.norm(x, axis=1)
        b_dists = 1.0 - norms
        
        # Combine all distances
        all_dists = np.concatenate([dists, b_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * dist)))
        # To avoid overflow, subtract max? No, exp(-beta * dist) is small for large dist.
        # We want to minimize the "smallest" distance.
        # So we want to minimize -softmin.
        # softmin(d) = -1/beta * log(sum(exp(-beta * d)))
        # We want to MAXIMIZE softmin(d).
        # So we MINIMIZE -softmin(d).
        
        # Numerical stability:
        # log_sum_exp(-beta * d) = -beta * max(d) + log(sum(exp(-beta * (d - max(d)))))
        # Let m = max(all_dists)
        # term = -beta * all_dists
        # log_sum = m * beta + log(sum(exp(term - beta * m))) ? No.
        
        # Let's just compute directly if beta is not too large.
        # If beta is large, exp(-beta * d) underflows for d > 0.
        # We only care about the smallest distances.
        
        # Better approach for stability:
        # We want to approximate min(all_dists).
        # Let d_min = min(all_dists).
        # The log-sum-exp of -beta * d is dominated by the term with smallest d.
        
        # To avoid underflow/overflow issues, we can shift by the current min?
        # Or just use a moderate beta.
        
        # Let's use the standard formula with clipping.
        # If beta is large, only the smallest few terms matter.
        
        # Calculate log-sum-exp of (-beta * all_dists)
        # Let z = -beta * all_dists
        # log_sum_exp(z) = max(z) + log(sum(exp(z - max(z))))
        
        z = -beta * all_dists
        z_max = np.max(z)
        if z_max > 700: # overflow in exp
             z = z - z_max
             z_max = 0
        elif z_max < -700: # underflow
             z = z - z_max
             z_max = 0
            
        sum_exp = np.sum(np.exp(z))
        log_sum_exp = z_max + np.log(sum_exp)
        
        soft_min = -log_sum_exp / beta
        
        # We want to maximize soft_min, so minimize -soft_min
        return -soft_min

    # We will run optimization with increasing beta
    best_x = x0.copy()
    best_score = -np.inf
    
    # Calculate current score for tracking
    def calc_score(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        min_dist = np.min(dists[mask])
        norms = np.linalg.norm(x, axis=1)
        min_bound = np.min(1.0 - norms)
        return min(min_dist / 2.0, min_bound)
    
    current_best_r = calc_score(best_x)
    
    # Optimization loop
    # Betas: start smooth, end sharp
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Reshape for optimizer
        x_flat = best_x.flatten()
        
        # Define the specific objective for this beta
        def obj_flat(xf):
            return objective(xf, beta)
        
        # Gradient is not provided, L-BFGS-B will approximate it.
        # For better performance, we could provide gradient, but it's complex.
        # Let's try without gradient first.
        
        try:
            res = minimize(
                obj_flat, 
                x_flat, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6, 'disp': False}
            )
            
            if res.success or res.fun < objective(x_flat, beta):
                new_x = res.x.reshape(N, 2)
                # Check if new_x is valid (finite)
                if np.all(np.isfinite(new_x)):
                    new_r = calc_score(new_x)
                    if new_r > current_best_r:
                        current_best_r = new_r
                        best_x = new_x
        except Exception as e:
            pass

    # Final Polish:
    # Use the best_x found.
    # Perform a few iterations of a simple "push" algorithm to resolve any tiny overlaps
    # or boundary violations that might have been smoothed over.
    
    # Simple gradient ascent on the raw min distance (subgradient)
    # Or just return best_x.
    
    # One more check: ensure no points are outside
    norms = np.linalg.norm(best_x, axis=1)
    # If any are outside, project them back
    for i in range(N):
        if norms[i] > 0.999:
            best_x[i] = best_x[i] / norms[i] * 0.999

    return best_x

# EVOLVE-BLOCK-END
