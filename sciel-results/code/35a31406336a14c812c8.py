"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _compute_score(points):
    """
    Compute the packing score: min(min_pair_dist/2, min_boundary_dist)
    """
    if len(points) < 2:
        return 0.0
    
    # Compute pairwise distances
    # dist[i, j] = ||points[i] - points[j]||
    # To save memory and time, we compute the minimum distance efficiently
    # For N=300, N^2 is 90,000, which is manageable.
    
    # Boundary distance
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    # Using matrix operations for speed
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=-1))
    # This creates a 300x300 matrix.
    
    # More memory efficient way for pairwise min distance:
    # We only need the minimum distance, not the full matrix.
    # But for gradient, we might need more. For now, let's compute full matrix 
    # as N=300 is small enough.
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Mask the diagonal (self-distance)
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_boundary)
    return score

def _objective_and_grad(points, target_r):
    """
    Objective function to maximize. 
    We want to maximize the minimum distance.
    Equivalent to minimizing -min_distance.
    However, min is non-smooth.
    
    Alternative: Minimize the sum of squared violations of constraints:
    1. ||p_i - p_j|| >= 2*r
    2. ||p_i|| <= 1 - r
    
    Let's use a soft constraint approach.
    We want to maximize r such that constraints are satisfied.
    
    A common trick is to define an objective that penalizes violations.
    Let d_ij = ||p_i - p_j||. We want d_ij >= 2r.
    Let d_i = 1 - ||p_i||. We want d_i >= r.
    
    We can define a loss function:
    L = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - d_i)^2
    
    To maximize r, we can treat r as a variable or fix r and check feasibility.
    Since we want to find the best r, we can iteratively increase r and check if feasible.
    Or, we can define an objective that pushes points apart.
    
    Let's try maximizing a "soft" minimum distance.
    f(points) = log-sum-exp(-beta * distances)
    Minimizing this is equivalent to maximizing the minimum distance (for large beta).
    
    However, SLSQP works best with smooth objectives.
    
    Let's stick to a simpler approach:
    Maximize r subject to:
    ||p_i - p_j||^2 - (2r)^2 >= 0
    (1-r)^2 - ||p_i||^2 >= 0
    
    This is non-convex.
    
    Let's use a penalty method.
    Objective: -r + penalty * (violations)
    
    Or, simpler:
    We know the initial r from the lattice.
    We can try to increase r slightly and optimize positions to satisfy constraints.
    If we can't, we reduce r.
    
    Let's implement a function that takes points and a target radius r, 
    and returns a scalar objective to minimize (penalty for violations).
    """
    r = target_r
    n = len(points)
    
    # Boundary violations
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    viol_boundary = np.maximum(0, r - dists_boundary)
    penalty_boundary = np.sum(viol_boundary**2)
    
    # Pairwise violations
    # We need to be careful with memory.
    # For N=300, 45k pairs.
    # Let's compute in chunks if necessary, but 300x300 is fine.
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We want dists_sq >= (2r)^2
    # viol = max(0, (2r)^2 - dists_sq)
    # This is not differentiable at the boundary.
    # Use a soft plus function or just squared error on the violation.
    
    target_sq = (2*r)**2
    viol_pair_sq = np.maximum(0, target_sq - dists_sq)
    penalty_pair = np.sum(viol_pair_sq**2)
    
    total_penalty = penalty_boundary + penalty_pair
    return total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Hexagonal Lattice
    # Generate a hex grid
    # Approximate number of points in a hex grid of radius R
    # Area = pi * R^2. Point area ~ sqrt(3)/2 * d^2.
    # N ~ pi * R^2 / (sqrt(3)/2 * d^2)
    # d ~ sqrt(2 * pi * R^2 / (N * sqrt(3)))
    
    # Let's just generate a grid and cut it.
    # Start with a spacing that gives roughly N points.
    # Radius of container = 1.
    # Area = pi.
    # Area per point = pi / 300.
    # Hex cell area = sqrt(3)/2 * d^2.
    # d^2 = 2 * pi / (300 * sqrt(3))
    d0 = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate points
    points = []
    # Loop over hex coordinates
    # q, r
    # x = d * (q + r/2)
    # y = d * r * sqrt(3)/2
    
    # Max q, r
    max_coord = int(np.ceil(1.0 / (d0 * 0.5))) + 2
    
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            x = d0 * (q + 0.5 * r)
            y = d0 * r * np.sqrt(3) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many, randomly select N or select closest to center
    if len(points) > N:
        # Select N points closest to center? Or random?
        # Closest to center might be better for symmetry.
        dists_to_center = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:N]]
    elif len(points) < N:
        # If too few, reduce spacing
        # This shouldn't happen with the estimate, but just in case
        while len(points) < N:
            d0 *= 0.98
            points = []
            max_coord = int(np.ceil(1.0 / (d0 * 0.5))) + 2
            for q in range(-max_coord, max_coord + 1):
                for r in range(-max_coord, max_coord + 1):
                    x = d0 * (q + 0.5 * r)
                    y = d0 * r * np.sqrt(3) / 2.0
                    if x*x + y*y <= 1.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) > N:
                dists_to_center = np.linalg.norm(points, axis=1)
                indices = np.argsort(dists_to_center)
                points = points[indices[:N]]
                break
            if time.time() > deadline:
                break

    if len(points) != N:
        # Fallback to simple grid if hex generation failed
        # This is a safety net
        points = np.random.rand(N, 2) * 2 - 1
        # Scale to fit
        norms = np.linalg.norm(points, axis=1)
        points = points / np.maximum(norms, 1e-6) * 0.9

    # 2. Optimization Loop
    # We will try to increase the radius r.
    # Start with the current minimum distance.
    current_r = _compute_score(points)
    best_points = points.copy()
    best_r = current_r
    
    # Iterative refinement
    # We'll use a "target_r" that we try to achieve.
    # If we fail, we reduce it.
    
    # Initial target slightly higher than current
    target_r = current_r * 1.01
    
    # Number of iterations
    max_iterations = 100
    current_iter = 0
    
    while current_iter < max_iterations and time.time() < deadline:
        # Define the objective for SLSQP
        # We want to minimize the penalty for not reaching target_r
        # And we want to keep points inside the unit circle.
        
        # Constraints for SLSQP:
        # 1. Boundary: ||p_i|| <= 1 - target_r
        # 2. Pairwise: ||p_i - p_j|| >= 2 * target_r
        
        # SLSQP handles inequalities.
        # Let's define the constraints.
        
        # This can be slow for N=300 because of N^2 constraints.
        # Alternative: Use a penalty function and minimize it.
        
        def penalty_func(pts):
            # Flatten points
            pts = pts.reshape(N, 2)
            
            # Boundary
            dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
            viol_boundary = np.maximum(0, target_r - dists_boundary)
            pen_b = np.sum(viol_boundary**2)
            
            # Pairwise
            # To avoid memory blowup, we can compute in blocks or just do it.
            # 300x300 is 90k floats, ~720KB. Fine.
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dists_sq, np.inf)
            
            target_sq = (2 * target_r)**2
            viol_pair_sq = np.maximum(0, target_sq - dists_sq)
            pen_p = np.sum(viol_pair_sq**2)
            
            return pen_b + pen_p
        
        # Use Nelder-Mead or Powell if gradients are hard?
        # SLSQP is good for constrained problems, but here we are using a penalty.
        # If we use a penalty, we can use BFGS or L-BFGS-B.
        # But we need to ensure points stay within the container roughly.
        # The penalty will push them back.
        
        # Let's use L-BFGS-B for speed.
        # Initial points are already valid for current_r, so hopefully close to valid for target_r.
        
        res = minimize(
            penalty_func, 
            points.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-8}
        )
        
        new_points = res.x.reshape(N, 2)
        
        # Check if the new points are valid (all finite, etc.)
        if not np.all(np.isfinite(new_points)):
            break
            
        # Calculate the actual score of the new points
        new_r = _compute_score(new_points)
        
        # If we improved, accept and increase target
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
            # Increase target slightly
            target_r = new_r * 1.005
            current_iter += 1
        else:
            # If we didn't improve, maybe target is too high.
            # Reduce target and retry? Or stop?
            # Let's reduce target a bit and try again a few times.
            target_r = best_r * 1.001
            if target_r <= best_r * 1.0001:
                # We are stuck
                break
            current_iter += 1
            
    # Final polish: try to maximize the score directly with a small step
    # Using the best points found so far.
    
    # One more optimization pass with the current best_r as target
    target_r = best_r * 1.001
    
    def final_penalty(pts):
        pts = pts.reshape(N, 2)
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        viol_boundary = np.maximum(0, target_r - dists_boundary)
        pen_b = np.sum(viol_boundary**2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        
        target_sq = (2 * target_r)**2
        viol_pair_sq = np.maximum(0, target_sq - dists_sq)
        pen_p = np.sum(viol_pair_sq**2)
        
        return pen_b + pen_p

    if time.time() < deadline:
        res = minimize(
            final_penalty, 
            best_points.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        if res.success:
            final_points = res.x.reshape(N, 2)
            final_r = _compute_score(final_points)
            if final_r > best_r:
                best_points = final_points
                best_r = final_r

    return best_points
# EVOLVE-BLOCK-END
