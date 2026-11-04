"""
Optimized program for Heilbronn convex regions n=14.
Uses a parameterization of convex polygons by radii and angle steps,
combined with soft-min annealing and L-BFGS-B optimization.
"""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 14

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Number of restarts
    n_restarts = 10
    
    # Beta annealing schedule
    # Start with small beta (smooth) and increase to large beta (sharp)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def compute_points(r, d_theta):
        """
        r: array of radii (N,)
        d_theta: array of angle steps (N,)
        Returns: array of points (N, 2)
        """
        # Cumulative sum of angle steps gives absolute angles
        # We fix theta_0 = 0
        theta = np.zeros(N)
        theta[1:] = np.cumsum(d_theta[:-1])
        
        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def polygon_area(points):
        """
        Compute area of convex polygon given points in order.
        """
        x = points[:, 0]
        y = points[:, 1]
        # Shoelace formula
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return area
    
    def triangle_areas(points):
        """
        Compute areas of all triangles formed by triples of points.
        points: (N, 2)
        Returns: array of triangle areas
        """
        # points[i], points[j], points[k]
        pi = points[tri_indices[:, 0]]
        pj = points[tri_indices[:, 1]]
        pk = points[tri_indices[:, 2]]
        
        # Area = 0.5 * |cross(pj-pi, pk-pi)|
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return areas
    
    def objective(vars, beta):
        """
        Objective function to minimize: -score
        score = min_tri_area / hull_area
        We use soft-min for min_tri_area.
        """
        r = vars[:N]
        d_theta = vars[N:]
        
        # Ensure positivity (bounds handle this, but just in case)
        r = np.maximum(r, 1e-6)
        d_theta = np.maximum(d_theta, 1e-6)
        
        points = compute_points(r, d_theta)
        
        # Check if points are valid (non-degenerate)
        # If area is too small, return high cost
        area = polygon_area(points)
        if area < 1e-6:
            return 1e6
            
        tri_areas = triangle_areas(points)
        
        # Soft-min approximation
        # min(x) ~ -1/beta * log(sum(exp(-beta*x)))
        # To avoid overflow, use log-sum-exp trick
        min_val = np.min(tri_areas)
        if min_val > 1e6:
             return 1e6
             
        # scaled = -beta * tri_areas
        # max_scaled = -beta * min_val
        # log_sum_exp = max_scaled + log(sum(exp(scaled - max_scaled)))
        # soft_min = -1/beta * log_sum_exp
        
        # Simpler stable soft min:
        # m = min(tri_areas)
        # soft_min = m - 1/beta * log(sum(exp(-beta*(tri_areas - m))))
        
        scaled = -beta * (tri_areas - min_val)
        # Clip to avoid overflow in exp
        scaled = np.clip(scaled, -500, 0)
        
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_min = min_val - (1.0 / beta) * log_sum
        
        score = soft_min / area
        return -score
    
    # Initial guesses
    # 1. Regular polygon
    # 2. Random convex polygons
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        if restart == 0:
            # Regular polygon
            r_init = np.ones(N)
            d_theta_init = np.full(N, 2 * np.pi / N)
        else:
            # Random radii and angles
            r_init = np.random.uniform(0.5, 1.5, N)
            # Random angle steps that sum to 2*pi
            # Generate N-1 random positive numbers, normalize
            raw_angles = np.random.uniform(0.5, 1.5, N-1)
            raw_angles /= np.sum(raw_angles) * (2 * np.pi / N) # Scale roughly to average
            d_theta_init = np.zeros(N)
            d_theta_init[:-1] = raw_angles
            # Last angle is determined by 2*pi - sum(others)
            # But we need it to be positive.
            # Let's just generate N positive numbers and normalize to 2*pi
            raw_angles = np.random.uniform(0.5, 1.5, N)
            d_theta_init = 2 * np.pi * raw_angles / np.sum(raw_angles)
            
        x0 = np.concatenate([r_init, d_theta_init])
        
        # Bounds
        # r in [0.1, 5.0]
        # d_theta in [0.1, 2.5] (must be less than pi to avoid self-intersection? 
        # Actually, for a convex polygon, the sum of exterior angles is 2*pi.
        # If we define d_theta as the angle step, the interior angle is pi - d_theta (if d_theta is exterior?).
        # Let's check: If points are on a circle, d_theta = 2*pi/N.
        # The vector from origin to point i is r_i (cos theta_i, sin theta_i).
        # The angle between consecutive vectors is d_theta.
        # For convexity, the points must wrap around the origin once.
        # So sum(d_theta) = 2*pi.
        # We optimize d_theta directly.
        # Bounds for d_theta: [0.1, 2.0] is safe.
        
        bounds = [(0.1, 5.0)] * N + [(0.1, 2.0)] * N
        
        # Annealing
        current_x = x0
        for beta in betas:
            if time.time() > deadline:
                break
                
            res = minimize(
                objective, 
                current_x, 
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res.success:
                current_x = res.x
                # Evaluate true score (not soft-min) to compare
                r = current_x[:N]
                d_theta = current_x[N:]
                points = compute_points(r, d_theta)
                area = polygon_area(points)
                tri_areas = triangle_areas(points)
                min_area = np.min(tri_areas)
                score = min_area / area
                
                if score > best_score:
                    best_score = score
                    best_points = points
                    
    # If no points found, return random
    if best_points is None:
        best_points = np.random.rand(N, 2)
        
    # Normalize points? 
    # The score is scale invariant, so normalization doesn't change the score.
    # However, the task says "normalized to unit area". 
    # "The notebook normalizes by the point set convex hull area."
    # "score is the minimum triangle area divided by the convex hull area"
    # So we just return the points. The evaluator computes the score.
    
    return best_points

# EVOLVE-BLOCK-END
