import numpy as np
import time

N = 130
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s

def construct():
    # 1. Initial Configuration: Fibonacci Spiral
    # This is the same as the seed, which we aim to improve.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Generate Test Points (Y)
    # Use a much denser Fibonacci spiral to approximate the integral over the sphere.
    # M=5000 is a good balance between accuracy and speed.
    M = 5000
    j = np.arange(M) + 0.5
    z_y = 1.0 - 2.0 * j / M
    phi_y = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_y = np.sqrt(1.0 - z_y * z_y)
    test_points = np.column_stack([r_y * np.cos(phi_y), r_y * np.sin(phi_y), z_y])
    
    # 3. Optimization Loop
    # We minimize a smooth surrogate of the max distance.
    # F = sum_{j,i} w_{ji} * dist^2(p_i, y_j)
    # where w_{ji} = exp(-beta * dist^2) / sum_k exp(-beta * dist^2)
    
    # Initial parameters
    # beta controls the "softness" of the min. 
    # Large beta -> closer to hard min. Small beta -> averages over all points.
    # We want to fix the "holes" (large distances). 
    # Using a moderate beta helps identify the worst points.
    beta = 50.0 
    eta = 0.01 # Learning rate
    max_iter = 500
    
    # Precompute test point norms (they are unit vectors)
    # We need to compute distances D (N x M)
    
    best_points = points.copy()
    best_score = np.inf
    
    # Helper to compute covering radius (approx) for tracking
    def get_score(pts):
        # Max distance to nearest point over test points
        # dist^2 = 2 - 2 * sum(pts * test_points, axis=1)
        dots = np.dot(pts, test_points.T) # N x M
        dots = np.clip(dots, -1.0, 1.0)
        dist2 = 2.0 - 2.0 * dots
        nearest_dist2 = np.min(dist2, axis=0)
        return np.sqrt(np.max(nearest_dist2))
    
    # Track best
    best_score = get_score(points)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Compute dots (N x M)
        dots = np.dot(points, test_points.T)
        dots = np.clip(dots, -1.0, 1.0)
        dist2 = 2.0 - 2.0 * dots
        
        # Compute weights for each test point j
        # We want weights that emphasize larger distances?
        # Standard Softmin: w = exp(-beta * dist2) / sum(exp(-beta * dist2))
        # This gives high weight to the *nearest* points. 
        # If we want to minimize the *maximum*, we want to move points that are *far* from some y.
        # The gradient of the "soft max" of distances is what we want.
        # However, the "soft min" gradient pulls points towards y if they are the "soft min".
        # If we use Softmin, points get pulled towards the cluster of test points they are closest to.
        # This reduces local density? No, it reduces the distance to the *nearest* test point.
        # But we want to reduce the *maximum* distance to any test point.
        # The "worst" points are those with large min-dist.
        # The gradient of max_j min_i d(y_j, p_i) is complicated.
        
        # Let's stick to the "repulsion from worst cases" idea or the "soft min" idea.
        # If we use soft min (weights on nearest), the objective is sum of weighted squared distances to nearest.
        # Minimizing this reduces the average distance to the nearest point for "well-covered" areas?
        # Actually, if a point y is far from all p_i, the distances are all large.
        # exp(-beta * large) is small. The weights will be roughly uniform if all are large.
        # Then the gradient is sum of (p_i - y) weighted uniformly.
        # This moves the centroid of p_i towards y. This is good!
        # If y is close to p_k, weight on k is high, others low.
        # Gradient is dominated by (p_k - y). This moves p_k towards y.
        # So this method effectively performs a "soft k-means" where centers move towards points.
        # This should reduce the covering radius.
        
        # Calculate weights
        # Shift by max for stability, but beta*dist2 is positive, so exp(-beta*dist2) is in (0,1].
        # No overflow risk. Underflow might happen for large beta/dist, but that's fine (weight 0).
        exp_vals = np.exp(-beta * dist2) # N x M
        weights = exp_vals / np.sum(exp_vals, axis=0, keepdims=True) # N x M
        
        # Gradient for each point i: sum_j weights[j,i] * 2 * (p_i - y_j)
        # grad[i] = 2 * sum_j weights[j,i] * (p_i - y_j)
        # = 2 * (sum_j weights[j,i] * p_i - sum_j weights[j,i] * y_j)
        # = 2 * (p_i * sum_j weights[j,i] - sum_j weights[j,i] * y_j)
        
        # sum_weights is 1 for each j, but here we sum over j for fixed i.
        # sum_j weights[j,i] is the total weight assigned to point i.
        w_sum = np.sum(weights, axis=1) # N
        # weighted_sum_y = sum_j weights[j,i] * y_j
        # This is weights.T (M x N) * test_points (M x 3) -> N x 3
        weighted_sum_y = np.dot(weights.T, test_points)
        
        grads = 2.0 * (points * w_sum[:, np.newaxis] - weighted_sum_y)
        
        # Update
        points = points - eta * grads
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Evaluate
        curr_score = get_score(points)
        if curr_score < best_score:
            best_score = curr_score
            best_points = points.copy()
            
        # Optional: Adaptive step size or restart?
        # If we get stuck, we might increase beta to make it "harder" or decrease eta.
        # Let's decay eta slightly to converge.
        eta *= 0.995
        
        # If beta is too low, it acts like a mean. If too high, it acts like hard min.
        # Increasing beta over time can help lock in improvements.
        # beta += 0.1 ?
        # Let's keep beta constant for stability.
        
    return best_points

# Run once to verify? No, construct is called by evaluator.
# We just define it.
