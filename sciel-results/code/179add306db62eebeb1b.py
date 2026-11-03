import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # Estimate spacing
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # Packing density ~ 0.9.
    # N * pi * r^2 ~ 0.9 * pi  => r ~ sqrt(0.9/N)
    # Spacing ~ 2r
    r_est = np.sqrt(0.9 / N)
    spacing = 2 * r_est * 1.1  # Slightly larger to ensure fit
    
    points = []
    # Generate lattice
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        # Fallback to random if lattice is too sparse (unlikely for N=300)
        theta = np.random.uniform(0, 2*np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
    else:
        # Sort by distance to origin to get a compact cluster
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)
        points = points[indices[:N]]
        
    # 2. Optimization
    # We will use a gradient ascent on the smooth surrogate of the min-radius.
    # Objective: J(c) = - 1/beta * log( sum exp(-beta * x_k) )
    # where x_k are the constraints (distances/2 and boundary distances).
    
    beta = 50.0  # Initial sharpness
    step_size = 0.01
    
    # Pre-compute indices for pairs
    # For N=300, N*(N-1)/2 = 44850 pairs. This is manageable.
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    current_best_radius = 0.0
    best_points = points.copy()
    
    # Annealing schedule
    # We will run for a certain number of iterations or until time limit.
    # Let's say max 5000 iterations.
    max_iter = 5000
    iter_count = 0
    
    while iter_count < max_iter:
        if time.time() > deadline:
            break
            
        # Compute distances and boundary distances
        # Vectorized computation for speed
        
        # Pairwise distances
        # diff[i, j] = c_i - c_j
        # We only need upper triangle
        # c[i_indices] - c[j_indices]
        diff = points[i_indices] - points[j_indices]
        dists = np.linalg.norm(diff, axis=1)
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - norms
        
        # Constraints x
        x_pairs = dists / 2.0
        x_bounds = bound_dists
        
        # Combine all constraints
        x_all = np.concatenate([x_pairs, x_bounds])
        
        # Check for invalid states (negative distances or outside)
        # If any x is negative, the penalty is huge.
        
        # Compute weights w = exp(-beta * x) / sum(exp(-beta * x))
        # To avoid overflow/underflow, subtract max x? No, we want to emphasize small x (large negative exponent).
        # Actually, if x is small, -beta*x is large positive. exp(large) is large.
        # So we should subtract min x to keep exponents manageable?
        # Let m = min(x_all).
        # exp(-beta * (x - m)) = exp(-beta*x + beta*m).
        # This scales all weights by exp(beta*m), which cancels in the sum.
        m = np.min(x_all)
        exponents = -beta * (x_all - m)
        
        # Clip exponents to prevent overflow
        exponents = np.clip(exponents, -500, 500)
        
        weights_unnorm = np.exp(exponents)
        sum_weights = np.sum(weights_unnorm)
        
        if sum_weights == 0:
            break
            
        weights = weights_unnorm / sum_weights
        
        # Objective value (approximate min)
        # J = -1/beta * log(sum_weights) - m
        # Actually, the true min is approximated by m + 1/beta * log(sum(exp(beta*(x-m))))?
        # The log-sum-exp of y is approx max(y).
        # Here we have log-sum-exp of -beta*x.
        # Let's just track the actual min for scoring.
        current_radius = np.min(x_all)
        
        if current_radius > current_best_radius:
            current_best_radius = current_radius
            best_points = points.copy()
            
        # Compute Gradient
        # grad J = sum_k w_k * grad x_k
        # grad x_pair (for pair i,j) with respect to c_i is (c_i - c_j) / (2 * dist)
        # with respect to c_j is (c_j - c_i) / (2 * dist)
        
        grad = np.zeros_like(points)
        
        # Pair contributions
        # dists > 0
        valid_pairs = dists > 1e-9
        if np.any(valid_pairs):
            # Normalize diff
            # diff is c_i - c_j
            # grad for i is diff / (2 * dist)
            # grad for j is -diff / (2 * dist)
            
            factors = weights[:len(x_pairs)] / (2.0 * dists)
            factors[~valid_pairs] = 0.0
            
            # Accumulate gradients
            # For each pair (i, j), add factor * (c_i - c_j) to grad[i]
            # and subtract factor * (c_i - c_j) from grad[j]
            
            # Use np.add.at for accumulation
            np.add.at(grad, i_indices[valid_pairs], factors[valid_pairs, None] * diff[valid_pairs])
            np.add.at(grad, j_indices[valid_pairs], -factors[valid_pairs, None] * diff[valid_pairs])
            
        # Boundary contributions
        # grad x_bound for c_i is -c_i / |c_i|
        valid_bounds = norms > 1e-9
        if np.any(valid_bounds):
            bound_factors = weights[len(x_pairs):]
            # grad is -c_i / norm_i
            # Only for active bounds
            grad_bounds = -points[valid_bounds] / norms[valid_bounds, None]
            # Weighted sum
            grad[valid_bounds] += bound_factors[valid_bounds, None] * grad_bounds
            
        # Update points
        # Add gradient step
        points += step_size * grad
        
        # Optional: Project back to unit circle if they go far outside
        # This helps stability
        new_norms = np.linalg.norm(points, axis=1)
        out_of_bounds = new_norms > 1.0
        if np.any(out_of_bounds):
            points[out_of_bounds] /= new_norms[out_of_bounds, None]
            
        # Annealing
        iter_count += 1
        # Reduce step size and increase beta
        if iter_count % 100 == 0:
            step_size *= 0.95
            beta *= 1.05
            
    # Final polish
    # Run a few more iterations with very small step size and high beta
    # to tighten the packing.
    beta = 200.0
    step_size = 0.001
    for _ in range(200):
        if time.time() > deadline:
            break
            
        diff = points[i_indices] - points[j_indices]
        dists = np.linalg.norm(diff, axis=1)
        norms = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - norms
        
        x_pairs = dists / 2.0
        x_bounds = bound_dists
        x_all = np.concatenate([x_pairs, x_bounds])
        
        m = np.min(x_all)
        exponents = -beta * (x_all - m)
        exponents = np.clip(exponents, -500, 500)
        weights_unnorm = np.exp(exponents)
        sum_weights = np.sum(weights_unnorm)
        
        if sum_weights == 0:
            break
            
        weights = weights_unnorm / sum_weights
        
        current_radius = np.min(x_all)
        if current_radius > current_best_radius:
            current_best_radius = current_radius
            best_points = points.copy()
            
        grad = np.zeros_like(points)
        valid_pairs = dists > 1e-9
        if np.any(valid_pairs):
            factors = weights[:len(x_pairs)] / (2.0 * dists)
            factors[~valid_pairs] = 0.0
            np.add.at(grad, i_indices[valid_pairs], factors[valid_pairs, None] * diff[valid_pairs])
            np.add.at(grad, j_indices[valid_pairs], -factors[valid_pairs, None] * diff[valid_pairs])
            
        valid_bounds = norms > 1e-9
        if np.any(valid_bounds):
            bound_factors = weights[len(x_pairs):]
            grad_bounds = -points[valid_bounds] / norms[valid_bounds, None]
            grad[valid_bounds] += bound_factors[valid_bounds, None] * grad_bounds
            
        points += step_size * grad
        
        new_norms = np.linalg.norm(points, axis=1)
        out_of_bounds = new_norms > 1.0
        if np.any(out_of_bounds):
            points[out_of_bounds] /= new_norms[out_of_bounds, None]
            
    return best_points
