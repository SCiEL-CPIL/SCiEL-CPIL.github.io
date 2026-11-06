import numpy as np
import time

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0 or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Distance to walls
    wall_dist = np.min(np.minimum(0.5 - centers, centers + 0.5))
    
    if wall_dist <= 0:
        return 0.0

    # Distance to neighbors
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N):
    """Generate a hexagonal grid of points."""
    # Approximate spacing for hex packing in unit square
    # Area per point ~ 1/N. Hex area per point ~ d^2 * sqrt(3)/2.
    # d^2 ~ 2/(N*sqrt(3)) -> d ~ sqrt(2/(N*sqrt(3)))
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _gradient_ascent_optimize(centers, start_time, time_limit):
    """
    Global vectorized gradient ascent on a smooth objective.
    Objective: Maximize min(d_ij, 2*wall_dist)
    Approximation: Maximize sum of soft-min terms or penalize violations.
    We use a potential energy approach:
    Force = sum over neighbors of repulsion if dist < target + wall repulsion.
    But to maximize the MIN distance, we can use a soft-min trick or simply 
    optimize for a fixed target r and increase r if feasible.
    
    Better approach for this specific task (max min dist):
    Use a "soft" objective: L = - (softmin of distances)
    Or simpler: Iterative "inflation" with global relaxation.
    Let's use a gradient descent on a loss function that penalizes 
    distances being below a current target radius R.
    
    Loss L(R, C) = sum_{i<j} max(0, R - d_ij/2)^2 + sum_i max(0, R - wall_i)^2
    We want to find C such that L=0 for the largest R.
    
    Algorithm:
    1. Start with R = current_score * 0.9
    2. While time remains:
       a. Compute gradients of L w.r.t C
       b. Update C -= lr * grad
       c. If L is small (feasible), increase R slightly
       d. Else, decrease R or keep and continue optimizing C
    """
    centers = centers.copy()
    n = len(centers)
    
    # Initial R estimate
    current_score = _compute_score(centers)
    R = current_score * 0.95
    
    best_score = current_score
    best_centers = centers.copy()
    
    lr = 0.01
    r_step = 1e-4
    max_iters = 10000
    stall_count = 0
    last_R = R
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(0.5 - centers, centers + 0.5)
        
        # Current actual min radius
        actual_r = min(np.min(dist)/2.0, np.min(wall_dists))
        
        if actual_r > best_score:
            best_score = actual_r
            best_centers = centers.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > 50:
                lr *= 0.9
                if lr < 1e-6:
                    break
                stall_count = 0

        # We want to increase R if the current configuration can support a larger R.
        # Check feasibility: Is there any pair closer than R? Or wall closer than R?
        # Gradient of Loss L = sum max(0, R - d/2)^2 + sum max(0, R - w)^2
        
        # Pairwise gradient
        # For each pair (i,j) where dist < 2R, force pushes them apart.
        # d/dx_i (R - d/2)^2 = 2(R - d/2) * (-1/2) * (x_i - x_j)/d = -(R - d/2) * (x_i - x_j)/d
        # Force on i is -grad = (R - d/2) * (x_i - x_j)/d
        
        force = np.zeros_like(centers)
        
        # Vectorized pairwise force calculation
        # Only compute for pairs where dist < 2R to save time? 
        # For N=200, N^2=40000, which is small. We can do it all.
        
        # Mask for active pairs
        active_mask = dist < 2 * R
        np.fill_diagonal(active_mask, False)
        
        # We need to sum forces.
        # force_i = sum_j (R - d_ij/2) * (c_i - c_j) / d_ij  for d_ij < 2R
        # Note: The term (R - d_ij/2) is positive when d_ij < 2R.
        
        # Let's compute the scalar factor for each pair
        # factor_ij = (R - d_ij/2) / d_ij
        # This is 0 if d_ij >= 2R
        
        with np.errstate(divide='ignore', invalid='ignore'):
            factors = np.where(active_mask, (R - dist/2.0) / dist, 0.0)
            
        # force_i = sum_j factors[i,j] * (c_i - c_j)
        # = sum_j factors[i,j] * c_i - sum_j factors[i,j] * c_j
        # = c_i * sum_j factors[i,j] - sum_j (factors[i,j] * c_j)
        
        sum_factors = np.sum(factors, axis=1)
        # Weighted sum of other centers
        # factors[i,j] * c_j
        weighted_sum = np.sum(factors[:, :, np.newaxis] * centers[np.newaxis, :, :], axis=1)
        
        pair_force = centers * sum_factors[:, np.newaxis] - weighted_sum
        
        # Wall force
        # For each point i and coord k, if wall_dist < R, push away.
        # Wall dist = min(0.5 - x, x + 0.5)
        # If x < 0, wall is 0.5 - x. d/dx (R - (0.5-x))^2 = 2(R - 0.5 + x) * 1
        # If x > 0, wall is x + 0.5. d/dx (R - (x+0.5))^2 = 2(R - x - 0.5) * 1
        # Force = -grad
        
        wall_force = np.zeros_like(centers)
        
        # Left wall (x < 0)
        left_mask = centers[:, 0] < 0
        left_violation = R - (0.5 - centers[:, 0])
        active_left = left_mask & (left_violation > 0)
        wall_force[active_left, 0] += 2 * left_violation[active_left] # grad is 2*viol*1, force is -grad? 
        # Wait. L = (R - w)^2. dL/dx = 2(R-w) * dw/dx.
        # w = 0.5 - x. dw/dx = -1.
        # dL/dx = 2(R - (0.5-x)) * (-1) = -2(R - 0.5 + x).
        # Force = -dL/dx = 2(R - 0.5 + x).
        # So wall_force[active_left, 0] = 2 * (R - 0.5 + centers[active_left, 0])
        # My code above: 2 * left_violation = 2 * (R - 0.5 + x). Correct.
        
        # Right wall (x > 0)
        right_mask = centers[:, 0] >= 0
        right_violation = R - (centers[:, 0] + 0.5)
        active_right = right_mask & (right_violation > 0)
        # w = x + 0.5. dw/dx = 1.
        # dL/dx = 2(R - (x+0.5)) * 1.
        # Force = -2(R - x - 0.5) = 2(x + 0.5 - R).
        wall_force[active_right, 0] += 2 * (centers[active_right, 0] + 0.5 - R)
        
        # Bottom wall (y < 0)
        bottom_mask = centers[:, 1] < 0
        bottom_violation = R - (0.5 - centers[:, 1])
        active_bottom = bottom_mask & (bottom_violation > 0)
        wall_force[active_bottom, 1] += 2 * (R - 0.5 + centers[active_bottom, 1])
        
        # Top wall (y > 0)
        top_mask = centers[:, 1] >= 0
        top_violation = R - (centers[:, 1] + 0.5)
        active_top = top_mask & (top_violation > 0)
        wall_force[active_top, 1] += 2 * (centers[active_top, 1] + 0.5 - R)
        
        total_force = pair_force + wall_force
        
        # Update
        centers += lr * total_force
        
        # Project to square (soft projection or hard clip?)
        # Hard clip might introduce discontinuities in gradient, but ensures validity.
        # Let's clip to [-0.5, 0.5]
        centers = np.clip(centers, -0.5, 0.5)
        
        # Adaptive R
        # If the max violation is small, we can increase R.
        max_pair_violation = np.max(np.maximum(0, 2*R - dist))
        max_wall_violation = np.max(np.maximum(0, R - wall_dists))
        total_violation = max(max_pair_violation, max_wall_violation)
        
        if total_violation < 1e-5:
            R += r_step
            # Reset lr if we just increased R? Maybe.
        elif total_violation > 0.1 * R:
            # Too hard, decrease R or reduce lr
            R *= 0.99
            lr *= 0.9
            
        # Small random jitter to escape local minima
        if i % 50 == 0 and stall_count > 10:
            centers += np.random.normal(0, 0.005, centers.shape)
            centers = np.clip(centers, -0.5, 0.5)

    return best_centers, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    # 1. Generate diverse initializations
    inits = []
    
    # Hex Grid
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    # Square Grid
    square_grid = _generate_square_grid(N)
    inits.append(square_grid)
    
    # Jittered Hex
    if hex_grid is not None:
        np.random.seed(42)
        jittered_hex = hex_grid + np.random.normal(0, 0.01, hex_grid.shape)
        jittered_hex = np.clip(jittered_hex, -0.49, 0.49)
        inits.append(jittered_hex)
        
    # Random Initializations
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)
        
    # Sort initializations by their initial score (best first)
    scored_inits = []
    for init in inits:
        score = _compute_score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Run gradient ascent optimization on each initialization
    for score, init_centers in scored_inits:
        if time.time() - start_time > time_limit:
            break
            
        # Run gradient ascent
        centers, local_score = _gradient_ascent_optimize(init_centers, start_time, time_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_centers = centers
            
    # 3. If time remains, do random restarts from the best found so far
    if best_centers is not None and time.time() - start_time < time_limit - 10:
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            np.random.seed(int(time.time() * 1000) % 10000)
            # Perturb best solution
            perturbed = best_centers + np.random.normal(0, 0.02, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            centers, local_score = _gradient_ascent_optimize(perturbed, start_time, time_limit)
            
            if local_score > best_score:
                best_score = local_score
                best_centers = centers
                
    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
