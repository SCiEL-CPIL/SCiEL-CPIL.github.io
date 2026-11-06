"""Packing 200 circles using vectorized repulsive force gradient ascent."""
import numpy as np
import time

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0:
        return 0.0
    # Distance to walls
    wall_dist = np.min(np.abs(0.5 - np.abs(centers)))
    # Distance to neighbors
    # Vectorized pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N, d_scale=1.0):
    """Generate a hexagonal grid of points."""
    # Estimate d
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * d_scale
    
    points = []
    # Generate rows
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
        pass 
        
    return points[:N] if len(points) >= N else None

def _repulsive_optimize(centers, max_iter=5000, time_limit=None, start_time=None):
    """
    Optimize packing using a repulsive force model.
    Each point is pushed away from other points and pushed towards the walls.
    This is equivalent to maximizing the minimum distance via gradient ascent
    on a soft potential.
    """
    centers = centers.copy()
    n = len(centers)
    
    # Adaptive learning rate
    lr = 0.005
    best_r = _compute_score(centers)
    best_centers = centers.copy()
    
    for i in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
                
        # Compute pairwise differences
        # diff[i, j] = centers[j] - centers[i]
        diff = centers[np.newaxis, :, :] - centers[:, np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e-10) # Avoid division by zero
        
        # Inverse distance squared for force magnitude
        # Force is repulsive, so direction is -diff (push i away from j)
        # Magnitude is 1/dist^2
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Gradient of potential U = -sum(1/dist) is -sum( (c_j - c_i) / dist^3 )
        # We want to maximize min dist, so we minimize sum(1/dist^k) or similar.
        # Let's use a simpler force: F_i = sum_j (c_i - c_j) / ||c_i - c_j||^3
        # This is the gradient of the sum of inverse distances.
        
        # Vectorized calculation of forces
        # diff is (i, j, 2) where diff[i,j] = c_j - c_i
        # We want force on i: sum_j (c_i - c_j) / dist^3 = - sum_j diff[i,j] / dist^3
        # dist^3 = (dist_sq^1.5)
        
        # Normalize diff by dist^3
        # inv_dist_cubed = 1 / (dist_sq * sqrt(dist_sq))
        inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        # Force from other points: F_points[i] = - sum_j (inv_dist_cubed[i,j] * diff[i,j])
        # diff[i,j] is c_j - c_i. We want c_i - c_j. So we sum -diff.
        # Or simply sum diff and negate.
        # Let's compute sum_j (inv_dist_cubed[i,j] * diff[i,j])
        # This gives a vector for each i.
        weighted_diff = inv_dist_cubed[:, :, np.newaxis] * diff
        force_points = -np.sum(weighted_diff, axis=1)
        
        # Wall forces: Push towards walls if close? No, push away from walls if close.
        # Actually, we want to maximize distance to walls.
        # Force from wall: If x > 0, push left (negative x). If x < 0, push right (positive x).
        # This is equivalent to -sign(x) * wall_weight.
        # To make it smooth and strong near walls, we can use a potential like (0.5 - |x|)^2?
        # Gradient of (0.5 - |x|)^2 w.r.t x is -2(0.5-|x|)sign(x).
        # We want to maximize (0.5 - |x|), so we move in direction of gradient of distance.
        # Grad dist_x = sign(x) for x>0, -sign(x) for x<0? No.
        # dist = 0.5 - |x|. d(dist)/dx = -sign(x).
        # To maximize dist, move in direction -grad(dist) = sign(x).
        # So if x > 0, move positive? No, if x > 0, we are closer to right wall.
        # We want to move towards center? No, we want to move away from the closest wall.
        # If x > 0.4, closest wall is right (0.5). Distance is 0.5 - x.
        # To increase 0.5 - x, we need to decrease x. So force should be negative.
        # My previous logic: "Push towards walls" was wrong. We push AWAY from walls.
        
        # Let's use a simple repulsive force from the boundary.
        # Force_x = -sign(x) * (1.0 / (0.5 - |x| + eps)^2) ?
        # This becomes very strong near the wall.
        
        eps = 1e-6
        dist_to_wall_x = 0.5 - np.abs(centers[:, 0])
        dist_to_wall_y = 0.5 - np.abs(centers[:, 1])
        
        # Ensure positive distance for calculation
        dist_to_wall_x = np.maximum(dist_to_wall_x, eps)
        dist_to_wall_y = np.maximum(dist_to_wall_y, eps)
        
        # Force magnitude
        # We want to balance the inter-particle forces and wall forces.
        # Inter-particle forces are ~ 1/d^2.
        # Wall forces should be similar scale.
        wall_force_x = -np.sign(centers[:, 0]) / (dist_to_wall_x ** 2)
        wall_force_y = -np.sign(centers[:, 1]) / (dist_to_wall_y ** 2)
        
        # If a point is exactly at 0, sign is 0. Force is 0. That's fine.
        
        total_force_x = force_points[:, 0] + wall_force_x
        total_force_y = force_points[:, 1] + wall_force_y
        
        # Update positions
        centers[:, 0] += lr * total_force_x
        centers[:, 1] += lr * total_force_y
        
        # Clip to square (with small margin to avoid numerical issues on boundary)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Check score periodically to save time
        if i % 50 == 0:
            current_r = _compute_score(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
                
    return best_centers, best_r

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    # Initializations
    inits = []
    
    # 1. Hex Grid
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    # 2. Square Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 3. Jittered Grid
    np.random.seed(42)
    jittered = grid[:N].copy()
    jittered += np.random.normal(0, 0.01, jittered.shape)
    jittered = np.clip(jittered, -0.49, 0.49)
    inits.append(jittered)
    
    # 4. Random
    for _ in range(3):
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)
        
    for init_centers in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Run repulsive optimization
        # We give each init a chunk of the remaining time
        # Or just run until time is up for the best one?
        # Better: run a fixed number of steps or time slice for each, keep best.
        
        # Estimate time per iteration:
        # 200x200 matrix ops are fast. ~1ms per iter?
        # Let's allow ~20s per init.
        init_time_limit = min(20.0, time_limit - (time.time() - start_time))
        
        opt_centers, score = _repulsive_optimize(
            init_centers, 
            max_iter=10000, 
            time_limit=init_time_limit, 
            start_time=start_time
        )
        
        if score > best_score:
            best_score = score
            best_centers = opt_centers
            
        # If we have time left, we can do a second pass on the best found so far
        # with smaller learning rate for fine-tuning
        
    # Final fine-tuning pass if time permits
    if best_centers is not None and time.time() - start_time < time_limit - 5.0:
        remaining_time = time_limit - (time.time() - start_time) - 5.0
        # Use smaller lr for fine tuning
        # The _repulsive_optimize uses fixed lr. 
        # We can call it again on the best centers.
        # To change lr, we'd need to modify the function or pass it.
        # Let's just run it again. It will start with lr=0.005.
        # Maybe add a parameter for lr?
        # For simplicity, let's just rely on the first pass.
        # Or, we can implement a quick local search with smaller steps here.
        
        # Let's do a manual fine-tune loop with smaller steps
        fine_tune_time_limit = remaining_time
        fine_tune_start = time.time()
        
        centers = best_centers.copy()
        lr = 0.001
        best_ft_score = best_score
        best_ft_centers = centers.copy()
        
        while time.time() - fine_tune_start < fine_tune_time_limit:
            diff = centers[np.newaxis, :, :] - centers[:, np.newaxis, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1e-10)
            
            inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
            np.fill_diagonal(inv_dist_cubed, 0.0)
            
            weighted_diff = inv_dist_cubed[:, :, np.newaxis] * diff
            force_points = -np.sum(weighted_diff, axis=1)
            
            eps = 1e-6
            dist_to_wall_x = np.maximum(0.5 - np.abs(centers[:, 0]), eps)
            dist_to_wall_y = np.maximum(0.5 - np.abs(centers[:, 1]), eps)
            
            wall_force_x = -np.sign(centers[:, 0]) / (dist_to_wall_x ** 2)
            wall_force_y = -np.sign(centers[:, 1]) / (dist_to_wall_y ** 2)
            
            total_force_x = force_points[:, 0] + wall_force_x
            total_force_y = force_points[:, 1] + wall_force_y
            
            centers[:, 0] += lr * total_force_x
            centers[:, 1] += lr * total_force_y
            centers = np.clip(centers, -0.5, 0.5)
            
            score = _compute_score(centers)
            if score > best_ft_score:
                best_ft_score = score
                best_ft_centers = centers.copy()
                
        if best_ft_score > best_score:
            best_score = best_ft_score
            best_centers = best_ft_centers

    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(_compute_score(centers))
