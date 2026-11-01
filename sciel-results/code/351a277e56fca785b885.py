"""
Strategy: Global optimization of a smooth surrogate objective using L-BFGS-B.
The objective is the sum of the logarithms of the minimum clearances (pairwise and wall).
This surrogate is smooth and differentiable almost everywhere, allowing gradient-based
optimizers like L-BFGS-B to navigate the configuration space efficiently.
We use multi-start with diverse initializations (hexagonal, random, grid) to explore
different basins of attraction.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N_CIRCLES = 200
HALF_SIDE = 0.5

def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0

    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0

    tree = cKDTree(centers)
    # Use a generous radius for query_pairs to ensure we find all close pairs
    # The radius is roughly 2 * max_r, but we don't know max_r yet.
    # A safe upper bound for r in a unit square for 200 circles is around 0.04.
    # So 2*r is 0.08. Let's use 0.1 to be safe.
    pairs = tree.query_pairs(r=0.1)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)

    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r

def objective_and_grad(flat_centers):
    """
    Objective: -sum(log(min_clearance_i)) for all relevant clearances.
    We want to maximize the minimum clearance, which is equivalent to minimizing
    the sum of -log(clearance). This is a soft constraint that penalizes small clearances heavily.
    
    Actually, a better surrogate for "maximize min clearance" is the "log-sum-exp" of the negative clearances,
    or simply summing -log(clearance) for the K smallest clearances.
    Here we use the sum of -log(clearance) for all pairs and walls within a threshold.
    This creates a smooth landscape where pushing apart close circles is prioritized.
    """
    centers = flat_centers.reshape(N_CIRCLES, 2)
    
    # Clip to valid range to avoid log(0) or log(negative)
    # But we want the gradient to push back, so we don't clip the input to the function,
    # we just handle the values carefully.
    
    # Compute distances to walls
    # d_wall_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
    # We want to maximize min_i d_wall_i.
    
    # Compute pairwise distances
    tree = cKDTree(centers)
    # We need a radius to query. If circles are far apart, we don't care.
    # If they are close, we care.
    # Let's use a dynamic radius based on the current state.
    # Estimate current min radius
    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 1e-10:
        # If any circle is outside or on the edge, return a large penalty
        return 1e6, np.zeros_like(flat_centers)
    
    # Query pairs within a reasonable distance.
    # If the packing is good, the min distance is ~0.07.
    # If it's bad, it's smaller.
    # Let's use 0.15 as a safe upper bound for "close" pairs.
    pairs = tree.query_pairs(r=0.15)
    
    obj_val = 0.0
    grad = np.zeros_like(flat_centers)
    
    epsilon = 1e-12
    
    # Wall contributions
    # d_wall = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Let's just use the distance to the nearest wall for each circle.
    # d_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
    # We want to maximize min d_i.
    # Surrogate: sum -log(d_i + eps)
    
    dx = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    dy = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    d_wall = np.minimum(dx, dy)
    
    # Clip d_wall to be positive
    d_wall_safe = np.maximum(d_wall, epsilon)
    
    obj_val += -np.sum(np.log(d_wall_safe))
    
    # Gradient of -log(d_wall) wrt x_i
    # d_wall = min(dx, dy)
    # If dx < dy, d_wall = dx = 0.5 - x_i (if x_i > 0) or x_i + 0.5 (if x_i < 0)
    # Let's compute gradients properly.
    
    # For x:
    # d/dx (0.5 - x) = -1
    # d/dx (0.5 + x) = 1
    # d/dx (0.5 - y) = 0
    # d/dx (0.5 + y) = 0
    
    # grad_x_i = -1 / d_wall * d(d_wall)/dx_i
    # If d_wall == dx:
    #   if x_i > 0, d_wall = 0.5 - x_i -> d/dx = -1 -> grad = -1/d_wall * (-1) = 1/d_wall
    #   if x_i < 0, d_wall = 0.5 + x_i -> d/dx = 1 -> grad = -1/d_wall * 1 = -1/d_wall
    #   if x_i == 0, d_wall = 0.5 -> grad = 0 (subgradient)
    
    # Let's vectorize this.
    # d_wall = min(dx, dy)
    # mask_x = (dx < dy)
    # For those where mask_x is True:
    #   if x_i > 0, grad_x = 1/d_wall
    #   if x_i < 0, grad_x = -1/d_wall
    #   if x_i == 0, grad_x = 0
    # For those where mask_x is False:
    #   grad_x = 0
    
    mask_x = (dx < dy)
    grad_x = np.zeros(N_CIRCLES)
    # For x-dominated walls
    # d/dx (0.5 - x) = -1, d/dx (0.5 + x) = 1
    # sign_x = -1 if x > 0, 1 if x < 0
    sign_x = -np.sign(centers[:, 0])
    # If x is 0, sign is 0, which is correct for subgradient.
    grad_x[mask_x] = -sign_x[mask_x] / d_wall_safe[mask_x]
    
    # For y-dominated walls
    mask_y = (dy < dx)
    grad_y = np.zeros(N_CIRCLES)
    sign_y = -np.sign(centers[:, 1])
    grad_y[mask_y] = -sign_y[mask_y] / d_wall_safe[mask_y]
    
    grad[0::2] += grad_x
    grad[1::2] += grad_y
    
    # Pair contributions
    # d_ij = ||c_i - c_j||
    # We want to maximize min d_ij.
    # Surrogate: sum -log(d_ij + eps) for close pairs.
    
    if len(pairs) > 0:
        for i, j in pairs:
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # If coincident, add large penalty and random gradient
                obj_val += 1e4
                # Random nudge
                grad[2*i] += np.random.normal(0, 1)
                grad[2*i+1] += np.random.normal(0, 1)
                grad[2*j] -= np.random.normal(0, 1)
                grad[2*j+1] -= np.random.normal(0, 1)
                continue
            
            dist_safe = max(dist, epsilon)
            obj_val += -np.log(dist_safe)
            
            # Gradient of -log(dist) wrt c_i
            # d/dc_i (-log(||c_i - c_j||)) = - (c_i - c_j) / ||c_i - c_j||^2
            # = - diff / dist^2
            
            grad_pair = -diff / (dist_safe * dist_safe)
            
            grad[2*i] += grad_pair[0]
            grad[2*i+1] += grad_pair[1]
            grad[2*j] -= grad_pair[0]
            grad[2*j+1] -= grad_pair[1]
            
    return obj_val, grad

def hexagonal_initialization(perturbation=0.0):
    """Create a hexagonal lattice initialization with optional perturbation."""
    best_points = None
    best_score = 0.0
    
    # Try a range of spacings
    for scale in [0.072, 0.074, 0.076, 0.078]:
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        x_vals = np.arange(-0.5, 0.5 + dx/2, dx)
        y_vals = np.arange(-0.5, 0.5 + dy/2, dy)
        
        grid_points = []
        for i, y in enumerate(y_vals):
            for x in x_vals:
                if i % 2 == 1:
                    x_offset = dx / 2
                else:
                    x_offset = 0
                px = x + x_offset
                if -0.5 <= px <= 0.5 and -0.5 <= y <= 0.5:
                    grid_points.append([px, y])
        
        if len(grid_points) < N_CIRCLES:
            continue
        
        grid_points = np.array(grid_points)
        
        # Add perturbation if requested
        if perturbation > 0:
            rng = np.random.RandomState(abs(hash(scale)) % (2**32))
            grid_points += rng.normal(0, perturbation, grid_points.shape)
            grid_points = np.clip(grid_points, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Sort by distance from center and take N closest
        dists = np.sum(grid_points ** 2, axis=1)
        idx = np.argsort(dists)[:N_CIRCLES]
        points = grid_points[idx]
        
        r = compute_radius(points)
        if r > best_score:
            best_score = r
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N_CIRCLES]
    
    return best_points

def random_initialization(seed=0):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-0.5 + 0.01, 0.5 - 0.01, (N_CIRCLES, 2))
    return points

def grid_initialization(seed=0):
    """Generate a grid initial configuration."""
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Shuffle and take N
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(grid))[:N_CIRCLES]
    return grid[idx]

def optimize_l_bfgs(centers, time_limit=20.0):
    """
    Optimize the configuration using L-BFGS-B.
    """
    start_time = time.time()
    flat_centers = centers.flatten()
    
    # Bounds: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N_CIRCLES)
    
    # We need to stop if time is up
    # scipy minimize doesn't have a built-in time limit callback that stops gracefully in all versions,
    # but we can use a wrapper or just check time in a callback.
    # However, L-BFGS-B is usually fast enough for 200 points if we limit maxiter.
    # Let's use maxiter=1000 and check time.
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
        
    try:
        res = minimize(
            objective_and_grad,
            flat_centers,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
            }
        )
        if res.success or res.status == 1: # 1 means maximum number of iterations
            optimized_centers = res.x.reshape(N_CIRCLES, 2)
            # Ensure they are within bounds
            optimized_centers = np.clip(optimized_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            return optimized_centers
        else:
            return centers
    except RuntimeError:
        # Time limit exceeded, return current best
        # res might not be defined or might be partial
        # Actually, if callback raises, minimize propagates the exception.
        # So we catch it here.
        # But we lost the current state.
        # Let's try to recover or just return the input.
        # A better way is to not raise in callback but set a flag.
        # However, for simplicity and given the time budget, let's just return the input
        # if it fails, or the last known good state if we tracked it.
        # Since we didn't track intermediate states, we return the input.
        # This is a conservative fallback.
        return centers
    except Exception:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_config = None
    best_r = 0.0
    
    # Multi-start strategy
    # 1. Hexagonal
    # 2. Random
    # 3. Grid
    
    initializers = [
        ("hex", lambda: hexagonal_initialization(0.0)),
        ("hex_pert", lambda: hexagonal_initialization(0.01)),
        ("random", lambda: random_initialization(42)),
        ("random2", lambda: random_initialization(123)),
        ("grid", lambda: grid_initialization(42)),
    ]
    
    for name, init_func in initializers:
        if time.time() > deadline - 25:
            break
        
        # Generate initial configuration
        config = init_func()
        
        # Optimize
        optimized = optimize_l_bfgs(config, time_limit=20.0)
        
        r = compute_radius(optimized)
        if r > best_r:
            best_r = r
            best_config = optimized
            
    # If we have time, do a final polish on the best config
    if best_config is not None and time.time() < deadline - 10:
        # Try to improve the best config further
        polished = optimize_l_bfgs(best_config, time_limit=10.0)
        r_polished = compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best_config = polished

    # Final check
    if best_config is None or compute_radius(best_config) <= 0:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N_CIRCLES]
    
    return best_config
