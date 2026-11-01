"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 115.0  # Safety margin before 120s limit

def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and container margin."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Container constraint: 1 - |c_i|
    dists_origin = np.linalg.norm(points, axis=1)
    container_margin = 1.0 - np.max(dists_origin)
    if container_margin < 0:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    return min(min_pair_dist, container_margin)

def _violation_and_grad(x, r):
    """
    Objective: Sum of squared violations of constraints for a fixed radius r.
    Constraints:
    1. ||p_i|| + r <= 1  =>  1 - r - ||p_i|| >= 0
    2. ||p_i - p_j|| >= 2r =>  ||p_i - p_j|| - 2r >= 0
    
    We minimize V(x) = sum (max(0, -g_i(x))^2)
    This is smooth (C1) except at the kink, but L-BFGS-B handles it well in practice.
    """
    points = x.reshape(N, 2)
    
    # Container violations
    dists_origin = np.linalg.norm(points, axis=1)
    container_violations = np.maximum(0, r + dists_origin - 1.0)
    
    # Pairwise violations
    # To save memory and time, we only compute upper triangle
    # But for gradient, we need to be careful.
    # Let's use the full matrix for simplicity in gradient calculation if N is small enough,
    # but N=50 means 2500 pairs. Full matrix is 50x50=2500. It's fine.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Violation for pair (i,j) is max(0, 2r - dist_ij)
    # Since dist is symmetric, we count each pair once in the sum?
    # Standard practice: sum over i < j.
    # Let's create a mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dists[mask]
    pair_violations = np.maximum(0, 2.0 * r - pair_dists)
    
    # Total objective
    obj = np.sum(container_violations**2) + np.sum(pair_violations**2)
    
    # Gradient
    grad = np.zeros_like(points)
    
    # Container gradient
    # d/dp_i [ max(0, r + ||p_i|| - 1)^2 ]
    # If violation > 0: 2 * (r + ||p_i|| - 1) * (p_i / ||p_i||)
    for i in range(N):
        if container_violations[i] > 0:
            r_i = dists_origin[i]
            if r_i > 1e-12:
                grad[i] += 2.0 * container_violations[i] * (points[i] / r_i)
    
    # Pairwise gradient
    # For pair (i, j) with violation v > 0:
    # Term is v^2 = (2r - ||p_i - p_j||)^2
    # d/dp_i = 2 * (2r - d) * (- (p_i - p_j) / d)
    # d/dp_j = 2 * (2r - d) * ( (p_i - p_j) / d )
    # Note: The sign is tricky. Let's re-derive.
    # f = (2r - d)^2
    # df/dp_i = 2(2r - d) * (-1) * d(d)/dp_i = -2(2r - d) * (p_i - p_j)/d
    # df/dp_j = 2(2r - d) * (-1) * d(d)/dp_j = -2(2r - d) * (p_j - p_i)/d = 2(2r - d) * (p_i - p_j)/d
    
    # We need to map the flattened pair_violations back to (i, j)
    # This is slow if done in a loop. Let's vectorize or use a smarter approach.
    # Given N=50, a loop over ~1225 pairs is acceptable if not done too frequently.
    # However, L-BFGS-B calls this many times.
    # Let's try to vectorize the gradient contribution.
    
    # Alternative: Compute full gradient matrix G where G[i,j] is grad contribution from pair (i,j) to p_i.
    # This is memory heavy but N=50 is small.
    
    # Let's stick to the loop for clarity and correctness, it's only ~1225 iterations.
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if pair_violations[idx] > 0:
                d_vec = points[i] - points[j]
                d = dists[i, j]
                if d > 1e-12:
                    factor = -2.0 * pair_violations[idx] / d
                    grad[i] += factor * d_vec
                    grad[j] -= factor * d_vec
            idx += 1
            
    return obj, grad.ravel()

def _hex_init(radius_scale=1.0):
    """Hexagonal lattice initialization."""
    points = []
    # Generate a hexagonal grid
    # Spacing s. We want to fit as many as possible.
    # For 50 points, a rough hex grid.
    # Let's generate a larger grid and pick the first 50 closest to origin?
    # Or just a fixed pattern.
    
    # Standard hex packing:
    # Rows offset.
    # Let's try to generate a grid that covers the unit disk.
    # Approximate radius of hex cell is s.
    # Let's try s = 0.2
    s = 0.2
    dx = s
    dy = s * np.sqrt(3) / 2
    
    # Generate points in a bounding box
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    
    x = x_min
    row = 0
    while x <= x_max:
        y = y_min
        offset = (row % 2) * dx / 2
        while y <= y_max:
            px = x + offset
            py = y
            if px**2 + py**2 <= 0.8**2: # Keep well inside for init
                points.append([px, py])
            y += dy
        x += dx
        row += 1
        
    # If we have too many, pick closest to origin? Or just first 50?
    # Sorting by distance to origin ensures a compact cluster.
    points = np.array(points)
    if len(points) < N:
        # Pad with random
        while len(points) < N:
            points = np.vstack([points, np.random.uniform(-0.8, 0.8, 2)])
    else:
        # Sort by distance to origin
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        points = points[sorted_idx[:N]]
        
    return points

def _ring_init():
    """Ring-based initialization."""
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]
    ring_sizes = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.2, 0.4, 0.6, 0.8]
    idx = 1
    for ring_size, radius in zip(ring_sizes[1:], ring_radii[1:]):
        angles = np.linspace(0, 2*np.pi, ring_size, endpoint=False)
        for a in angles:
            points[idx] = [radius * np.cos(a), radius * np.sin(a)]
            idx += 1
    return points

def _random_init():
    """Random initialization inside unit disk."""
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _feasibility_search(r, initial_points, time_limit=5.0):
    """
    Try to find a feasible configuration for radius r starting from initial_points.
    Returns (success, points).
    """
    start_time = time.time()
    
    # L-BFGS-B on violation function
    def obj_and_grad(x):
        return _violation_and_grad(x, r)
    
    # Bounds: points must be in [-1, 1] x [-1, 1] (loose, constraint handles disk)
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(obj_and_grad, initial_points.ravel(), method='L-BFGS-B', 
                       jac=True, bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        # Check if feasible
        obj, _ = _violation_and_grad(res.x, r)
        if obj < 1e-8:
            return True, res.x.reshape(N, 2)
        else:
            return False, res.x.reshape(N, 2)
            
    except Exception:
        return False, initial_points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Binary search parameters
    # Lower bound: definitely feasible
    r_low = 0.05
    # Upper bound: definitely infeasible
    r_high = 0.2
    
    # Initializations to try
    inits = [
        _hex_init(),
        _ring_init(),
        _random_init()
    ]
    
    best_points = inits[0].copy()
    best_score = _compute_score(best_points)
    
    # Binary search
    # We want to find the max r such that a feasible solution exists.
    # We'll run a few iterations of binary search.
    # Each iteration tries to find a feasible solution for r_mid.
    
    n_binary_steps = 15
    
    for step in range(n_binary_steps):
        if time.time() > deadline - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        success = False
        # Try each initialization for this r_mid
        # To save time, we can rotate inits or just use the best one from previous step?
        # Let's use all inits but with short time limits.
        
        for init in inits:
            if time.time() > deadline - 10:
                break
                
            # Limit time per optimization call
            # L-BFGS-B doesn't have a direct time limit, but we can monitor iterations.
            # Or we can just run it and hope it's fast.
            # For safety, let's run it.
            
            is_feasible, points = _feasibility_search(r_mid, init)
            
            if is_feasible:
                success = True
                # Update best points if this configuration is better (larger r or better score)
                # Actually, if feasible for r_mid, the score is at least r_mid.
                # We should update best_points to this feasible solution.
                score = _compute_score(points)
                if score > best_score:
                    best_points = points.copy()
                    best_score = score
                
                # If feasible, try larger r
                r_low = r_mid
                break # Move to next binary step
                
        if not success:
            # Not feasible, try smaller r
            r_high = r_mid
            
    # Final polish with Nelder-Mead on the exact score function
    if time.time() < deadline - 5:
        def neg_score(x):
            pts = x.reshape(N, 2)
            return -_compute_score(pts)
            
        # Use L-BFGS-B result as start for Nelder-Mead? 
        # Nelder-Mead is slow in high dimensions (100D). 
        # It's better to use the L-BFGS-B result directly if it's good.
        # Or do a local search on the score.
        
        # Let's try a few random perturbations and local search
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            # Perturb best points
            perturbed = best_points + np.random.normal(0, 0.01, (N, 2))
            # Project to disk
            dists = np.linalg.norm(perturbed, axis=1)
            mask = dists > 0.99
            if np.any(mask):
                perturbed[mask] *= 0.99 / dists[mask]
                
            # Optimize
            res = minimize(neg_score, perturbed.ravel(), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            score = _compute_score(res.x.reshape(N, 2))
            if score > best_score:
                best_points = res.x.reshape(N, 2).copy()
                best_score = score

    return best_points
# EVOLVE-BLOCK-END
