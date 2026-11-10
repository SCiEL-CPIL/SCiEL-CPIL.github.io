"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _hexagonal_initial():
    """Create a hexagonal grid arrangement of N points in [-0.5, 0.5]^2."""
    # Hexagonal packing: rows offset by half the horizontal spacing
    # For N=50, use roughly 7x7 grid with hexagonal offset
    rows = 7
    cols = 7
    # Horizontal spacing
    dx = 1.0 / cols
    # Vertical spacing (hexagonal: dy = dx * sqrt(3)/2)
    dy = dx * np.sqrt(3) / 2
    
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = -0.5 + (j + 0.5) * dx
            y = -0.5 + (i + 0.5) * dy
            # Offset every other row by half dx
            if i % 2 == 1:
                x += dx / 2
            # Keep within bounds
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts[:N], dtype=np.float64)
    
    # If we don't have enough points, add random points
    if len(pts) < N:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    return pts[:N]

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Min distance between any pair
    from scipy.spatial.distance import cdist
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Min distance to boundary
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    min_boundary = np.min(np.minimum(dx, dy))
    return min(min_dist / 2.0, min_boundary)

def _smooth_objective(x, beta=100.0):
    """Smooth objective using soft-min for better gradient flow."""
    pts = x.reshape(-1, 2)
    
    # Distance to boundaries
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(dx, dy)
    
    # Pairwise distances
    dist = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Soft-min of pairwise distances (approximate min with log-sum-exp)
    # soft_min = -log(sum(exp(-d/beta))) / beta
    # For numerical stability, shift by min
    min_d = np.min(dist)
    soft_min_pair = min_d - np.log(np.sum(np.exp(-(dist - min_d) / beta))) / beta
    
    # Soft-min of boundary distances
    min_b = np.min(boundary_dists)
    soft_min_boundary = min_b - np.log(np.sum(np.exp(-(boundary_dists - min_b) / beta))) / beta
    
    # Combined soft-min radius
    r_soft = min(soft_min_pair / 2.0, soft_min_boundary)
    
    # Penalty for violations
    boundary_violation = np.maximum(0, -boundary_dists).sum()
    overlap_penalty = np.maximum(0, 2.0 * r_soft - soft_min_pair).sum()
    
    penalty = boundary_violation + overlap_penalty
    
    return -r_soft + penalty * 10.0

def _simulated_annealing(pts, time_limit=100.0, rng=None):
    """Simulated annealing to improve circle packing."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    start_time = time.time()
    current_pts = pts.copy()
    current_radius = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    max_iters = 50000
    iter_per_sec = 5000  # Target iterations per second
    
    # Estimate how many iterations we can do
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    max_iters = int(min(max_iters, remaining_time * iter_per_sec))
    
    for it in range(max_iters):
        # Check time limit
        if it % 1000 == 0 and time.time() - start_time > time_limit - 5:
            break
        
        # Temperature
        progress = it / max_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a random move
        move_type = rng.random()
        if move_type < 0.7:
            # Move one point
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] += delta
            new_pts = np.clip(new_pts, -0.5, 0.5)
        elif move_type < 0.9:
            # Move two points
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            delta1 = rng.normal(0, T, size=2)
            delta2 = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx1] += delta1
            new_pts[idx2] += delta2
            new_pts = np.clip(new_pts, -0.5, 0.5)
        else:
            # Swap positions of two points (with small perturbation)
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            new_pts = current_pts.copy()
            new_pts[idx1], new_pts[idx2] = new_pts[idx2], new_pts[idx1]
            new_pts[idx1] += rng.normal(0, T * 0.5, size=2)
            new_pts[idx2] += rng.normal(0, T * 0.5, size=2)
            new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_radius = _compute_radius(new_pts)
        delta_r = new_radius - current_radius
        
        # Accept if better, or with probability exp(delta_r / T) if worse
        if delta_r >= 0 or rng.random() < np.exp(delta_r / T):
            current_pts = new_pts
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
    
    return best_pts

def _gradient_refine(pts, time_limit=30.0):
    """Gradient-based refinement using smooth objective."""
    start_time = time.time()
    x0 = pts.flatten().astype(np.float64)
    best_x = x0.copy()
    best_val = _smooth_objective(x0)
    
    # Use L-BFGS-B for constrained optimization
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    maxiter = 500
    while time.time() - start_time < time_limit:
        res = minimize(_smooth_objective, best_x, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
        else:
            break
    
    return best_x.reshape(-1, 2)

def construct():
    """Construct centers of N equal circles inside the square [-1/2, 1/2]^2."""
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Start with hexagonal grid
    pts = _hexagonal_initial()
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated annealing (main optimization)
    remaining = time_limit - (time.time() - start_time) - 30  # Leave 30s for refinement
    if remaining > 10:
        try:
            improved_pts = _simulated_annealing(pts, time_limit=remaining, rng=rng)
            improved_radius = _compute_radius(improved_pts)
            if improved_radius > best_radius:
                best_radius = improved_radius
                best_pts = improved_pts.copy()
        except Exception as e:
            pass
    
    # Phase 2: Gradient refinement
    remaining = time_limit - (time.time() - start_time) - 5
    if remaining > 10:
        try:
            refined_pts = _gradient_refine(best_pts, time_limit=remaining)
            refined_pts = np.clip(refined_pts, -0.5, 0.5)
            refined_radius = _compute_radius(refined_pts)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_pts = refined_pts.copy()
        except Exception as e:
            pass
    
    # Final check
    if time.time() - start_time > time_limit - 5:
        return best_pts
    
    return best_pts

# EVOLVE-BLOCK-END
