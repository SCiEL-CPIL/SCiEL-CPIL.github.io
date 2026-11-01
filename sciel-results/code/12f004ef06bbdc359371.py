"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import sys

def _compute_radius(pts):
    """Compute the score for a given set of centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds: centers must be in [-0.5, 0.5]^2
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_wall = np.min(wall_dists)
    # Pairwise distances
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_sq = np.min(sq_dists)
    min_pair = np.sqrt(min_pair_sq)
    r = min(min_pair / 2.0, min_wall)
    return r

def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    # For n circles, use a hexagonal packing
    # In hexagonal packing, rows are offset
    # Let's generate points and trim to n
    points = []
    row = 0
    while len(points) < n:
        if row % 2 == 0:
            x_start = -0.5
            x_spacing = 1.0
        else:
            x_start = -0.5 + 0.5
            x_spacing = 1.0
        
        # Determine how many points fit in this row
        # For row 0: x = -0.5, 0.5 (2 points if spacing 1)
        # Actually, let's use a standard hexagonal lattice with spacing 1
        # Row 0: x = -0.5, 0.5
        # Row 1: x = 0, 1 -> but 1 is outside, so x = 0
        # Let's be more systematic
        
        # Standard hexagonal: 
        # Row 0 (y=0): x = -0.5, 0.5
        # Row 1 (y=h): x = 0, 1 -> clamp to [-0.5, 0.5] so x=0
        # Row 2 (y=2h): x = -0.5, 0.5
        # etc.
        
        y = row * (np.sqrt(3) / 2)
        if y > 0.5:
            break
        
        if row % 2 == 0:
            xs = np.arange(-0.5, 0.51, 1.0)
        else:
            xs = np.arange(0.0, 0.51, 1.0)
        
        for x in xs:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
        
        row += 1
    
    # If we don't have enough, use a grid
    if len(points) < n:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = [[x, y] for y in ticks for x in ticks]
    
    return np.array(points[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial configuration: hexagonal
    pts = _hexagonal_init(N)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Use SLSQP to maximize the minimum distance
    # We want to maximize r such that:
    # 1. |p_i - p_j| >= 2r for all i < j
    # 2. p_i in [-0.5 + r, 0.5 - r]^2 for all i
    
    # This is a non-convex problem. We'll use a sequential approach:
    # Start with a small r and increase it, solving feasibility at each step.
    
    # Alternatively, use a smooth surrogate. Let's try a direct optimization
    # of the minimum distance using a log-sum-exp based smooth minimum.
    
    def _smooth_min_pairwise(pts, k=100.0):
        """Smooth approximation of the minimum pairwise distance."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, 1e10)
        # Use log-sum-exp for smooth minimum
        # min(x_i) ≈ -log(sum(exp(-k*x_i))/n) / k
        # But we want to maximize the minimum, so minimize -min
        # Let's just use the actual minimum for now and add smoothing if needed
        min_sq = np.min(sq_dists)
        return np.sqrt(min_sq)
    
    def _objective(pts_flat):
        """Negative of the radius to minimize."""
        pts = pts_flat.reshape(N, 2)
        # Penalty for going outside bounds
        out_of_bounds = np.maximum(0, np.abs(pts) - 0.5)
        penalty = 100.0 * np.sum(out_of_bounds ** 2)
        
        # Compute minimum pairwise distance and wall distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        min_pair = np.sqrt(np.min(sq_dists))
        
        wall_dists = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        min_wall = np.min(wall_dists)
        
        r = min(min_pair / 2.0, min_wall)
        return -r + penalty
    
    def _constraints(pts_flat):
        """Constraints: points must be within [-0.5, 0.5]^2."""
        pts = pts_flat.reshape(N, 2)
        # For each point and each coordinate, -0.5 <= x <= 0.5
        cons = []
        for i in range(N):
            for j in range(2):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[i*2+j] + 0.5})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 0.5 - x[i*2+j]})
        return cons
    
    # Run SLSQP
    try:
        res = minimize(
            _objective,
            pts.flatten(),
            method='SLSQP',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < -best_r:
            new_pts = res.x.reshape(N, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_pts
    
    # Try L-BFGS-B with a different objective: maximize the sum of log distances
    # This is a smooth surrogate that encourages all distances to be large
    
    def _log_obj(pts_flat):
        pts = pts_flat.reshape(N, 2)
        # Penalty for bounds
        out = np.maximum(0, np.abs(pts) - 0.5 + 1e-6)
        penalty = 1000.0 * np.sum(out ** 2)
        
        # Log of pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, 1.0)
        # Clip to avoid log(0)
        sq_dists = np.maximum(sq_dists, 1e-12)
        log_dists = 0.5 * np.log(sq_dists)
        # Sum of log distances (excluding self)
        total = np.sum(log_dists) - N * np.log(1.0)  # diagonal is 1, log=0
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        wall_dists = np.maximum(wall_dists, 1e-12)
        log_walls = np.sum(np.log(wall_dists))
        
        return -(total + 10.0 * log_walls) + penalty
    
    try:
        res2 = minimize(
            _log_obj,
            best_pts.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
        )
        if res2.success:
            new_pts = res2.x.reshape(N, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_pts
    
    # Final polish with SLSQP on the minimum distance
    def _min_obj(pts_flat):
        pts = pts_flat.reshape(N, 2)
        out = np.maximum(0, np.abs(pts) - 0.5 + 1e-8)
        penalty = 1e6 * np.sum(out ** 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        min_pair = np.sqrt(np.min(sq_dists))
        
        wall_dists = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        min_wall = np.min(wall_dists)
        
        r = min(min_pair / 2.0, min_wall)
        return -r + penalty
    
    try:
        res3 = minimize(
            _min_obj,
            best_pts.flatten(),
            method='SLSQP',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
        )
        if res3.success:
            new_pts = res3.x.reshape(N, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
