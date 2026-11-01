"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _score(pts):
    """Calculate the radius for a given set of points."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall constraint
    # pts are in [-0.5, 0.5]
    # Distance to wall is min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    # Since centered at 0, distance to boundary is 0.5 - |coord|
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances
    # Using broadcasting for pairwise differences
    # To save memory/time for N=50, we can do it in chunks or just full matrix
    # 50x50 is small.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Radius is half the minimum distance
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _feasibility_penalty(pts, target_r):
    """
    Calculate penalty for points not fitting with radius target_r.
    Returns 0 if feasible, >0 otherwise.
    """
    # Wall violation
    # Points must be at least target_r away from walls
    # i.e., |x| <= 0.5 - target_r
    wall_violation = np.maximum(0, np.abs(pts) - (0.5 - target_r))
    
    # Pairwise violation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Distance must be >= 2 * target_r
    pair_violation = np.maximum(0, 2 * target_r - dists)
    
    # Sum of squares of violations (smooth)
    return np.sum(wall_violation**2) + np.sum(pair_violation**2)

def _local_search(pts, target_r, max_iters, time_limit):
    """
    Local search to minimize feasibility penalty for a given target radius.
    Uses a simple gradient-free approach (random perturbations) with annealing.
    """
    best_pts = pts.copy()
    best_penalty = _feasibility_penalty(best_pts, target_r)
    
    # Initial temperature
    T = 1.0
    cooling = 0.99
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Generate candidate
        # Perturbation size decreases with time
        step_size = 0.01 * (T + 0.1)
        candidate = best_pts + rng.normal(0, step_size, size=best_pts.shape)
        
        # Keep inside square roughly to avoid huge penalties from walls
        candidate[:, 0] = np.clip(candidate[:, 0], -0.5, 0.5)
        candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.5)
        
        candidate_penalty = _feasibility_penalty(candidate, target_r)
        
        # Acceptance criterion (Metropolis)
        if candidate_penalty < best_penalty:
            best_pts = candidate
            best_penalty = candidate_penalty
        else:
            delta = candidate_penalty - best_penalty
            if T > 1e-6:
                if np.log(rng.random()) < -delta / T:
                    best_pts = candidate
                    best_penalty = candidate_penalty
        
        T *= cooling
        
        # If we found a perfect solution, we can stop early or just keep going
        # to ensure robustness, but 0 penalty is the goal.
        if best_penalty < 1e-12:
            break
            
    return best_pts, best_penalty

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # 1. Binary search on radius
    # Lower bound: Grid packing radius
    # Upper bound: Theoretical max or slightly above known best
    # Known best is ~0.071377. Let's search [0.06, 0.072]
    
    lo = 0.06
    hi = 0.072
    
    # Initial configuration: Hexagonal grid
    # This provides a good starting point for local search
    
    def get_hex_initial():
        # Generate points in a hexagonal pattern
        pts = []
        # Approximate number of points per row
        # For 50 points, maybe 7 rows?
        # Let's just use a dense grid and jitter it, or construct a specific hex pattern
        # Simpler: Random uniform initialization within the valid box for the current radius
        # But random is bad. Let's use a "quasi-random" or grid.
        
        # Let's use a 7x8 grid roughly, then shuffle/optimize
        # Actually, for binary search, we need a robust initializer.
        # A standard grid is a safe bet for the lower bound, 
        # but for higher radii, we need a better start.
        
        # Let's try a few different initializers and pick the one with lowest penalty
        initializers = []
        
        # 1. Square Grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        initializers.append(grid[:N])
        
        # 2. Hexagonal Grid
        # Generate hex points
        hex_pts = []
        spacing = 1.0 / 7.0 # rough estimate
        dy = spacing * np.sqrt(3) / 2
        for row in range(8):
            y = -0.5 + spacing/2 + row * dy
            if y > 0.5: break
            offset = 0 if row % 2 == 0 else spacing / 2
            for col in range(8):
                x = -0.5 + spacing/2 + col * spacing + offset
                if x > 0.5: break
                hex_pts.append([x, y])
        
        # If we have more than N, take first N. If less, pad?
        # For N=50, 7x7 grid is 49. 8x7 is 56.
        # Let's just use the grid for now, it's safer.
        # The local search will move them.
        
        # 3. Random
        initializers.append(np.random.uniform(-0.4, 0.4, (N, 2)))
        
        return initializers

    best_radius = 0.0
    best_pts = None
    
    # Binary search iterations
    # We have ~140s. 
    # Each local search takes some time. 
    # Let's do ~15-20 steps of binary search.
    # Time per step: ~5-7s?
    
    for i in range(20):
        mid = (lo + hi) / 2.0
        
        if time.time() > deadline - 10:
            break
            
        # Try to find a packing for radius mid
        # Use multiple restarts? No, time is tight. 
        # Use the best previous points as a warm start?
        # Or generate a fresh grid?
        
        # Strategy: Start with a grid, run local search.
        # If it fails (penalty > 0), maybe try a random start?
        
        init_pts = np.array([(x, y) for y in ticks for x in ticks][:N])
        # Jitter the grid slightly to break symmetry
        init_pts += np.random.uniform(-0.01, 0.01, init_pts.shape)
        
        # Run local search
        # Allocate time based on remaining budget
        time_left = deadline - time.time()
        alloc_time = min(6.0, time_left * 0.8)
        
        pts, penalty = _local_search(init_pts, mid, max_iters=2000, time_limit=alloc_time)
        
        if penalty < 1e-8:
            # Feasible
            lo = mid
            best_pts = pts
            best_radius = mid
        else:
            # Not feasible
            hi = mid
            
    # If we found a best_pts, we can try to polish it a bit more if time remains
    if best_pts is not None and time.time() < deadline - 5:
        # Try to maximize radius from the found feasible point
        # This is a local optimization of the radius itself
        # But binary search already maximized it.
        # Let's just verify the score
        pass
        
    if best_pts is None:
        # Fallback to grid
        return grid[:N]
        
    return best_pts

# EVOLVE-BLOCK-END
