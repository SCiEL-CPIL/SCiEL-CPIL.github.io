"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import sys


def _compute_score(centres):
    """Compute the maximum common radius for the given centres."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(np.abs(centres) > 0.5 + 1e-12):
        return 0.0
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    return min(min_pairwise / 2.0, min_wall)


def _fcc_lattice_init(n_points):
    """Initialize with FCC lattice points."""
    # FCC lattice in a cube: 8 corners + 6 face centers per unit cell
    # For a cube of side 1, we can place points at FCC positions
    # Scale to fit in [-0.5, 0.5]^3
    
    # Generate FCC points in a 4x4x4 grid of unit cells
    # Each unit cell has 4 FCC points
    # Total points in 4x4x4 = 4*4*4*4 = 256, we need 100
    
    fcc_points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # FCC points in unit cell [i, i+1] x [j, j+1] x [k, k+1]
                # Corners: (0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)
                # Face centers: (0.5,0.5,0), (0.5,0.5,1), (0.5,0,0.5), (0.5,1,0.5), (0,0.5,0.5), (1,0.5,0.5)
                # But we only want interior points to avoid boundary issues
                # Let's use a simpler approach: generate FCC in a larger region and select
                
                # Actually, let's generate FCC points more carefully
                # FCC in a cube: positions at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in each unit cell
                # For a 5x5x5 grid of unit cells, we get 5^3 * 4 = 500 points
                pass
    
    # Simpler: generate FCC points in a 5x5x5 grid
    # Each cell contributes 4 points
    # We'll generate and then select the best 100
    
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # FCC positions in unit cell [i, i+1] x [j, j+1] x [k, k+1]
                # (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    # Scale to fit in [-0.5, 0.5]^3
    # The points range from 0 to 4.5 in each dimension
    # Scale by 1/4.5 and center
    points = (points - 2.25) / 4.5
    
    # Now we have 500 points, select 100 that are well-distributed
    # Use a greedy selection: pick the point that maximizes the minimum distance to already selected points
    rng = np.random.RandomState(42)
    selected = []
    available = list(range(len(points)))
    rng.shuffle(available)
    
    # Start with a random point
    selected.append(available.pop(0))
    
    while len(selected) < n_points and available:
        best_idx = -1
        best_min_dist = -1
        for idx in available:
            dists = np.linalg.norm(points[idx] - points[selected], axis=1)
            min_dist = np.min(dists) if len(selected) > 0 else np.inf
            # Also consider distance to walls
            wall_dist = 0.5 - np.abs(points[idx])
            min_wall = np.min(wall_dist)
            # Combine
            score = min(min_dist, min_wall)
            if score > best_min_dist:
                best_min_dist = score
                best_idx = idx
        
        if best_idx >= 0:
            selected.append(available.pop(available.index(best_idx)))
        else:
            break
    
    if len(selected) < n_points:
        # Pad with random points
        extra = rng.uniform(-0.5, 0.5, size=(n_points - len(selected), 3))
        return np.vstack([points[selected], extra])
    
    return points[selected]


def _max_min_move(centres, max_iters=200, time_limit=30, start_time=None):
    """Iteratively move each point to the center of the largest empty space."""
    if start_time is None:
        start_time = time.time()
    
    current = centres.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for idx in range(N):
            # Find the best position for point idx
            # Try moving it in the direction that maximizes the minimum distance
            # to all other points and walls
            
            # Compute current distances
            others = np.delete(current, idx, axis=0)
            dists_to_others = np.linalg.norm(others - current[idx], axis=1)
            wall_dists = 0.5 - np.abs(current[idx])
            min_dist = min(np.min(dists_to_others), np.min(wall_dists))
            
            # Try several candidate moves
            best_move = None
            best_score_move = current_score
            
            # Radial search: try moving in 8 directions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        new_pos = current[idx] + np.array([dx, dy, dz]) * 0.02
                        new_pos = np.clip(new_pos, -0.5, 0.5)
                        
                        # Compute new score
                        new_centres = current.copy()
                        new_centres[idx] = new_pos
                        new_score = _compute_score(new_centres)
                        
                        if new_score > best_score_move:
                            best_score_move = new_score
                            best_move = new_pos
            
            if best_move is not None:
                current[idx] = best_move
                current_score = best_score_move
                improved = True
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
        
        if not improved:
            break
    
    return best, best_score


def _soft_min_objective(x, beta, centres):
    """Soft-min surrogate for the minimum distance objective."""
    # x is the flattened position of one point
    # centres is the full array with x replaced
    x = x.reshape(3)
    new_centres = centres.copy()
    new_centres[0] = x
    
    # Compute pairwise distances
    dists = cdist(new_centres, new_centres)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of pairwise distances
    log_sum_exp = np.log(np.sum(np.exp(-beta * dists))) / beta
    soft_min_pair = -log_sum_exp
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(x)
    soft_min_wall = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
    
    # Objective: maximize min(soft_min_pair, soft_min_wall)
    # Equivalent to minimize -min(soft_min_pair, soft_min_wall)
    # Use soft-min of the two
    result = -np.log(np.sum(np.exp(-beta * np.array([soft_min_pair, soft_min_wall])))) / beta
    return -result  # Negate for minimization


def _gradient_refine(centres, beta_start=5.0, beta_end=50.0, max_iters=100, time_limit=60, start_time=None):
    """Gradient-based refinement using soft-min surrogate."""
    if start_time is None:
        start_time = time.time()
    
    current = centres.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # Progressively increase beta
    betas = np.linspace(beta_start, beta_end, max_iters)
    
    for it, beta in enumerate(betas):
        if time.time() - start_time > time_limit:
            break
        
        # Optimize each point sequentially
        for idx in range(N):
            if time.time() - start_time > time_limit:
                break
            
            # Define objective for point idx
            def objective(x):
                new_centres = current.copy()
                new_centres[idx] = x.reshape(3)
                # Compute soft-min objective
                dists = cdist(new_centres, new_centres)
                np.fill_diagonal(dists, np.inf)
                log_sum_exp = np.log(np.sum(np.exp(-beta * dists))) / beta
                soft_min_pair = -log_sum_exp
                wall_dists = 0.5 - np.abs(x.reshape(3))
                soft_min_wall = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
                result = -np.log(np.sum(np.exp(-beta * np.array([soft_min_pair, soft_min_wall])))) / beta
                return -result
            
            # Optimize
            x0 = current[idx].flatten()
            bounds = [(-0.5, 0.5)] * 3
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 10, 'gtol': 1e-4})
            
            if res.success:
                current[idx] = res.x
                new_score = _compute_score(current)
                if new_score > best_score:
                    best = current.copy()
                    best_score = new_score
    
    return best, best_score


def construct():
    """Main entry point: find the best packing of 100 equal spheres in a unit cube."""
    start_time = time.time()
    total_time_limit = 140  # Leave 10s margin
    
    # Start with FCC lattice initialization
    best_centres = _fcc_lattice_init(N)
    best_score = _compute_score(best_centres)
    
    # Phase 1: Max-min move to improve local packing
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 40:
        result, score = _max_min_move(best_centres, max_iters=50, 
                                       time_limit=min(30, remaining_time - 10),
                                       start_time=start_time)
        if score > best_score:
            best_score = score
            best_centres = result
    
    # Phase 2: Gradient-based refinement
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 30:
        result, score = _gradient_refine(best_centres, beta_start=5.0, beta_end=50.0,
                                          max_iters=50, time_limit=min(60, remaining_time - 5),
                                          start_time=start_time)
        if score > best_score:
            best_score = score
            best_centres = result
    
    # Final check
    if best_centres is None or not np.all(np.isfinite(best_centres)):
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
