"""
Strategy: Simulated Annealing with Soft-Sphere Potential.
We treat the spheres as charged particles with a strong repulsive force 
when they get too close. We minimize the total potential energy, which 
corresponds to maximizing the minimum distance between spheres and walls.
We use multiple restarts with different initializations to avoid local minima.
"""

import numpy as np
import time
import math

N = 50

# EVOLVE-BLOCK-START

def _score(pts):
    """Compute the maximum radius for given centers."""
    if len(pts) != N:
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    if np.any(~np.isfinite(pts)):
        return 0.0
        
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    # Wall distances
    # Distance to nearest wall for each point
    wall_dists = 0.5 - np.abs(pts)
    wmin = np.min(wall_dists)
    
    return min(dmin / 2.0, wmin)

def _energy(pts):
    """
    Soft-sphere potential energy.
    We want to minimize this.
    E = sum_{i<j} (1/d_ij)^12 + sum_i (1/w_i)^12
    Using power 12 makes it very steep near contact, acting like a hard core
    but differentiable everywhere (except at 0, which we clip).
    """
    pts = np.clip(pts, -0.499, 0.499)
    n = len(pts)
    
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    # We want to penalize small distances.
    # Using 1/d^12 is standard Lennard-Jones repulsive part.
    # However, for optimization stability, a softer potential like 1/d^4 or 1/d^2 
    # might be easier for gradient descent, but Simulated Annealing can handle 
    # steeper potentials if steps are small.
    # Let's use a combination: Strong repulsion for very close, weaker for far.
    
    # Clip distances to avoid division by zero
    dists_sq_clipped = np.maximum(dists_sq, 1e-6)
    
    # Sum of 1/d^12 for i<j
    # To save memory and time, we can compute this in chunks or just use the matrix
    # For N=50, 50x50 matrix is fine.
    repulsive_pairs = np.sum(1.0 / np.power(dists_sq_clipped, 6.0)) # 1/d^12
    # Subtract diagonal (which is 1/1^12 = 1) to exclude self-interaction
    repulsive_pairs -= N
    
    # Wall repulsion
    wall_dists = 0.5 - np.abs(pts)
    wall_dists_clipped = np.maximum(wall_dists, 1e-3)
    repulsive_walls = np.sum(1.0 / np.power(wall_dists_clipped, 12.0))
    
    return repulsive_pairs + repulsive_walls

def _random_init():
    return np.random.uniform(-0.4, 0.4, (N, 3))

def _halton_init(n, bases=(2, 3, 5)):
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, n + 1):
        p = [halton(i, b) for b in bases]
        points.append(p)
    pts = np.array(points)
    pts = pts * 0.9 - 0.45
    return pts

def _grid_init():
    # 50 points. 4x4x4 grid is 64 points. 
    # We can take a subset or just use a 3x4x5 grid = 60 points.
    # Let's try a 3x4x5 grid.
    nx, ny, nz = 3, 4, 5
    total = nx * ny * nz
    if total < N:
        # Fallback to random if grid is too small
        return _random_init()
    
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    grid_pts = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid_pts.append([x, y, z])
                
    grid_pts = np.array(grid_pts)
    # Shuffle and take first N
    np.random.shuffle(grid_pts)
    return grid_pts[:N]

def _simulated_annealing(pts, deadline, target_score=0.1359):
    """
    Run Simulated Annealing to minimize energy.
    """
    current_pts = pts.copy()
    current_energy = _energy(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    best_score = _score(best_pts)
    
    # Cooling schedule
    T0 = 0.05
    Tend = 1e-6
    # Number of steps depends on time budget
    # Let's run for a fixed number of steps or until time runs out.
    # Each step involves energy calculation O(N^2) and update O(N).
    # For N=50, this is fast. We can do many steps.
    
    steps = 5000
    # Calculate cooling rate
    if steps > 1:
        alpha = (Tend / T0) ** (1.0 / steps)
    else:
        alpha = 1.0
    
    T = T0
    step_size = 0.05
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Generate candidate
        candidate_pts = current_pts.copy()
        # Perturb a random subset of points
        num_perturb = np.random.randint(1, 5) # 1 to 4 points
        indices = np.random.choice(N, num_perturb, replace=False)
        
        for idx in indices:
            # Random step
            step = np.random.uniform(-step_size, step_size, 3)
            candidate_pts[idx] += step
            
        # Keep inside bounds
        candidate_pts = np.clip(candidate_pts, -0.499, 0.499)
        
        candidate_energy = _energy(candidate_pts)
        
        delta_E = candidate_energy - current_energy
        
        # Metropolis criterion
        if delta_E < 0:
            current_pts = candidate_pts
            current_energy = candidate_energy
        else:
            # Accept with probability exp(-delta_E / T)
            if np.random.rand() < np.exp(-delta_E / T):
                current_pts = candidate_pts
                current_energy = candidate_energy
                
        # Update best
        current_score = _score(current_pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            best_energy = current_energy
            
        # Cool down
        T *= alpha
        
        # Adapt step size? 
        # If we are accepting most moves, increase step size.
        # If rejecting most, decrease.
        # Simple heuristic: keep step size constant or slowly decrease.
        # Let's decrease step size slowly to refine.
        step_size *= 0.999
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initializations to try
    inits = [
        ("random", _random_init()),
        ("halton", _halton_init(N)),
        ("grid", _grid_init())
    ]
    
    # Add a few more random inits if time permits
    # We will process these in a loop, allocating time based on remaining budget.
    
    # Strategy:
    # 1. Run SA on each initial config.
    # 2. If time allows, re-run SA on the best result found so far with a lower T0 (anneal further).
    
    configs = inits.copy()
    
    # Process each config
    for name, init_pts in configs:
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this run
        remaining = deadline - time.time()
        run_deadline = time.time() + min(remaining * 0.8, 30) # Max 30s per run initially
        
        pts, score = _simulated_annealing(init_pts, run_deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
        # If we found a good score, try to refine it further
        if best_score > 0.135:
            # Refine with a second pass
            if time.time() < deadline - 5:
                refine_deadline = time.time() + min(deadline - time.time() - 2, 20)
                # Use the best found as init for refinement
                # We can use a smaller T0 for refinement
                # Modify _simulated_annealing to accept T0? 
                # For now, just call it again. The T0 is fixed inside.
                # To make it better, we could create a variant, but let's stick to simple.
                pts2, score2 = _simulated_annealing(best_pts, refine_deadline)
                if score2 > best_score:
                    best_score = score2
                    best_pts = pts2.copy()

    # If we still have time, do a local search polish on the best result
    if time.time() < deadline - 5 and best_pts is not None:
        # Simple local search: try to move closest pair apart
        pts = best_pts.copy()
        score = best_score
        step = 0.01
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            dmin = np.min(dists)
            
            wall_dists = 0.5 - np.abs(pts)
            wmin = np.min(wall_dists)
            
            current_r = min(dmin / 2.0, wmin)
            
            if dmin / 2.0 < wmin:
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                i, j = idx
                if i > j: i, j = j, i
                d = pts[i] - pts[j]
                norm = np.linalg.norm(d)
                if norm > 1e-10:
                    pts[i] += (d / norm) * step
                    pts[j] -= (d / norm) * step
            else:
                idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
                i, k = idx
                if pts[i, k] > 0:
                    pts[i, k] -= step
                else:
                    pts[i, k] += step
                    
            pts = np.clip(pts, -0.499, 0.499)
            
            new_score = _score(pts)
            if new_score > score:
                score = new_score
                best_pts = pts.copy()
            else:
                step *= 0.9
                if step < 1e-6:
                    break
        best_score = score

    # Fallback
    if best_pts is None or best_score < 0.1:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _score(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
