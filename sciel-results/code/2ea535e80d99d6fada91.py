"""
Strategy: Direct Search with Collision-Resolution Polish.

The previous strategy used L-BFGS-B on a smooth surrogate (soft-min). This new strategy:
1. Uses a population-based evolution strategy (similar to CMA-ES but simplified) 
   that directly optimizes the true objective: the minimum radius.
   This avoids the approximation error of the soft-min and handles non-differentiability naturally.
2. Incorporates a "collision resolution" step: if spheres overlap, we explicitly push them apart.
   This acts as a hard constraint enforcement that can jump out of local minima where 
   gradient methods might stall due to flat regions or sharp corners.
3. Uses a "jitter" mechanism to escape local optima by randomly perturbing points 
   and accepting if the score improves.

The objective is strictly the true score: min(min_pair_dist/2, min_wall_margin).
"""

import numpy as np
import time

N = 100
CUBE_HALF = 0.5

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _resolve_collisions(pts, r_target, max_iter=50):
    """
    Attempt to resolve collisions by pushing spheres apart.
    This is a heuristic local search step.
    """
    pts = pts.copy()
    for _ in range(max_iter):
        # Check for overlaps
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find pairs that are too close
        min_dist = dists.min()
        if min_dist >= 2 * r_target:
            break
            
        # Find the closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Move them apart
        vec = pts[i] - pts[j]
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            # Random kick
            vec = np.random.normal(0, 1, 3)
            norm = np.linalg.norm(vec)
        
        # Required separation
        sep = 2 * r_target
        current_sep = norm
        deficit = sep - current_sep
        
        if deficit > 0:
            # Move i and j apart by half the deficit each
            move = (deficit / 2) * (vec / norm)
            pts[i] += move
            pts[j] -= move
            
            # Ensure they stay in bounds
            # If they hit the wall, push the other one more? 
            # For now, just clamp and let the score handle it.
            # A better approach: if i hits wall, push j fully.
            # Let's just clamp for simplicity in this heuristic.
            pts[i] = np.clip(pts[i], -0.5, 0.5)
            pts[j] = np.clip(pts[j], -0.5, 0.5)
            
    return pts

def _local_polish(pts, r_current):
    """
    Improve a configuration by slightly increasing r and resolving collisions.
    """
    best_pts = pts.copy()
    best_r = r_current
    
    # Try to increase radius slightly
    for _ in range(20):
        r_new = best_r * 1.001 + 1e-6
        # Resolve collisions with the new target radius
        candidate = _resolve_collisions(best_pts, r_new, max_iter=100)
        # Check if candidate is valid and better
        r_actual = _score(candidate)
        if r_actual > best_r:
            best_r = r_actual
            best_pts = candidate
        else:
            # If we can't improve, break
            break
            
    return best_pts, best_r

def _evolution_step(population, scores, sigma):
    """
    Simple evolution strategy step.
    population: (P, N, 3)
    scores: (P,)
    sigma: step size
    """
    P = len(population)
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    population = population[sorted_indices]
    scores = scores[sorted_indices]
    
    # Keep top half as parents
    n_parents = max(2, P // 2)
    parents = population[:n_parents]
    
    # Generate children
    children = []
    for _ in range(P):
        # Pick a random parent
        p_idx = np.random.randint(n_parents)
        parent = parents[p_idx]
        
        # Add noise
        noise = np.random.normal(0, sigma, (N, 3))
        child = parent + noise
        
        # Clamp to cube
        child = np.clip(child, -0.5, 0.5)
        
        children.append(child)
        
    children = np.array(children)
    return children

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial population
    # 1. Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    grid_pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # 2. Random
    rng = np.random.default_rng(42)
    population = [grid_pts]
    for _ in range(9):
        rand_pts = rng.uniform(-0.45, 0.45, (N, 3))
        population.append(rand_pts)
        
    population = np.array(population)
    P = len(population)
    
    # Evaluate initial population
    scores = np.array([_score(p) for p in population])
    
    # Update best
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_points = population[best_idx].copy()
    
    sigma = 0.05 # Initial step size
    
    # Evolution loop
    while time.time() < deadline - 10:
        # Local polish on the best few
        for i in range(3):
            if time.time() > deadline - 10:
                break
            if scores[i] > 0:
                polished_pts, polished_score = _local_polish(population[i], scores[i])
                if polished_score > scores[i]:
                    population[i] = polished_pts
                    scores[i] = polished_score
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = polished_pts.copy()
        
        # Evolution step
        new_population = _evolution_step(population, scores, sigma)
        
        # Evaluate new population
        new_scores = np.array([_score(p) for p in new_population])
        
        # Combine and select
        all_population = np.concatenate([population, new_population], axis=0)
        all_scores = np.concatenate([scores, new_scores])
        
        # Sort and keep top P
        sorted_indices = np.argsort(all_scores)[::-1]
        population = all_population[sorted_indices[:P]]
        scores = all_scores[sorted_indices[:P]]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_points = population[0].copy()
            
        # Adapt sigma
        # If best score improved, keep sigma or increase slightly? 
        # If stagnation, decrease sigma.
        # Simple heuristic: if top 5 scores are very close, decrease sigma.
        if len(scores) > 5:
            spread = scores[0] - scores[-1]
            if spread < 1e-5:
                sigma *= 0.8
            else:
                sigma *= 1.05 # Explore more if spread is large? Or keep?
                # Actually, if we are improving, we might want to keep exploring.
                # Let's just keep sigma constant for simplicity, or decay slowly.
                sigma = max(0.001, sigma * 0.99)
                
    # Final polish on the best point
    if best_points is not None and time.time() < deadline - 2:
        polished_pts, polished_score = _local_polish(best_points, best_score)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    if best_points is None:
        best_points = grid_pts
        
    return best_points

# EVOLVE-BLOCK-END
