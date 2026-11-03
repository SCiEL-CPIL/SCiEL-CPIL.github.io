"""
SQP-based packer for 100 equal spheres in a unit cube.

Strategy:
1. Start with a grid initialization.
2. Formulate the problem as maximizing r subject to:
   - ||c_i - c_j|| >= 2r for all i < j
   - ||c_i||_inf <= 0.5 - r for all i, k
3. Use a Sequential Quadratic Programming (SQP) approach:
   - At each step, linearize the nonlinear constraints.
   - Solve a quadratic programming (QP) subproblem to find the step direction.
   - Update the points and radius.
4. Since scipy.optimize.minimize doesn't natively support QP subproblems with 
   inequality constraints on the step directly in a standard SQP loop without 
   custom implementation, we use a penalty method or a specific trick:
   
   Alternative Strategy (More Robust for Python/SciPy stack):
   Use a "soft" constraint satisfaction approach combined with a dedicated 
   local search that specifically targets the "active set".
   
   However, the most effective different strategy from "L-BFGS on min-dist" 
   is often a **Repulsive Force / Energy Minimization** approach where we 
   minimize the sum of inverse powers of distances (or similar) which is 
   smooth and differentiable, driving points apart until they hit the 
   "equilibrium" where the minimum distance is maximized.
   
   Let's implement a **Smooth Potential Energy Minimization**:
   Objective: E(x, r) = sum_{i<j} (1 / max(d_ij, epsilon))^p + sum_i (1 / max(m_i, epsilon))^p
   where d_ij is distance, m_i is margin.
   Minimizing this with L-BFGS-B is smooth and guides the optimizer to 
   increase the minimum distance and margin simultaneously.
   
   To ensure we maximize the *minimum* radius specifically, we can use a 
   "barrier" or "penalty" that becomes very steep as the minimum distance 
   decreases.
   
   Actually, a better different strategy is **Active Set Search**:
   1. Find the current bottleneck constraints (pairs with min distance, points 
      with min margin).
   2. Move only those points to increase the minimum distance, keeping others fixed.
   3. This is essentially a specialized local search.
   
   Let's combine:
   1. Initial Grid.
   2. Global-ish search using **Repulsive Forces** (smooth objective) to get 
      into a good basin.
   3. Polish with **Active Set Quadratic Programming** (using scipy.optimize 
      or manual gradient ascent on the active constraints).

   Implementation Plan:
   1. _score: Standard exact score.
   2. _energy: Smooth surrogate E = sum (1/d^2) + sum (1/m^2). Minimizing this 
      tends to maximize min(d, m).
   3. _optimize_energy: Use L-BFGS-B to minimize _energy.
   4. _polish_active: Identify active constraints. Use a small Newton step 
      or projection to satisfy them exactly and maximize r.
   5. Main loop: Multi-start with Energy Minimization + Active Polish.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

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
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _energy(points, power=2.0, epsilon=1e-6):
    """
    Smooth energy function to be minimized.
    High energy when points are close or near walls.
    """
    pts = np.asarray(points, dtype=np.float64)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero
    dists_safe = np.maximum(dists, epsilon)
    # Sum of inverse powers for pairs
    pair_energy = np.sum(1.0 / (dists_safe ** power))
    
    # Margins
    margin = 0.5 - np.abs(pts).max(axis=1)
    margin_safe = np.maximum(margin, epsilon)
    # Sum of inverse powers for margins
    margin_energy = np.sum(1.0 / (margin_safe ** power))
    
    return pair_energy + margin_energy

def _neg_soft_radius(points, beta=100.0):
    """
    Soft minimum approximation using log-sum-exp.
    Minimizing -soft_min is equivalent to maximizing soft_min.
    """
    pts = np.asarray(points, dtype=np.float64)
    if np.any(np.abs(pts) > 0.5):
        return 1e10
        
    margin = 0.5 - np.abs(pts).max(axis=1)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    dists_half = dists / 2.0
    
    # Combine all constraints into a single vector
    # We want max( min(margin), min(dists_half) )
    # This is equivalent to max( min( all_margins, all_dists_half ) )
    
    # Flatten constraints
    cons = np.concatenate([margin, dists_half.flatten()])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # To avoid overflow, subtract max
    x_min = cons.min()
    # If x_min is very small, exp(-beta * (x - x_min)) might overflow if x is large?
    # No, x - x_min >= 0, so exp(-beta * (x - x_min)) <= 1. Safe.
    shifted = cons - x_min
    # Clip shifted to avoid underflow issues in sum if beta is huge, but exp(-large) -> 0 is fine.
    exp_terms = np.exp(-beta * shifted)
    s = np.sum(exp_terms)
    soft_min_val = x_min - (1.0 / beta) * np.log(s)
    
    return -soft_min_val

def _grid_init():
    """Generate a 5x5x4 grid."""
    nx, ny, nz = 5, 5, 4
    # Spacing:
    # x: 5 pts in [0.5, 0.5] -> range 0.8? No, cube is [-0.5, 0.5].
    # If we place 5 points, spacing is 1/4 = 0.25?
    # Let's use linspace with padding to ensure they are inside.
    # Actually, for a 5x5x4 grid to maximize min dist:
    # x: 5 points. Spacing dx. 4*dx <= 1. dx = 0.25.
    # y: 5 points. dy = 0.25.
    # z: 4 points. dz. 3*dz <= 1. dz = 1/3 = 0.333.
    # Min dist would be 0.25. Radius 0.125.
    # But we can't have dx=0.25 and dy=0.25 and dz=0.333 simultaneously with 
    # centers at 0? 
    # x: -0.5, -0.25, 0, 0.25, 0.5. (Points on boundary).
    # Margin for boundary points is 0. So radius is 0.
    # We need to pull them in.
    # Optimal 1D packing of n points in [-0.5, 0.5] with max min-dist to boundary and each other:
    # Points at -0.5 + i*(1/(n-1))? No, that puts them on boundary.
    # To maximize min(dist_to_neighbor, dist_to_wall):
    # Let r be radius. Points must be in [-0.5+r, 0.5-r].
    # Length available: 1 - 2r.
    # n points, n-1 gaps. Gap size g. (n-1)g <= 1 - 2r.
    # Also g >= 2r.
    # So (n-1)2r <= 1 - 2r => 2r(n-1) + 2r <= 1 => 2rn <= 1 => r <= 1/(2n).
    # For n=5, r <= 1/10 = 0.1.
    # For n=4, r <= 1/8 = 0.125.
    # So 5x5x4 grid limits r to 0.1.
    # 4x4x6? 4*4*6=96. Need 100.
    # 5x5x4 is the standard grid.
    
    # Let's just create a grid and let the optimizer fix it.
    # Initial guess: r=0.1
    # x: 5 pts in [-0.4, 0.4]. Spacing 0.2.
    # y: 5 pts in [-0.4, 0.4]. Spacing 0.2.
    # z: 4 pts in [-0.375, 0.375]. Spacing 0.25.
    
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _optimize_smooth(points, method='L-BFGS-B', maxiter=100):
    """Minimize the soft radius or energy."""
    def obj(x):
        return _neg_soft_radius(x.reshape(N, 3), beta=50.0)
    
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(obj, x0, method=method, bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.fun < obj(x0):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return points

def _optimize_energy(points, method='L-BFGS-B', maxiter=100):
    """Minimize the repulsive energy."""
    def obj(x):
        return _energy(x.reshape(N, 3), power=2.0)
    
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(obj, x0, method=method, bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.fun < obj(x0):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return points

def _local_polish(points, max_iter=50):
    """
    Active set local search.
    1. Calculate current r.
    2. Find active constraints (pairs with dist < 2r + eps, walls with margin < r + eps).
    3. Solve a small QP to increase r slightly while satisfying active constraints.
    4. Repeat.
    """
    pts = points.copy()
    r = _score(pts)
    
    for _ in range(max_iter):
        # Calculate distances and margins
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        margin = 0.5 - np.abs(pts).max(axis=1)
        
        # Active constraints: dist < 2r * 1.01 or margin < r * 1.01
        # We want to move points to increase the minimum of these.
        
        # Simple heuristic move:
        # For each point, compute gradient of the "bottleneck".
        # This is complex. Instead, use a random perturbation biased towards 
        # increasing the min distance.
        
        # Find the pair with min distance
        min_dist_idx = np.unravel_index(dists.argmin(), dists.shape)
        i, j = min_dist_idx
        if i == j:
            continue
            
        d_ij = dists[i, j]
        if d_ij > 2 * r:
             # No pair constraint is active? Then wall is active.
             min_margin_idx = margin.argmin()
             i = min_margin_idx
             j = -1 # Special case for wall
        else:
             # Pair constraint active
             pass
             
        # This heuristic is weak. Let's stick to the smooth optimization 
        # which is more robust for Python.
        # The "different strategy" is the smooth potential/soft-min optimization 
        # vs the raw min objective.
        
        # Let's do a final Newton-like step using the gradient of the soft-min.
        # We already did that in _optimize_smooth.
        
        break
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial solution
    points = _grid_init()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Smooth optimization from grid
    if time.time() < deadline - 60:
        # Try soft-min optimization
        p1 = _optimize_smooth(points, maxiter=200)
        s1 = _score(p1)
        if s1 > best_score:
            best_points = p1
            best_score = s1
            
        # Try energy optimization
        p2 = _optimize_energy(points, maxiter=200)
        s2 = _score(p2)
        if s2 > best_score:
            best_points = p2
            best_score = s2

    # Phase 2: Multi-start
    # Generate diverse starts using jittered grids and random clusters
    while time.time() < deadline - 20:
        # Create a new start
        # Option A: Jittered Grid
        if random.random() < 0.7:
            new_start = _grid_init() + np.random.normal(0, 0.02, (N, 3))
            new_start = np.clip(new_start, -0.5, 0.5)
        else:
            # Option B: Random points, then cluster
            new_start = np.random.uniform(-0.4, 0.4, (N, 3))
            
        # Optimize the new start
        # Alternate between soft-min and energy
        if random.random() < 0.5:
            opt_pts = _optimize_smooth(new_start, maxiter=100)
        else:
            opt_pts = _optimize_energy(new_start, maxiter=100)
            
        opt_score = _score(opt_pts)
        
        if opt_score > best_score:
            best_points = opt_pts
            best_score = opt_score
            
        # If we have time, do a second pass on the best found so far
        if time.time() < deadline - 10 and random.random() < 0.2:
            # Polish the current best with a different method
            if random.random() < 0.5:
                polish_pts = _optimize_smooth(best_points, maxiter=50)
            else:
                polish_pts = _optimize_energy(best_points, maxiter=50)
                
            polish_score = _score(polish_pts)
            if polish_score > best_score:
                best_points = polish_pts
                best_score = polish_score

    # Final Polish
    if time.time() < deadline - 5:
        # Use the method that worked best previously or just try both
        p1 = _optimize_smooth(best_points, maxiter=50)
        s1 = _score(p1)
        if s1 > best_score:
            best_points = p1
            best_score = s1
            
        p2 = _optimize_energy(best_points, maxiter=50)
        s2 = _score(p2)
        if s2 > best_score:
            best_points = p2
            best_score = s2

    return best_points

# EVOLVE-BLOCK-END
