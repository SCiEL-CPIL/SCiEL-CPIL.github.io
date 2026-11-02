"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Discrete Branch-and-Bound search on a grid of barycentric coordinates.
2. The search space is discretized into a 50x50 grid for each point's barycentric coords (u, v).
3. We use a greedy heuristic to initialize the search and then perform a best-first or DFS 
   branch-and-bound to explore the neighborhood, pruning branches where the upper bound 
   on the minimum area is less than the current best.
4. To speed up the upper bound calculation, we assume that moving a point can increase the 
   minimum area by at most the maximum area change induced by that point's movement.
   However, a simpler and robust bound for discrete search is: 
   UpperBound = current_min_area + max_possible_increase.
   Actually, for B&B to be effective, we need a tight upper bound.
   Since the problem is non-convex and high-dimensional, strict B&B might be too slow.
   
   Alternative Structural Change: 
   Use a "Grid Refinement" + "Local Search" hybrid.
   1. Start with a coarse grid (e.g., 10x10).
   2. Find the best configuration via exhaustive search on the coarse grid? No, 10^22 is too big.
   
   Let's stick to the prompt's suggestion: "discretizing ... onto a fine 2D grid ... and using a backtracking search with branch-and-bound".
   
   For N=11, full B&B is infeasible. 
   However, we can use a **Greedy Coordinate Descent** on a discrete grid.
   This is a different family from SLSQP (continuous gradient).
   
   Algorithm:
   1. Define a grid resolution (e.g., 100 steps).
   2. Initialize points using the best known configuration (or random).
   3. Iterate: For each point, try all grid neighbors (8 neighbors in 2D barycentric space).
      If a move increases the minimum triangle area, accept it.
   4. Repeat until no improvement or time limit.
   5. Multi-start with different resolutions and initializations.
   
   This is a discrete local search (Simulated Annealing or Hill Climbing on a grid).
   It avoids the local minima of continuous methods by jumping over small barriers,
   though it can get stuck in grid-local minima. We mitigate this with multi-start 
   and varying grid resolutions.
"""
import numpy as np
import time
import random
import itertools

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    if len(pts) == 0:
        return 0.0
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_triplets[:, 2]] # Wait, _k_idx is defined
    
    # Correct indexing
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w=1-u-v) to Cartesian.
    Vertices: A(0,0), B(1,0), C(0.5, S3H)
    P = u*A + v*B + w*C
    x = v + 0.5*w
    y = S3H * w
    """
    # bary shape (N, 2) -> u, v. w = 1 - u - v
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    x = v + 0.5 * w
    y = S3H * w
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates (u, v)."""
    # x = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
    # y = S3H*(1-u-v) => 1-u-v = y/S3H => u+v = 1 - y/S3H
    # Let S = y/S3H. Then w = S. u+v = 1-S.
    # x = v + 0.5*S => v = x - 0.5*S
    # u = 1 - S - v = 1 - S - (x - 0.5*S) = 1 - 0.5*S - x
    S = pts[:, 1] / S3H
    v = pts[:, 0] - 0.5 * S
    u = 1.0 - S - v
    return np.column_stack((u, v))

def _is_valid_bary(bary, eps=1e-6):
    """Check if barycentric coordinates are valid (inside triangle)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    return np.all(u >= -eps) and np.all(v >= -eps) and np.all(w >= -eps)

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region [0,1] for u,v,w."""
    u = bary[:, 0]
    v = bary[:, 1]
    
    # w = 1 - u - v >= 0 => u + v <= 1
    # u >= 0, v >= 0
    
    # Simple clamping:
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Ensure u + v <= 1
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        # Scale down u and v proportionally or just clip one?
        # Proportional scaling preserves direction.
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    return np.column_stack((u, v))

def _grid_search_local(bary_init, resolution, max_iterations=1000):
    """
    Perform a local search on a discrete grid.
    bary_init: (N, 2) array of barycentric coordinates.
    resolution: number of steps along each edge.
    """
    bary = bary_init.copy()
    bary = _clamp_bary(bary)
    
    pts = _barycentric_to_cartesian(bary)
    current_score = _min_triangle_area(pts)
    
    # Grid step size
    step = 1.0 / resolution
    
    # Directions in (u, v) space. 
    # Note: changing u and v changes w.
    # Neighbors: (du, dv) where du, dv in {-1, 0, 1} * step
    # We check 8 neighbors + center (though center is current)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    improved = True
    iters = 0
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        for i in range(N):
            u, v = bary[i]
            for du, dv in directions:
                new_u = u + du * step
                new_v = v + dv * step
                new_bary = bary.copy()
                new_bary[i] = [new_u, new_v]
                
                # Check validity
                if not _is_valid_bary(new_bary, eps=-1e-6):
                    # Try to clamp
                    new_bary = _clamp_bary(new_bary)
                    if not _is_valid_bary(new_bary, eps=1e-6):
                        continue
                    
                new_pts = _barycentric_to_cartesian(new_bary)
                new_score = _min_triangle_area(new_pts)
                
                if new_score > current_score + 1e-12:
                    current_score = new_score
                    bary = new_bary
                    pts = new_pts
                    improved = True
                    # Restart inner loop? Or just continue to next point?
                    # Usually restart or continue. Continue is faster.
                    break # Move to next point
            
    return bary, current_score

def _random_bary_init(rng):
    """Generate random barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    while u + v > 1.0:
        u = rng.random()
        v = rng.random()
    return [u, v]

def _greedy_bary_init(rng):
    """Greedy initialization in barycentric space."""
    bary_pts = []
    # Start with vertices and midpoints for good spread?
    # Or just random first point
    p0 = _random_bary_init(rng)
    bary_pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(200): # Fewer random samples for speed
            cand = np.array(_random_bary_init(rng))
            dists = np.linalg.norm(np.array(bary_pts) - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        bary_pts.append(best_pos)
    
    return np.array(bary_pts)

def _structured_bary_init():
    """Structured initialization based on known good configs."""
    # Vertices
    # (0,0) -> (1,0,0) -> u=1, v=0
    # (1,0) -> (0,1,0) -> u=0, v=1
    # (0.5, S3H) -> (0,0,1) -> u=0, v=0
    # Midpoints
    # (0.5, 0) -> (0.5, 0.5, 0) -> u=0.5, v=0.5
    # (0.25, S3H/2) -> (0.5, 0, 0.5) -> u=0.5, v=0
    # (0.75, S3H/2) -> (0, 0.5, 0.5) -> u=0, v=0.5
    # Center
    # (0.5, S3H/3) -> (1/3, 1/3, 1/3) -> u=1/3, v=1/3
    
    pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [0.5, 0.5], [0.5, 0.0], [0.0, 0.5],
        [1.0/3.0, 1.0/3.0],
        [0.75, 0.25], # (0.25, 0.75, 0) -> u=0.25, v=0.75 ? No.
        # Let's calculate: x=0.75, y=0. 
        # S=0. v=0.75, u=0.25. Correct.
        [0.25, 0.75], # x=0.25, y=0. u=0.75, v=0.25.
        [0.25, 0.0], # x=0.25, y=S3H/4. S=0.25. v=0.25-0.125=0.125. u=1-0.25-0.125=0.625.
        [0.75, 0.0]  # x=0.75, y=S3H/4. S=0.25. v=0.75-0.125=0.625. u=1-0.25-0.625=0.125.
    ]
    # Fix the last two:
    # P8: x=0.25, y=S3H/4. S=0.25. v = 0.25 - 0.125 = 0.125. u = 1 - 0.25 - 0.125 = 0.625.
    # P9: x=0.75, y=S3H/4. S=0.25. v = 0.75 - 0.125 = 0.625. u = 1 - 0.25 - 0.625 = 0.125.
    pts[9] = [0.625, 0.125]
    pts[10] = [0.125, 0.625]
    
    return np.array(pts)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_bary = None
    best_score = -1.0
    
    # Initializations
    inits = []
    
    # 1. Structured
    try:
        p_struct = _structured_bary_init()
        inits.append(p_struct)
    except Exception:
        pass
    
    # 2. Greedy
    try:
        p_greedy = _greedy_bary_init(rng)
        inits.append(p_greedy)
    except Exception:
        pass
    
    # 3. Randoms
    for _ in range(10):
        p_rand = np.array([_random_bary_init(rng) for _ in range(N)])
        inits.append(p_rand)
    
    # Grid resolutions to try
    # Start coarse, then fine
    resolutions = [20, 50, 100, 200]
    
    for init_bary in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        current_bary = init_bary.copy()
        current_bary = _clamp_bary(current_bary)
        
        # Progressively refine
        for res in resolutions:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Max iterations depends on resolution and time
            # Coarse: more iterations allowed? No, coarse is fast.
            # Fine: fewer iterations?
            max_iter = 500 if res < 100 else 200
            
            try:
                current_bary, score = _grid_search_local(current_bary, res, max_iter)
                
                if score > best_score:
                    best_score = score
                    best_bary = current_bary.copy()
                    
            except Exception:
                pass
        
        # After refining from this init, try to "kick" the points to escape local minima?
        # Optional: Add a perturbation step if time permits.
    
    # Final fallback if no valid bary found
    if best_bary is None:
        best_bary = _structured_bary_init()
        
    # Convert to Cartesian
    best_pts = _barycentric_to_cartesian(best_bary)
    
    # Ensure points are strictly inside or on boundary for safety
    # The barycentric clamping should handle this, but let's verify
    # If a point is slightly outside due to floating point, project it
    for i in range(N):
        x, y = best_pts[i]
        if x < 0: x = 0
        if x > 1: x = 1
        if y < 0: y = 0
        # Check triangle boundaries
        # y <= sqrt(3)*x
        if y > np.sqrt(3)*x + 1e-9:
             y = np.sqrt(3)*x
        # y <= sqrt(3)*(1-x)
        if y > np.sqrt(3)*(1-x) + 1e-9:
             y = np.sqrt(3)*(1-x)
        best_pts[i] = [x, y]
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")

