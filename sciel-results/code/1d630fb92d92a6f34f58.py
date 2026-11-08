"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1
H = np.sqrt(3) / 2.0     # height of the triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _barycentric_to_cartesian(b):
    """Convert barycentric coordinates (u,v,w) to Cartesian."""
    u, v, w = b
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.array([x, y])


def _cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric coordinates."""
    x, y = p
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.array([u, v, w])


def _clamp_to_triangle(p, eps=1e-9):
    """Clamp point to be inside the triangle using barycentric coordinates."""
    b = _cartesian_to_barycentric(p)
    # If any barycentric coordinate is negative, clamp it to 0 and renormalize
    # This is a simple projection for points slightly outside.
    # A more accurate projection onto the convex hull would be better, but this is fast.
    if np.any(b < -eps):
        b = np.maximum(b, 0.0)
        s = b.sum()
        if s < 1e-12:
            return (V0 + V1 + V2) / 3.0
        b = b / s
    return _barycentric_to_cartesian(b)


def _clamp_points(points):
    """Clamp an array of points to the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])


def _min_triangle_area_fast(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    Uses vectorization for speed.
    """
    n = len(points)
    # We need to compute areas for all combinations (i, j, k)
    # To save memory and time, we can iterate over i, and vectorize j, k
    # But for n=11, there are C(11,3) = 165 triangles. 
    # Vectorizing all 165 at once is possible.
    
    # Create indices
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter valid indices i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Extract indices
    i = i_idx[mask]
    j = j_idx[mask]
    k = k_idx[mask]
    
    # Get coordinates
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    # Compute 2x2 determinant for area: 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)


def _min_triangle_area_normalized(points):
    """Return min triangle area divided by triangle area."""
    return _min_triangle_area_fast(points) / AREA


def _generate_initial_config(rng, strategy='structured'):
    """Generate an initial configuration of 11 points."""
    if strategy == 'structured':
        # A more symmetric initial guess often helps
        # Vertices (slightly inset)
        inset = 0.05
        p0 = V0 + inset * (V1 - V0) + inset * (V2 - V0)
        p1 = V1 + inset * (V0 - V1) + inset * (V2 - V1)
        p2 = V2 + inset * (V0 - V2) + inset * (V1 - V2)
        
        # Midpoints
        m01 = (V0 + V1) / 2.0
        m12 = (V1 + V2) / 2.0
        m20 = (V2 + V0) / 2.0
        
        # Center
        c = (V0 + V1 + V2) / 3.0
        
        # Additional points to fill gaps
        # Let's use a simple grid in barycentric coords
        points = [p0, p1, p2, m01, m12, m20, c]
        
        # Add 4 more points
        # Sample random points but ensure they are not too close to existing ones
        while len(points) < N:
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            w = 1.0 - u - v
            if w > 0.05:
                p = _barycentric_to_cartesian(np.array([u, v, w]))
                # Check distance to existing points
                min_dist = min(np.linalg.norm(p - q) for q in points)
                if min_dist > 0.05:
                    points.append(p)
        return np.array(points)
    elif strategy == 'random':
        points = []
        for _ in range(N):
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                p = [u + 0.5 * v, H * v]
                points.append(p)
            else:
                # Retry if outside
                points.append([rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.4)])
        return np.array(points)
    else:
        return _generate_initial_config(rng, 'random')


def _simulated_annealing(points_init, time_limit, rng):
    """
    Run Simulated Annealing to maximize the min triangle area.
    """
    start_time = time.time()
    current_points = points_init.copy()
    current_area = _min_triangle_area_fast(current_points)
    
    best_points = current_points.copy()
    best_area = current_area
    
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.999
    # Step size
    step_size = 0.05
    
    iterations = 0
    while time.time() < time_limit:
        # Generate a new candidate
        # Perturb a random subset of points
        num_points_to_move = rng.randint(1, 3)
        idxs = rng.sample(range(N), num_points_to_move)
        
        new_points = current_points.copy()
        for idx in idxs:
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] += noise
        
        # Clamp to triangle
        new_points = _clamp_points(new_points)
        
        new_area = _min_triangle_area_fast(new_points)
        
        # Acceptance probability
        delta = new_area - current_area
        if delta > 0:
            # Always accept better solutions
            current_points = new_points
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_points = current_points.copy()
        else:
            # Accept worse solutions with probability exp(delta / T)
            if T > 1e-6:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_area = new_area
        
        # Cooling
        T *= alpha
        # Occasionally reheat or adjust step size if stuck? 
        # For now, simple cooling.
        
        iterations += 1
        # If T is very low, we are in local refinement mode.
        # We can reduce step size as we cool
        step_size = max(0.001, 0.05 * (T / 0.01))

    return best_points, best_area


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points_global = None
    best_score_global = -float('inf')
    
    # Strategies to try
    strategies = ['structured', 'random']
    
    # Allocate time for each strategy
    # We will run multiple SA runs
    total_runs = 10
    time_per_run = (deadline - start_time) / total_runs - 5.0 # 5s buffer
    
    for i in range(total_runs):
        if time.time() > deadline - 10.0:
            break
            
        # Alternate strategies
        strategy = strategies[i % len(strategies)]
        
        # Generate initial config
        points_init = _generate_initial_config(rng, strategy)
        
        # Run SA
        run_deadline = time.time() + time_per_run
        points_sa, area_sa = _simulated_annealing(points_init, run_deadline, rng)
        
        score_sa = area_sa / AREA
        
        if score_sa > best_score_global:
            best_score_global = score_sa
            best_points_global = points_sa.copy()
            
    # Final polishing step
    if best_points_global is not None and time.time() < deadline - 5.0:
        # Use Nelder-Mead on the best result found by SA
        # This should be fast if we start near the optimum
        from scipy.optimize import minimize
        
        def objective(x):
            pts = x.reshape(-1, 2)
            pts = _clamp_points(pts)
            return -_min_triangle_area_fast(pts)
        
        x0 = best_points_global.flatten()
        
        # Short run of NM
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        points_nm = result.x.reshape(-1, 2)
        points_nm = _clamp_points(points_nm)
        score_nm = _min_triangle_area_normalized(points_nm)
        
        if score_nm > best_score_global:
            best_score_global = score_nm
            best_points_global = points_nm.copy()

    # Ensure final points are valid
    if best_points_global is not None:
        best_points_global = _clamp_points(best_points_global)
        
    return best_points_global


# EVOLVE-BLOCK-START
# (The construct function is defined above, but we keep this block for compatibility)
# EVOLVE-BLOCK-END
