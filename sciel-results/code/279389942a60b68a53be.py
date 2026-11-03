import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def compute_covering_radius_exact(points):
    """
    Computes the exact covering radius using the ConvexHull Voronoi vertices.
    This is used for final evaluation and verification, not for the main optimization loop
    due to computational cost and non-differentiability.
    """
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    faces = hull.simplices
    voronoi_vertices = []
    
    for face in faces:
        a = points[face[0]]
        b = points[face[1]]
        c = points[face[2]]
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        voronoi_vertices.append(o)
    
    if not voronoi_vertices:
        return np.pi
        
    voronoi_vertices = np.array(voronoi_vertices)
    
    max_dist = 0.0
    for p in points:
        dots = np.dot(voronoi_vertices, p)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dist = np.min(dists)
        if min_dist > max_dist:
            max_dist = min_dist
            
    return max_dist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Initial Guess: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    initial_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = initial_points.copy()
    best_exact_radius = compute_covering_radius_exact(best_points)
    
    # 2. Generate Probe Points for Smooth Approximation
    # We use a dense set of points to approximate the "max over sphere"
    # K=2000 is a good balance between accuracy and speed
    K = 2000
    i_probe = np.arange(K) + 0.5
    z_probe = 1.0 - 2.0 * i_probe / K
    phi_probe = np.pi * (1.0 + np.sqrt(5.0)) * i_probe
    r_probe = np.sqrt(1.0 - z_probe * z_probe)
    probe_points = np.column_stack([r_probe * np.cos(phi_probe), r_probe * np.sin(phi_probe), z_probe])
    
    # Pre-compute nothing, but we will use vectorized operations
    
    # 3. Define Smooth Objective Function
    # Objective: Soft-Max of (Min Angular Distance from each probe to any site)
    # Soft-Max parameter beta controls how "hard" the max is.
    # Higher beta -> closer to true max, but less smooth (potentially).
    # We start with a moderate beta and might increase it if time permits.
    
    def objective_smooth(x):
        if time.time() > deadline:
            return best_exact_radius
            
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        # Compute angular distances between probes (Kx3) and points (Nx3)
        # cos(theta) = dot(p_probe, p_point)
        # We want min theta, which is max cos(theta)
        # dist = arccos(cos)
        
        # Vectorized dot product: (K, 1, 3) dot (1, N, 3) -> (K, N)
        dots = np.einsum('ij, kj -> ik', probe_points, pts)
        dots = np.clip(dots, -1.0, 1.0)
        
        # Angular distances
        dists = np.arccos(dots)
        
        # Min distance for each probe
        min_dists = np.min(dists, axis=1)
        
        # Soft-Max approximation of the max of min_dists
        # Using log-sum-exp trick for stability
        beta = 50.0  # High enough to approximate max closely
        # max_dists = -1/beta * log(sum(exp(-beta * min_dists)))
        # To avoid underflow/overflow:
        m = np.min(min_dists)
        shifted = min_dists - m
        sum_exp = np.sum(np.exp(-beta * shifted))
        if sum_exp == 0:
            return m # Fallback
        soft_max_val = m - (1.0 / beta) * np.log(sum_exp)
        
        return soft_max_val
    
    # 4. Optimization Phase 1: L-BFGS-B with Smooth Objective
    # Use multiple restarts if time permits
    current_points = initial_points.flatten()
    
    # We can try to increase beta in later stages if we have time
    # But let's stick to one robust run first
    
    try:
        res = minimize(
            objective_smooth, 
            current_points, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.success or res.fun < best_exact_radius:
            new_points = res.x.reshape(N, 3)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_exact_radius = compute_covering_radius_exact(new_points)
            
            if new_exact_radius < best_exact_radius:
                best_points = new_points
                best_exact_radius = new_exact_radius
                current_points = new_points.flatten()
                
    except Exception as e:
        pass # Keep best so far
    
    # 5. Optimization Phase 2: Refinement with Higher Beta or Local Search
    # If we have time, we can try to polish the result.
    # A common technique is to use the current best as a start for a new optimization
    # with a higher beta (sharper max) or a different initial noise.
    
    if time.time() < deadline - 20.0:
        # Try a few random restarts from the current best
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb current best
            noise = np.random.normal(0, 0.05, (N, 3))
            perturbed = best_points + noise
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            try:
                res2 = minimize(
                    objective_smooth, 
                    perturbed.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'maxfun': 2000}
                )
                
                if res2.success or res2.fun < best_exact_radius:
                    new_points = res2.x.reshape(N, 3)
                    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                    new_exact_radius = compute_covering_radius_exact(new_points)
                    
                    if new_exact_radius < best_exact_radius:
                        best_points = new_points
                        best_exact_radius = new_exact_radius
                        
            except Exception:
                pass
    
    # 6. Final Polish (Optional)
    # If time is very tight, we might just return. 
    # The smooth objective is a good proxy, but the exact radius might be slightly higher.
    # We could do a few steps of Nelder-Mead on the exact radius, but it's slow.
    # Given the 150s limit, we prioritize the L-BFGS-B result.
    
    # Ensure output is normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
