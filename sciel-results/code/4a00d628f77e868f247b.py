import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Define ring structure: 1, 6, 12, 18, 13
    ring_sizes = [1, 6, 12, 18, 13]
    num_rings = len(ring_sizes)
    
    # Initial guess for radii and phases
    # Radii should be roughly evenly spaced, but outer rings need more space?
    # Actually, for equal circles, the radial spacing is roughly 2*r.
    # If r ~ 0.125, spacing ~ 0.25.
    # R1 ~ 0.25, R2 ~ 0.5, R3 ~ 0.75, R4 ~ 0.95
    init_radii = [0.0, 0.25, 0.5, 0.75, 0.95]
    init_phases = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Variables to optimize: R1..R4, phi1..phi4
    # We fix R0=0, phi0=0
    x0 = np.array(init_radii[1:] + init_phases[1:])
    
    def get_points(x):
        radii = np.concatenate(([0.0], x[:num_rings-1]))
        phases = np.concatenate(([0.0], x[num_rings-1:]))
        points = []
        for i, (r, phi, n) in enumerate(zip(radii, phases, ring_sizes)):
            if n == 0:
                continue
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phi
            if n == 1:
                angles = np.array([0.0]) # Center is just one point
            pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            points.append(pts)
        return np.vstack(points)
    
    def objective(x):
        if time.time() > deadline:
            return 1e10 # Penalize timeout
        pts = get_points(x)
        # Calculate all pairwise distances
        # This is O(N^2) which is 2500 ops, very fast
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Calculate boundary distances
        norms = np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(1.0 - norms)
        
        # Soft min of min_pair_dist/2 and min_bound_dist
        # We want to maximize r = min(min_pair_dist/2, min_bound_dist)
        # Let a = min_pair_dist/2, b = min_bound_dist
        # Soft min: -1/k * log(exp(-k*a) + exp(-k*b))
        k = 50.0
        a = min_pair_dist / 2.0
        b = min_bound_dist
        soft_min = -1.0/k * np.log(np.exp(-k*a) + np.exp(-k*b))
        
        return -soft_min # Minimize negative soft max
    
    # Run optimization
    best_result = None
    best_score = -np.inf
    
    # Try a few different initializations or just one good one
    # BFGS is good for this smooth problem
    try:
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-6})
        if time.time() < deadline:
            # Evaluate actual score
            pts = get_points(res.x)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            r1 = np.min(dists) / 2.0
            r2 = np.min(1.0 - np.linalg.norm(pts, axis=1))
            score = min(r1, r2)
            
            if score > best_score:
                best_score = score
                best_result = pts.copy()
                
        # If time allows, try a second pass with a different initial phase or radius perturbation
        if time.time() < deadline - 10.0:
             # Perturb initial guess slightly
             x0_perturbed = x0 + 0.01 * np.random.randn(len(x0))
             # Ensure radii are positive
             x0_perturbed[:num_rings-1] = np.abs(x0_perturbed[:num_rings-1])
             res2 = minimize(objective, x0_perturbed, method='BFGS', options={'maxiter': 500, 'gtol': 1e-6})
             if time.time() < deadline:
                 pts2 = get_points(res2.x)
                 diff = pts2[:, np.newaxis, :] - pts2[np.newaxis, :, :]
                 dists = np.linalg.norm(diff, axis=2)
                 np.fill_diagonal(dists, np.inf)
                 r1 = np.min(dists) / 2.0
                 r2 = np.min(1.0 - np.linalg.norm(pts2, axis=1))
                 score2 = min(r1, r2)
                 
                 if score2 > best_score:
                     best_score = score2
                     best_result = pts2.copy()
                     
    except Exception:
        pass
        
    if best_result is None:
        # Fallback to initial guess
        best_result = get_points(x0)
        
    return best_result
