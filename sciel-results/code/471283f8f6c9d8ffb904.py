import numpy as np
from scipy.optimize import minimize

N = 50

def compute_score(points):
    """
    Compute the score: min(min_pair_dist/2, min_wall_dist)
    """
    # min wall distance
    norms = np.linalg.norm(points, axis=1)
    wall_dists = 1.0 - norms
    min_wall = np.min(wall_dists)
    
    # min pair distance
    min_pair = 1e9
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
                
    return min(min_pair / 2.0, min_wall)

def objective(params):
    """
    Objective function for minimization.
    We use a repulsion-based energy model.
    Energy = sum of pairwise repulsion + wall repulsion.
    """
    points = params.reshape(-1, 2)
    
    # Calculate norms for wall distance
    norms = np.linalg.norm(points, axis=1)
    
    # Wall repulsion: penalize points close to or outside the boundary
    # Use a soft potential to avoid division by zero
    # dist_to_wall = 1.0 - norms
    # If norm > 1, dist is negative. We want strong penalty.
    # Potential: 1 / (dist + eps)^2 if dist > 0, else (1 - dist)^2 * large_factor
    eps = 1e-6
    wall_dists = 1.0 - norms
    # Clip to avoid negative base if we square, but we want strong penalty for outside
    # Better: if wall_dist < eps, penalty is huge.
    # Let's use: penalty = 1 / (wall_dist + eps)**2
    # This grows as 1/x^2 as x->0.
    wall_energy = np.sum(1.0 / (wall_dists + eps)**2)
    
    # Pairwise repulsion
    pair_energy = 0.0
    for i in range(N):
        # Vectorized distance calculation for j > i
        diffs = points[i+1:] - points[i]
        dists_sq = np.sum(diffs**2, axis=1)
        # Avoid division by zero
        dists_sq = np.maximum(dists_sq, eps**2)
        pair_energy += np.sum(1.0 / dists_sq)
        
    return wall_energy + pair_energy

def construct():
    # Strategy: Multi-start local optimization with Repulsion Energy
    best_points = None
    best_score = -1e9
    
    inits = []
    
    # 1. Fibonacci Sphere / Spiral Initialization
    # This provides a very even distribution
    N_init = N
    golden_ratio = (1 + np.sqrt(5)) / 2
    angles = 2 * np.pi * np.arange(N_init) / golden_ratio
    # Map to 2D disk using area-preserving mapping? 
    # Simpler: Use a spiral in 2D
    # r = sqrt(k/N) * R, theta = 2*pi*k/N * twist
    k = np.arange(N_init)
    radii = np.sqrt((k + 0.5) / N_init) * 0.95 # Slightly inside to start
    thetas = 2 * np.pi * k / np.sqrt(N_init) # Archimedean spiral
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    inits.append(np.column_stack((x, y)))
    
    # 2. Hexagonal Lattice (Refined)
    for spacing in [0.2, 0.22, 0.24, 0.26]:
        points = []
        k = 10
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 0.9: # Keep well inside
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Select N points closest to center or random
            dists = np.linalg.norm(points, axis=1)
            points = points[np.argsort(dists)[:N]]
            inits.append(points)
        elif len(points) > 0:
             # If not enough, pad with random
             random_pts = np.random.default_rng(42).uniform(-0.5, 0.5, size=(N - len(points), 2))
             points = np.vstack([points, random_pts])
             inits.append(points)

    # 3. Random Uniform
    rng = np.random.default_rng(123)
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        while True:
            pts = rng.uniform(-1, 1, size=(N, 2))
            norms = np.linalg.norm(pts, axis=1)
            if np.all(norms <= 0.8): # Start well inside
                break
        inits.append(pts)
        
    # Run optimization for each initial guess
    for init_pts in inits:
        params = init_pts.flatten()
        
        try:
            # Use Powell's method. It is derivative-free and handles the non-smooth-ish landscape of repulsion energy better than Nelder-Mead for higher dimensions.
            # Maxiter must be high enough. 
            # N=50 -> 100 vars. Powell is O(N) function evals per iteration cycle.
            res = minimize(objective, params, method='Powell', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'maxfev': 50000})
            opt_pts = res.x.reshape(-1, 2)
            
            # Check validity
            if np.all(np.isfinite(opt_pts)):
                # Ensure points are inside (with small tolerance)
                norms = np.linalg.norm(opt_pts, axis=1)
                if np.max(norms) < 1.0:
                    score = compute_score(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts
        except Exception:
            continue
            
    # Fallback: if all optimizations failed or returned bad results, return the best init
    if best_points is None:
        best_points = inits[0]
        best_score = compute_score(best_points)
        
    return best_points
