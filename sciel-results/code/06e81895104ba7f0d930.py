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
    
    # If any point is outside, score is effectively 0 or negative
    # But we add a penalty in the objective function for optimization.
    # For the final check, we just return the value.
    
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
    We want to maximize the score, so we minimize -score.
    We also add a penalty for points outside the container to help the optimizer stay in bounds.
    """
    points = params.reshape(-1, 2)
    
    # Calculate raw score
    norms = np.linalg.norm(points, axis=1)
    min_wall = np.min(1.0 - norms)
    
    min_pair = 1e9
    # Vectorized pair distance calculation for speed
    # N=50 is small, loop is fine, but let's try to be efficient
    for i in range(N):
        diffs = points[i+1:] - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) > 0:
            min_pair = min(min_pair, np.min(dists))
            
    score = min(min_pair / 2.0, min_wall)
    
    # Penalty for being outside
    outside_mask = norms > 1.0
    penalty = 0.0
    if np.any(outside_mask):
        penalty = np.sum(norms[outside_mask] - 1.0) * 10.0 # Strong penalty
        
    return -score + penalty

def construct():
    # Strategy: Multi-start local optimization
    # We try several initial configurations and run a local optimizer on each.
    
    best_points = None
    best_score = -1e9
    
    # Initial guesses
    inits = []
    
    # 1. Hexagonal lattice with different spacings
    for spacing in [0.18, 0.20, 0.22, 0.24, 0.26]:
        points = []
        k = 10
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 1.0:
                    points.append([x, y])
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        points = points[np.argsort(dists)[:N]]
        inits.append(points)
        
    # 2. Random uniform points (several seeds)
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3]:
        rng = np.random.default_rng(seed)
        # Generate random points in unit circle
        while True:
            pts = rng.uniform(-1, 1, size=(N, 2))
            norms = np.linalg.norm(pts, axis=1)
            if np.all(norms <= 1.0):
                break
        inits.append(pts)
        
    # 3. Ring-based initialization
    # 1 center, 6, 12, 12, 9, 10 ?
    # Let's try a generic ring init
    radii = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    counts = [1, 6, 12, 12, 9, 10]
    points = []
    angle_offset = 0
    for r, count in zip(radii, counts):
        if count == 0: continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + angle_offset
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
        angle_offset += np.pi / count # Shift next ring
    points = np.array(points[:N])
    inits.append(points)

    # Run optimization for each initial guess
    for init_pts in inits:
        params = init_pts.flatten()
        
        # Use Powell's method as it handles non-smooth objectives reasonably well
        # and doesn't require gradients.
        # Maxiter needs to be tuned. 120s limit.
        # N=50 -> 100 variables. Powell might be slow.
        # Let's try Nelder-Mead with maxiter=5000.
        
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            opt_pts = res.x.reshape(-1, 2)
            
            # Check validity
            if np.all(np.isfinite(opt_pts)):
                score = compute_score(opt_pts)
                # Ensure points are inside (with small tolerance)
                if np.max(np.linalg.norm(opt_pts, axis=1)) < 1.0 + 1e-9:
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts
        except Exception:
            continue
            
    # Fallback: if all optimizations failed or returned bad results, return the best init
    if best_points is None:
        # Just return the first init
        best_points = inits[0]
        best_score = compute_score(best_points)
        
    return best_points
