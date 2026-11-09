import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def get_score(points):
    N = len(points)
    if N < 3:
        return 0.0
    
    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # We need to compute the area for all triples (i, j, k) with i < j < k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # Instead of loops, we can use indices.
    # But for N=13, a simple loop over i, j and vectorizing over k is already fast.
    # Let's try to be even faster by pre-computing differences or using full broadcasting if possible.
    # Full broadcasting creates a (13, 13, 13) array which is 2197 elements. That's tiny.
    
    # Let's do full broadcasting for maximum speed.
    # p_i, p_j, p_k
    # We want to avoid i=j, j=k, k=i.
    # And we want i < j < k to avoid duplicates and sign issues (though abs handles sign).
    
    # Actually, just computing all pairs and taking min is fine, but we must exclude i=j=k.
    # Let's stick to a slightly optimized loop for clarity and safety against memory spikes, 
    # but vectorize the innermost part.
    
    # Actually, the previous loop was:
    # for i: for j: for k:
    # This is 13*12*11 / 6 = 286 iterations.
    # Each iteration does a few float ops.
    # 286 * 10 ops = ~3000 ops.
    # In Python, this takes ~10-50 microseconds.
    # If we call this 1000 times, that's 0.05s.
    # The bottleneck in the original was likely the overhead of the function call and the Python loop overhead combined with the slow optimizer.
    
    # Let's try a pure numpy approach for the areas.
    # Create indices
    i, j, k = np.indices((N, N, N))
    # Mask where i < j < k
    mask = (i < j) & (j < k)
    
    # Extract points
    pi = points[i[mask]]
    pj = points[j[mask]]
    pk = points[k[mask]]
    
    # Cross product
    # (pj - pi) x (pk - pi)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Strategy: Powell's method with many restarts.
    # Powell is robust for non-smooth functions (like min of areas).
    # We will use multiple diverse initial configurations.
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. 12-gon + center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points_12 = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
    center = np.array([0.0, 0.0])
    points_init_2 = np.vstack([points_12, center])
    
    # 3. Two concentric rings (e.g., 6+7)
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6 # offset
    inner = 0.5 * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    points_init_3 = np.vstack([outer, inner])
    
    initial_configs = [points_init, points_init_2, points_init_3]
    
    # Evaluate initial configs
    for cfg in initial_configs:
        s = get_score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()
    
    # Optimization loop
    # We will run Powell on the 26-dimensional space.
    # To handle the scale invariance, we could fix one point or normalize, 
    # but Powell usually handles it okay if the function is scale-invariant 
    # (gradient is zero along scale direction, so it might drift, but score stays same).
    # Actually, since score is scale invariant, the landscape is flat along radial scaling.
    # This can cause issues. 
    # A common trick is to fix the centroid at origin and one point at a fixed location, 
    # or just let it be and hope the optimizer doesn't drift to infinity/zero.
    # Given the short time, let's just run Powell on the raw coordinates.
    
    n_restarts = 0
    max_restarts = 100 # Limit to ensure we don't run forever if Powell is fast
    
    while time.time() < deadline and n_restarts < max_restarts:
        n_restarts += 1
        
        # Choose starting point
        if n_restarts <= 3:
            # Use the diverse initial configs
            x_start = initial_configs[n_restarts - 1].flatten()
        elif n_restarts % 10 == 0:
            # Random restart
            # Mix of regular polygon and noise
            angle_noise = np.random.normal(0, 0.5, N)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + angle_noise
            pts = np.column_stack((np.cos(angles), np.sin(angles)))
            # Add some radial noise
            radii = 1.0 + np.random.normal(0, 0.2, N)
            pts[:, 0] *= radii
            pts[:, 1] *= radii
            x_start = pts.flatten()
        else:
            # Perturb best
            x_start = best_points.flatten() + np.random.normal(0, 0.2, 2*N)
        
        def neg_score_flat(x_flat):
            pts = x_flat.reshape(-1, 2)
            return -get_score(pts)
        
        try:
            # Powell method
            # maxiter controls the number of iterations.
            # We give it a decent budget per restart.
            res = minimize(neg_score_flat, x_start, method='Powell', 
                           options={'maxiter': 500, 'xtol': 1e-6, 'ftol': 1e-8})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(-1, 2)
                
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    return best_points
