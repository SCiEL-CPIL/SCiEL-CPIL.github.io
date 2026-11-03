import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if points.shape[0] < 3:
        return 0.0
    
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(centered)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas
    # Indices for all triples
    # n=13, C(13,3) = 286 triangles
    # We can vectorize this
    i = np.arange(N)
    # Create all combinations of 3 indices
    # This might be memory heavy if N is large, but for 13 it's fine
    # Using itertools or manual loops is safer for clarity, but vectorized is faster
    # Let's use a simple loop for clarity and safety, or precompute indices
    
    # Precompute indices
    indices = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                indices.append((a, b, c))
    indices = np.array(indices)
    
    # Get points for each triangle
    p1 = centered[indices[:, 0]]
    p2 = centered[indices[:, 1]]
    p3 = centered[indices[:, 2]]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(params):
    """
    Objective function for minimization.
    params: (2N,) array
    """
    points = params.reshape(N, 2)
    # Normalize scale to prevent drift
    # Fix mean squared radius to 1
    r_sq = np.sum(points**2)
    if r_sq < 1e-10:
        return 1.0
    points = points / np.sqrt(r_sq)
    
    score = _compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Define initial guesses
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = np.ones(N)
    init_regular = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    
    # 2. Vogel Spiral
    i = np.arange(N)
    r_vogel = np.sqrt(i / N)
    theta_vogel = i * 2.399963229728653
    init_vogel = np.column_stack((r_vogel * np.cos(theta_vogel), r_vogel * np.sin(theta_vogel)))
    
    # 3. 12-gon + center
    angles_12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    r_12 = np.ones(12)
    pts_12 = np.column_stack((r_12 * np.cos(angles_12), r_12 * np.sin(angles_12)))
    init_12c = np.vstack([pts_12, [[0.0, 0.0]]])
    
    # 4. Random
    rng = np.random.default_rng(42)
    init_random = rng.random((N, 2)) * 2 - 1
    
    inits = [init_regular, init_vogel, init_12c, init_random]
    
    # Add a few more random inits if time permits
    for _ in range(5):
        if time.time() > deadline:
            break
        inits.append(rng.random((N, 2)) * 2 - 1)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Normalize initial points
        init_centered = init - np.mean(init, axis=0)
        r_sq = np.sum(init_centered**2)
        if r_sq > 1e-10:
            init_centered = init_centered / np.sqrt(r_sq)
        else:
            continue
            
        params0 = init_centered.flatten()
        
        # Use Nelder-Mead for local optimization
        try:
            res = minimize(_objective, params0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
        except Exception:
            pass
        
        # Also try Powell method which can be better for this
        if time.time() > deadline:
            break
        try:
            res_powell = minimize(_objective, params0, method='Powell', 
                                  options={'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-8})
            if res_powell.fun < -best_score:
                score = -res_powell.fun
                if score > best_score:
                    best_score = score
                    best_points = res_powell.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        best_points = init_regular
        best_score = _compute_score(best_points)
        
    # Final normalization: Ensure the output is in a standard form
    # The score is invariant to translation and scaling, but we should return
    # points that are well-behaved. The evaluator normalizes by hull area.
    # We can just return the points.
    
    return best_points
# EVOLVE-BLOCK-END
