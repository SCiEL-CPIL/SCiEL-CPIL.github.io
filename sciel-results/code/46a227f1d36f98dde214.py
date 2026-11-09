import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def get_points(coeffs):
    """
    Reconstruct points from radial coefficients.
    coeffs: array of coefficients for r(theta) = 1 + sum c_k cos(k * theta)
    We use fixed angular steps theta_i = 2*pi*i/20.
    """
    n = 20
    thetas = 2 * np.pi * np.arange(n) / n
    
    # Start with radius 1
    r = np.ones(n)
    
    # Add cosine terms
    # We assume coeffs are for k=1, 2, ...
    for k, c in enumerate(coeffs, start=1):
        r += c * np.cos(k * thetas)
        
    # Ensure r is positive to maintain convexity/star-shape
    # If r becomes <= 0, the point crosses the origin or flips, breaking convexity.
    # We can clip or let the optimizer handle it via penalty.
    # For robustness in get_points, we just compute.
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    
    return np.column_stack((x, y))

def vertex_score(v_idx, points):
    """
    Calculate score for vertex v_idx.
    """
    # Distances to all other points
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf # Exclude self
    
    # Sort 19 distances
    sorted_dists = np.sort(dists)
    
    # Check for repeated points (dist < 0.05)
    if sorted_dists[0] < 0.05:
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    # d_0...d_18
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    # gaps length 18
    # "first index minimizing"
    # If multiple, take the first one.
    min_gap = np.min(gaps)
    k = np.argmin(gaps)
    
    # D = (d_k + d_{k+1}) / 2
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19] # d_15..d_18
    else:
        d_sel = sorted_dists[k-1:k+3] # d_{k-1}..d_{k+2}
        
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def compute_min_vertex_score(points):
    """
    Compute the minimum vertex score for a given set of points.
    """
    n = len(points)
    scores = np.array([vertex_score(i, points) for i in range(n)])
    return np.min(scores)

def objective(coeffs):
    """
    Objective function for optimizer: negative of the minimum vertex score.
    Includes a penalty for non-convexity or invalid points.
    """
    points = get_points(coeffs)
    
    # Check for finite points
    if not np.all(np.isfinite(points)):
        return 1e6
        
    # Check convexity
    # For a star-shaped polygon with fixed angular order, convexity is guaranteed
    # if the radius function is such that the polygon doesn't self-intersect and
    # turns consistently.
    # A simple check: compute cross products of consecutive edges.
    # For convex CCW polygon, cross products should be all positive (or all negative).
    
    n = len(points)
    valid = True
    signs = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-9:
            # Degenerate, not strictly convex
            valid = False
            break
        signs.append(cross)
        
    if not valid:
        return 1e6
        
    # Check if all signs are the same
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        # Not strictly convex
        return 1e6
        
    # Also check for repeated points
    min_dist = np.inf
    # O(N^2) check, N=20 is small
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 0.05:
        return 1e6
        
    score = compute_min_vertex_score(points)
    
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    # Define number of harmonics
    # Too many harmonics might overfit or cause non-convexity easily.
    # Start with 4 harmonics: c1, c2, c3, c4
    num_coeffs = 4
    
    # Bounds for coefficients
    # Keep them small to ensure convexity
    bounds = [(-0.2, 0.2) for _ in range(num_coeffs)]
    
    best_points = get_points(np.zeros(num_coeffs))
    best_score = -objective(np.zeros(num_coeffs))
    
    # 1. Differential Evolution for global search
    # DE is stochastic, so set seed for reproducibility if needed, but random is fine.
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-4,
            mutation=(0.3, 1.2),
            recombination=0.9,
            polish=False # We will polish manually
        )
        
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_points = get_points(res_de.x)
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_points
        
    # 2. Local refinement with Nelder-Mead
    # Use the best point from DE (or regular polygon if DE failed) as start
    x0 = res_de.x if hasattr(res_de, 'x') and res_de.x is not None else np.zeros(num_coeffs)
    
    try:
        res_nm = minimize(
            objective, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_points = get_points(res_nm.x)
            
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline:
        return best_points
        
    # 3. Try a few random restarts with local optimization if time permits
    # This helps escape local minima.
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        x_rand = np.random.uniform(-0.1, 0.1, num_coeffs)
        try:
            res_rand = minimize(
                objective, 
                x_rand, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_rand.fun < best_score:
                best_score = res_rand.fun
                best_points = get_points(res_rand.x)
                
        except Exception:
            pass
            
    return best_points

if __name__ == "__main__":
    pts = equidistant_points_polygon20()
    print(pts)
