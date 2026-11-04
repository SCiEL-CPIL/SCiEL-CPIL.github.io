"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
TRI_AREA = np.sqrt(3)/4

# Precompute indices for all triplets
indices = list(itertools.combinations(range(N), 3))
num_triplets = len(indices)

def _bary_to_cart(u, w):
    """
    Convert barycentric-like parameters to Cartesian.
    u in [0,1], w in [0,1].
    v = w * (1 - u)
    Point = u*V0 + v*V1 + (1-u-v)*V2
    """
    v = w * (1.0 - u)
    # Barycentric coords: (1-u-v, v, u) relative to V2, V1, V0? 
    # Standard: P = b0*V0 + b1*V1 + b2*V2 with sum=1.
    # Let's define:
    # b0 = 1 - u - v
    # b1 = v
    # b2 = u
    # Check: if u=0, v=0 -> P=V0. Correct.
    # if u=1, v=0 -> P=V2. Correct? 
    # My mapping:
    # u=0, w=0 -> v=0 -> P = 1*V0 + 0*V1 + 0*V2 = V0.
    # u=1, w=0 -> v=0 -> P = 0*V0 + 0*V1 + 1*V2 = V2.
    # u=0, w=1 -> v=1 -> P = 0*V0 + 1*V1 + 0*V2 = V1.
    # This covers the triangle.
    
    b0 = 1.0 - u - v
    b1 = v
    b2 = u
    return b0 * V0 + b1 * V1 + b2 * V2

def _get_areas(points):
    """
    Calculate areas of all triangles formed by the points.
    points: shape (N, 2)
    returns: array of shape (num_triplets,)
    """
    areas = np.zeros(num_triplets)
    for k, (i, j, k_idx) in enumerate(indices):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k_idx]
        # Cross product magnitude / 2
        area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        areas[k] = area
    return areas

def _min_area(points):
    """
    Calculate the minimum area of any triangle.
    """
    return np.min(_get_areas(points))

def _objective_flat(x, beta):
    """
    Objective function for optimization.
    x: flat array of length 2*N (u0, w0, u1, w1, ...)
    beta: temperature parameter for log-sum-exp approximation of min
    We minimize - (approx min area)
    """
    # Reshape to (N, 2) where col 0 is u, col 1 is w
    params = x.reshape(N, 2)
    u = params[:, 0]
    w = params[:, 1]
    
    # Convert to Cartesian
    points = np.array([_bary_to_cart(ui, wi) for ui, wi in zip(u, w)])
    
    areas = _get_areas(points)
    
    # Stable log-sum-exp for minimum:
    # min(a) approx (1/beta) * log(sum(exp(-beta * a)))
    # We want to minimize this, so we return it.
    # Wait, maximizing min area is equivalent to minimizing -min area.
    # LSE gives an approximation of min area.
    # So we return LSE? No, we want to MINIMIZE the objective.
    # If we return LSE, we are minimizing the approximate min area. That's wrong.
    # We want to MAXIMIZE min area.
    # So we should MINIMIZE - (approx min area).
    # approx_min = (1/beta) * log(sum(exp(-beta * areas)))
    # objective = - approx_min
    
    # Calculate LSE
    # Shift for numerical stability
    min_a = np.min(areas)
    # exp(-beta * (a - min_a)) is safe
    shifted_areas = areas - min_a
    exp_terms = np.exp(-beta * shifted_areas)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 0.0
        
    lse = (1.0 / beta) * np.log(sum_exp) + min_a
    # lse is the approximate minimum area.
    return -lse

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 110s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Multi-start strategy
    rng = np.random.default_rng(42)
    
    num_restarts = 30
    beta_values = [10, 50, 200] # Increase beta for sharper minimum
    
    # Bounds for u and w are [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial parameters (u, w)
        # Random inside the square [0,1]x[0,1]
        x0 = rng.random(2 * N)
        
        # Run continuation method
        current_x = x0
        for beta in beta_values:
            if time.time() > deadline:
                break
                
            # Use L-BFGS-B which supports box constraints
            res = minimize(
                _objective_flat, 
                current_x, 
                args=(beta,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 0: # res.fun is negative min area
                current_x = res.x
                # Convert back to Cartesian to check score
                params = current_x.reshape(N, 2)
                u = params[:, 0]
                w = params[:, 1]
                pts = np.array([_bary_to_cart(ui, wi) for ui, wi in zip(u, w)])
                
                # Verify points are valid (they should be by construction)
                score = _min_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            else:
                # If optimization failed, keep current_x or break
                pass
                
    # Final check and return
    if best_points is None:
        # Fallback: Generate random points in triangle
        # Using the same barycentric mapping
        x0 = rng.random(2 * N)
        params = x0.reshape(N, 2)
        u = params[:, 0]
        w = params[:, 1]
        best_points = np.array([_bary_to_cart(ui, wi) for ui, wi in zip(u, w)])
        
    return best_points
