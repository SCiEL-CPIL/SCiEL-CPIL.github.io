import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _sphere_to_cartesian(theta_phi):
    """
    Convert spherical coordinates (theta, phi) to Cartesian coordinates on the unit sphere.
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2*pi]
    """
    x = np.sin(theta_phi[:, 0]) * np.cos(theta_phi[:, 1])
    y = np.sin(theta_phi[:, 0]) * np.sin(theta_phi[:, 1])
    z = np.cos(theta_phi[:, 0])
    return np.column_stack([x, y, z])

def _smooth_objective(theta_phi):
    """
    Objective function for differential evolution.
    We minimize a combination of distance variance and the max/min ratio penalty.
    The points are constrained to the unit sphere via parameterization.
    """
    pts = _sphere_to_cartesian(theta_phi)
    
    # Calculate pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    mn = d.min()
    mx = d.max()
    
    if mn < 1e-9:
        return 1e10
        
    # The true score is (mx/mn)^2. 
    # To make it smooth for the optimizer, we can minimize:
    # 1. The variance of distances (encourages uniformity)
    # 2. A penalty if mx is too large relative to mn
    
    # A robust proxy: minimize (mx - mn)^2 + epsilon * (mx/mn)
    # Or simply minimize the ratio directly but it's non-smooth at ties.
    # Let's try minimizing the sum of squared deviations from the mean distance,
    # plus a term that penalizes large spread.
    
    mean_d = d.mean()
    variance = np.var(d)
    
    # The ratio R = mx/mn. We want R close to 1.
    # (R-1)^2 is a good penalty.
    ratio = mx / mn
    penalty = (ratio - 1.0) ** 2
    
    # Combine: primarily minimize variance (smooth), secondarily minimize ratio deviation
    return 100.0 * variance + penalty

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    # Parameters for differential evolution
    # We are optimizing 2*N parameters (theta, phi for each point)
    bounds = [(0, np.pi), (0, 2 * np.pi)] * N
    
    # Initial population size
    # Max iterations
    # Tolerance
    
    best_result = None
    best_score = np.inf
    best_pts = None
    
    # Use multiple runs with different seeds to avoid local minima
    # DE is stochastic.
    
    seeds = [42, 123, 456, 789, 101]
    
    for seed in seeds:
        if time.time() - _t0 > DEADLINE - 10:
            break
            
        try:
            result = differential_evolution(
                _smooth_objective,
                bounds,
                seed=seed,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=True,  # Use L-BFGS-B for final polish
                disp=False
            )
        except Exception:
            continue
            
        if result.success or result.fun < 1e10:
            theta_phi = result.x
            pts = _sphere_to_cartesian(theta_phi)
            
            # Calculate true score
            score = _true_score(pts)
            
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
                best_result = result
                
    # If DE didn't find a good solution, try a local refinement from a random start
    if best_pts is None or best_score > 2.0:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() - _t0 > DEADLINE - 5:
                break
            # Random spherical points
            theta = rng.uniform(0, np.pi, N)
            phi = rng.uniform(0, 2 * np.pi, N)
            theta_phi = np.column_stack([theta, phi])
            pts = _sphere_to_cartesian(theta_phi)
            
            score = _true_score(pts)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
                
    # Final local refinement using L-BFGS-B on the Cartesian coordinates 
    # constrained to the sphere? No, that's hard.
    # Instead, we can do a simple gradient step on the angular coordinates if we have time.
    # But DE with polish=True already did a local refinement.
    
    # One more check: if we have time, try to improve the best point found by DE
    # using a more aggressive local search on the angular params.
    if best_pts is not None and time.time() - _t0 < DEADLINE - 2:
        # Re-derive theta/phi from best_pts? 
        # It's safer to keep the theta_phi from the best DE run if we stored it.
        # But we only stored pts. 
        # Let's just return the best pts found. The sphere constraint is maintained.
        
        # Optional: Center the points? 
        # Points on a unit sphere are already centered at origin (centroid is 0 if symmetric, 
        # but not necessarily for arbitrary subsets). 
        # The problem allows translation invariance. 
        # Translating the sphere off-center might improve the ratio? 
        # For N=14, the optimal is likely a subset of a spherical design or similar.
        # Translating a sphere doesn't change the pairwise distances. 
        # So the position relative to origin doesn't matter for the score, 
        # only the shape. The unit sphere parameterization fixes the scale.
        
        pass

    if best_pts is None:
        # Fallback: Fibonacci sphere
        i = np.arange(N)
        phi_angle = np.arccos(1 - 2 * (i + 0.5) / N)
        theta_angle = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi_angle) * np.cos(theta_angle)
        y = np.sin(phi_angle) * np.sin(theta_angle)
        z = np.cos(phi_angle)
        best_pts = np.column_stack([x, y, z])
        
    return best_pts
