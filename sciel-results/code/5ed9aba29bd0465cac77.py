"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def get_score(pts):
    """Calculate the squared ratio of max to min pairwise distance."""
    dists = pdist(pts)
    if np.any(dists < 1e-10):
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d == 0:
        return 1e10
    return (max_d / min_d) ** 2

def objective(x):
    """Flattened objective function for optimizers."""
    pts = x.reshape(N, D)
    return get_score(pts)

def local_refinement(pts, maxiter=1000):
    """Perform local refinement using Powell's method."""
    x0 = pts.flatten()
    
    # Powell is a derivative-free local optimizer
    res = minimize(objective, x0, method='Powell', 
                   options={'maxiter': maxiter, 'xtol': 1e-8, 'ftol': 1e-8})
    
    opt_points = res.x.reshape(N, D)
    return opt_points

def construct():
    rng = np.random.default_rng(42)
    
    # Define bounds for the points. 
    # Since the problem is scale-invariant (ratio of distances), 
    # we can constrain the points to a reasonable box, e.g., [-5, 5]^3.
    # The optimizer will scale/shift as needed, but bounds help DE converge.
    bounds = [(-5.0, 5.0)] * (N * D)
    
    best_pts = None
    best_score = float('inf')
    
    # Run Differential Evolution
    # maxiter is the number of generations. 
    # popsize is the factor for population size.
    # We use a moderate number of iterations to fit within time limits.
    # tol=1e-6 is sufficient for the global phase.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=10, 
            tol=1e-5,
            polish=False  # We will do our own refinement
        )
        de_pts = res.x.reshape(N, D)
        de_score = get_score(de_pts)
        
        # Local refinement from DE result
        refined_pts = local_refinement(de_pts, maxiter=500)
        refined_score = get_score(refined_pts)
        
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts
        elif de_score < best_score:
            best_score = de_score
            best_pts = de_pts
            
    except Exception as e:
        pass

    # If DE failed or we want to try another strategy, we can add a fallback
    # But DE is generally robust. Let's add a second run with different parameters if time permits?
    # Given 120s limit, 1 DE run + Powell should be fast enough.
    # Let's try a second DE run with different seed/popsize if the first one is not "perfect"?
    # Actually, let's just do one robust run.
    
    if best_pts is None:
        # Fallback to random if something went wrong
        best_pts = rng.normal(0, 1, size=(N, D))
        best_score = get_score(best_pts)

    return best_pts

# EVOLVE-BLOCK-END
