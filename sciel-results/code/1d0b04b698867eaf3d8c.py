import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import time

N = 13

def construct():
    # Objective function for Differential Evolution
    def objective(params):
        # params shape (2*N,)
        pts = params.reshape(N, 2)
        
        # Check for degenerate points
        if np.any(np.isnan(pts)):
            return 1e6
            
        # Compute Convex Hull
        try:
            hull = ConvexHull(pts)
            hull_area = hull.area
        except Exception:
            return 1e6
            
        if hull_area < 1e-10:
            return 1e6
            
        # Calculate min triangle area
        min_area = np.inf
        # Vectorized area calculation for all triples is possible but 286 is small
        # Using loops for clarity and to avoid huge temporary arrays if needed, 
        # but vectorization is faster in Python.
        
        # Let's use a semi-vectorized approach
        # i, j, k indices
        # For N=13, 286 triangles.
        # We can compute all cross products.
        
        # Pairs of vectors
        # This might be memory heavy if not careful, but N=13 is small.
        
        # Simple loop is fast enough for 286 iterations inside the objective
        # which is called ~1000 times. 286 * 1000 = 286,000 ops. Trivial.
        
        for i in range(N):
            pi = pts[i]
            for j in range(i+1, N):
                pj = pts[j]
                v1 = pj - pi
                for k in range(j+1, N):
                    pk = pts[k]
                    v2 = pk - pi
                    # Cross product magnitude in 2D
                    area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
                        
        if min_area == np.inf:
            return 1e6
            
        score = min_area / hull_area
        return -score

    # Bounds for x, y
    bounds = [(-1, 1)] * (2 * N)
    
    # Run Differential Evolution
    # polish=True runs Nelder-Mead on the best result
    # maxiter controls the number of generations
    # popsize is the number of individuals
    start_time = time.time()
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            polish=True, 
            tol=1e-8,
            seed=42
        )
    except Exception:
        # Fallback to regular polygon if DE fails
        theta0 = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack((np.cos(theta0), np.sin(theta0)))
        return pts
        
    # Extract best points
    pts = res.x.reshape(N, 2)
    
    # Optional: One more local refinement pass if time permits
    # DE with polish already does this, but we can do a quick check.
    
    return pts
