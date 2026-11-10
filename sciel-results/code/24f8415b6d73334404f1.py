import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import sys

def compute_score(points):
    """
    Calculate the score: product of distances / 8^8, scaled to diameter 2.
    Returns log(score) to avoid overflow.
    """
    # points shape (8, 2)
    # Compute pairwise distances
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Get unique pairs i < j
    # Mask upper triangle
    mask = np.triu(np.ones((8, 8), dtype=bool), k=1)
    unique_dists = dists[mask]
    
    if np.any(unique_dists < 1e-10):
        return -np.inf
        
    D = np.max(unique_dists)
    
    if D < 1e-10:
        return -np.inf
        
    # Log of product of distances
    log_prod = np.sum(np.log(unique_dists))
    
    # The score is (prod / D^28) * (2^28 / 8^8) ?? 
    # Wait, the definition: "score is Delta/n^n, where Delta is the product over ordered pairs... after scaling the set to diameter 2"
    # Let P be the set. Let D(P) be diameter.
    # Scaled set P' = P * (2 / D(P)).
    # Delta(P') = product of distances in P'.
    # Delta(P') = Delta(P) * (2/D(P))^28.
    # Score = Delta(P') / 8^8.
    # We want to maximize Score.
    # This is equivalent to maximizing Delta(P) / D(P)^28.
    # So we maximize log(Delta(P)) - 28 log(D(P)).
    
    return log_prod - 28 * np.log(D)

def objective(x):
    """
    Objective function for minimization.
    x is a flat array of 16 floats.
    """
    points = x.reshape(8, 2)
    score = compute_score(points)
    if not np.isfinite(score):
        return 1e10
    return -score

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit
    
    # Bounds for coordinates
    # Since the problem is scale invariant in the objective formulation (we normalize by D),
    # the actual magnitude of coordinates doesn't matter for the score, only relative positions.
    # However, DE needs bounds. [-10, 10] is safe.
    bounds = [(-10, 10)] * 16
    
    # Regular octagon as a good starting point / seed
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = octagon.flatten()
    
    # Differential Evolution
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15,
            tol=1e-10,
            polish=True, # Use local optimizer at the end
            workers=1,
            init='latinhypercube'
        )
        
        best_points = result.x.reshape(8, 2)
        best_score = -result.fun
        
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)
        best_points = octagon
        best_score = compute_score(octagon)

    # Local Polish
    # Use L-BFGS-B on the best candidate found
    if time.time() - start_time < time_limit:
        try:
            res_local = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res_local.fun < result.fun: # If improvement
                best_points = res_local.x.reshape(8, 2)
                best_score = -res_local.fun
        except Exception as e:
            print(f"Local polish failed: {e}", file=sys.stderr)

    # Ensure distinct points and valid format
    # Check for duplicates
    # If any two points are too close, the score is effectively -inf (or very low)
    # We should return the best valid points.
    
    # Final check: ensure coordinates are within limits (they are by construction)
    # Return list of lists
    return [[float(p[0]), float(p[1])] for p in best_points]

if __name__ == "__main__":
    pts = construct()
    # Print score for debugging if needed
    # print(compute_score(np.array(pts)))
