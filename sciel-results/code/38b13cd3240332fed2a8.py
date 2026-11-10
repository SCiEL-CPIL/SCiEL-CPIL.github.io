import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 630.0
    SAFETY_MARGIN = 30.0
    N = 16

    def get_cost(params):
        """
        Minimize d_max^2 subject to d_min^2 >= 1.
        Uses a penalty method for the constraint.
        """
        pts = params.reshape(N, 2)
        # Compute pairwise squared Euclidean distances
        # pdist returns condensed distance matrix
        dists_sq = pdist(pts, metric='sqeuclidean')
        
        d_max_sq = np.max(dists_sq)
        d_min_sq = np.min(dists_sq)
        
        # Penalty for violations of d_min^2 >= 1
        # violation = max(0, 1 - d_ij^2)
        violations = np.maximum(0, 1.0 - dists_sq)
        penalty = np.sum(violations * violations)
        
        # Large weight to ensure feasibility is prioritized over minimizing d_max
        # If penalty is 0, we minimize d_max^2
        # If penalty > 0, we strongly penalize it
        return d_max_sq + 1e6 * penalty

    def get_score(params):
        """Calculate the actual score R = (d_min/d_max)^2"""
        pts = params.reshape(N, 2)
        dists_sq = pdist(pts, metric='sqeuclidean')
        d_max_sq = np.max(dists_sq)
        d_min_sq = np.min(dists_sq)
        if d_max_sq == 0:
            return 0.0
        return (d_min_sq / d_max_sq)

    best_result = None
    best_cost = np.inf
    best_score = -np.inf

    # DE Bounds
    # For 16 points with min dist 1, max dist is roughly < 6
    # So [-5, 5] is a safe bound
    bounds = [(-5.0, 5.0)] * (N * 2)

    # 1. Global Search with Differential Evolution
    try:
        de_res = differential_evolution(
            get_cost, 
            bounds, 
            maxiter=100, 
            popsize=40, 
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        
        if de_res.fun < best_cost:
            best_cost = de_res.fun
            best_result = de_res.x
            best_score = get_score(best_result)
            
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)

    # Fallback if DE failed or returned poor result
    if best_result is None or best_cost > 100: # 100 is arbitrary threshold for "bad"
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)])
        best_cost = get_cost(best_result)
        best_score = get_score(best_result)

    # 2. Local Polish with Nelder-Mead
    time_elapsed = time.time() - start_time
    if time_elapsed < TIME_LIMIT - SAFETY_MARGIN - 10:
        try:
            nm_res = minimize(
                get_cost, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if nm_res.fun < best_cost:
                best_cost = nm_res.fun
                best_result = nm_res.x
                best_score = get_score(best_result)
        except Exception as e:
            print(f"NM failed: {e}", file=sys.stderr)

    # 3. Second Global Search with different seed if time permits
    time_elapsed = time.time() - start_time
    if time_elapsed < TIME_LIMIT - SAFETY_MARGIN - 60:
        try:
            de_res2 = differential_evolution(
                get_cost, 
                bounds, 
                maxiter=50, 
                popsize=30, 
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=123,
                polish=False,
                init='latinhypercube'
            )
            
            if de_res2.fun < best_cost:
                best_cost = de_res2.fun
                best_result = de_res2.x
                best_score = get_score(best_result)
                
                # Polish the new result
                time_elapsed = time.time() - start_time
                if time_elapsed < TIME_LIMIT - SAFETY_MARGIN - 10:
                    nm_res2 = minimize(
                        get_cost, 
                        best_result, 
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if nm_res2.fun < best_cost:
                        best_cost = nm_res2.fun
                        best_result = nm_res2.x
                        best_score = get_score(best_result)
                        
        except Exception as e:
            print(f"Second DE failed: {e}", file=sys.stderr)

    # 4. Final Polish
    time_elapsed = time.time() - start_time
    if time_elapsed < TIME_LIMIT - SAFETY_MARGIN - 5:
        try:
            nm_final = minimize(
                get_cost, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if nm_final.fun < best_cost:
                best_cost = nm_final.fun
                best_result = nm_final.x
                best_score = get_score(best_result)
        except Exception as e:
            print(f"Final NM failed: {e}", file=sys.stderr)

    # Final check for validity
    if not np.all(np.isfinite(best_result)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_result.reshape(16, 2)

if __name__ == "__main__":
    # Simple test
    pts = min_max_dist_dim2_16()
    print(pts.shape)
    d = pdist(pts, metric='sqeuclidean')
    print(f"Min dist: {np.sqrt(np.min(d)):.4f}, Max dist: {np.sqrt(np.max(d)):.4f}")
    print(f"Score: {(np.min(d)/np.max(d)):.4f}")
