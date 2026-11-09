"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if np.any(np.isnan(points)):
            return 0.0
            
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        # Precompute all triangle areas using vectorized approach
        # For each pair (i,j), compute cross product with all k
        # This is O(N^3) but vectorized
        
        # Use the formula: area of triangle (i,j,k) = 0.5 * |det(pj-pi, pk-pi)|
        # We can compute all determinants at once
        
        # Create index arrays
        i_idx, j_idx, k_idx = np.indices((N, N, N))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _smooth_score(points, beta=100.0):
    """
    Smooth surrogate for the min triangle area score using log-sum-exp.
    This makes the objective differentiable.
    """
    try:
        if np.any(np.isnan(points)):
            return 0.0
            
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        # Precompute all triangle areas
        i_idx, j_idx, k_idx = np.indices((N, N, N))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Use log-sum-exp to get smooth minimum
        # min(a) = -logsumexp(-a) / beta approximation
        # Actually, soft-min: -log(sum(exp(-a_i/beta))) * beta
        # For large beta, this approaches min(a)
        
        # To avoid overflow, subtract max
        neg_areas = -areas / beta
        max_neg = np.max(neg_areas)
        log_sum = np.log(np.sum(np.exp(neg_areas - max_neg))) + max_neg
        soft_min = -beta * log_sum
        
        return soft_min / hull_area
    except Exception:
        return 0.0

def _negative_smooth_score(points_flat, beta=100.0):
    """
    Negative smooth score for minimization.
    """
    points = points_flat.reshape(N, 2)
    return -_smooth_score(points, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Initial guesses
    guesses = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    guesses.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Points on a circle with radial and angular noise
    for _ in range(15):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.normal(0, 0.08, N)
        radii = 1.0 + rng.normal(0, 0.04, N)
        guesses.append(np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii]))
        
    # 3. Random points in unit square
    for _ in range(5):
        guesses.append(rng.random((N, 2)))
        
    # 4. Points on concentric circles
    for r in [0.5, 0.7, 0.85]:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        guesses.append(np.column_stack([np.cos(angles)*r, np.sin(angles)*r]))
    
    for guess in guesses:
        if time.time() > deadline - 10:
            break
            
        # First pass: use smooth objective with L-BFGS-B for faster convergence
        try:
            # Use moderate beta for smoothness
            res_smooth = minimize(_negative_smooth_score, guess.flatten(), 
                                  args=(50.0,),
                                  method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Second pass: refine with exact objective using Powell
            if time.time() < deadline - 5:
                res = minimize(_negative_score, res_smooth.x, method='Powell', 
                               options={'maxiter': 1000, 'ftol': 1e-10, 'xtol': 1e-10})
            else:
                res = res_smooth
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(N, 2)
        except Exception:
            # Fallback to direct Powell
            try:
                res = minimize(_negative_score, guess.flatten(), method='Powell', 
                               options={'maxiter': 1500, 'ftol': 1e-10, 'xtol': 1e-10})
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2)
            except Exception:
                pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Final polish: run one more optimization with tighter tolerance
    if time.time() < deadline - 3:
        try:
            res_final = minimize(_negative_score, best_points.flatten(), method='Powell', 
                               options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12})
            final_score = -res_final.fun
            if final_score > best_score:
                best_points = res_final.x.reshape(N, 2)
                best_score = final_score
        except Exception:
            pass
        
    return best_points
# EVOLVE-BLOCK-END
