import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    n = points.shape[0]
    
    # Calculate minimum triangle area
    # Vectorized calculation for speed
    # We need areas of all triples (i, j, k)
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    min_area = np.inf
    
    # Loop is faster than fully vectorizing all 364 triangles at once for small N in Python overhead,
    # but we can vectorize the inner loops.
    for i in range(n):
        pi = points[i]
        # Vectors from i to all j > i
        diffs_j = points[i+1:] - pi
        for j_idx in range(len(diffs_j)):
            v1 = diffs_j[j_idx]
            # Vectors from i to all k > j
            # k indices in original points: i+1+j_idx+1 to n-1
            # Actually, easier to just loop k
            # Let's stick to the efficient triple loop but optimized
            pass
            
    # Let's use a more efficient vectorized approach for the min area
    # Create indices
    # This might be memory heavy if not careful, but N=14 is small.
    # Total triples = 364.
    # We can compute all cross products at once?
    
    # Let's try a semi-vectorized approach
    # For each i, compute cross products with all pairs (j,k) where j>i, k>j
    # This is complex to vectorize perfectly without creating large intermediate arrays.
    # The original loop is O(N^3) = 2744 operations. In Python, this is ~1-2ms.
    # DE will call this ~1000-5000 times. 5000 * 2ms = 10s. Acceptable.
    
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if not np.isfinite(min_area) or min_area <= 0:
        return 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 0:
        return 0.0

    return min_area / hull_area

def _negative_score(flat_points):
    points = flat_points.reshape(N, 2)
    return -_calculate_score(points)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    # 1. Global Search using Differential Evolution
    # Bounds: points in [-1, 1] square
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_points = None
    best_score = 0.0
    
    # Initialize with a regular polygon to have a valid starting point
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_points = init_points.copy()
    best_score = _calculate_score(best_points)
    
    # Run Differential Evolution
    # popsize=15, maxiter=50 is a reasonable start.
    # We will monitor time.
    try:
        # Use 'best1bin' strategy
        res_de = differential_evolution(
            _negative_score, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            disp=False,
            polish=False, # We will do local refinement manually to control time
            seed=42
        )
        
        if res_de.x is not None:
            cand_points = res_de.x.reshape(N, 2)
            cand_score = _calculate_score(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points.copy()
                
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout - 10:
        return best_points
    
    # 2. Local Refinement using Powell
    # Powell is effective for non-smooth minima and doesn't need gradients.
    # We run it from the best point found by DE.
    try:
        # Set a time limit for Powell by checking in the callback? 
        # SciPy minimize doesn't have a direct time limit option.
        # We can use a wrapper that raises an exception if time is up.
        
        def _time_check_callback(xk):
            if time.time() - start_time > timeout - 5:
                raise TimeoutError("Local optimization timeout")
        
        res_powell = minimize(
            _negative_score, 
            best_points.flatten(), 
            method='Powell', 
            options={'maxiter': 200, 'disp': False, 'xtol': 1e-6, 'ftol': 1e-6},
            callback=_time_check_callback
        )
        
        if res_powell.x is not None:
            cand_points = res_powell.x.reshape(N, 2)
            cand_score = _calculate_score(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points.copy()
                
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # 3. If time remains, try Nelder-Mead on the best point
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
        try:
            def _nm_time_check_callback(xk):
                if time.time() - start_time > timeout - 5:
                    raise TimeoutError("NM optimization timeout")
            
            res_nm = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 300, 'disp': False},
                callback=_nm_time_check_callback
            )
            
            if res_nm.x is not None:
                cand_points = res_nm.x.reshape(N, 2)
                cand_score = _calculate_score(cand_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_points.copy()
                    
        except TimeoutError:
            pass
        except Exception:
            pass
            
    return best_points
