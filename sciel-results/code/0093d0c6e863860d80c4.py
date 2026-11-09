import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    n = points.shape[0]
    if n < 3:
        return 0.0

    # Calculate minimum triangle area
    min_area = np.inf
    
    # Vectorized calculation for all triangles
    # We can compute all cross products at once for better performance
    # Indices for i, j, k with i < j < k
    # Total combinations for N=14 is 364, which is small.
    # But let's do a semi-vectorized approach or just a tight loop.
    # A tight loop in Python for 364 iterations is very fast (~0.1ms).
    
    for i in range(n - 2):
        pi = points[i]
        # Vectors from i to j and k
        # j ranges from i+1 to n-1
        # k ranges from j+1 to n-1
        for j in range(i + 1, n - 1):
            pj = points[j]
            v1 = pj - pi
            # v2 for all k > j
            # points[j+1:] - pi
            v2s = points[j+1:] - pi
            # Cross product 2D: v1[0]*v2[1] - v1[1]*v2[0]
            # v2s shape (m, 2)
            # cross = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    
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
    
    # Strategy:
    # 1. Initialize with a regular polygon (known good baseline).
    # 2. Perform a multi-start local optimization using Powell's method.
    #    We will generate several diverse initial points (perturbed regular polygons, 
    #    random convex polygons) and run Powell from each.
    # 3. Keep the best result.
    
    best_points = None
    best_score = 0.0
    
    def update_best(points):
        nonlocal best_points, best_score
        score = _calculate_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

    # 1. Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    update_best(init_points)
    
    # 2. Multi-start Powell optimization
    # We will try a few different initial configurations.
    # The key is to find a good local optimum. Powell is robust for this.
    
    # Define a helper to run Powell with time check
    def run_powell(initial_points):
        nonlocal start_time, timeout
        def _time_check_callback(xk):
            if time.time() - start_time > timeout - 5:
                raise TimeoutError("Local optimization timeout")
        
        try:
            res = minimize(
                _negative_score, 
                initial_points.flatten(), 
                method='Powell', 
                options={'maxiter': 300, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8},
                callback=_time_check_callback
            )
            if res.x is not None:
                cand_points = res.x.reshape(N, 2)
                return cand_points
        except TimeoutError:
            pass
        except Exception:
            pass
        return None

    # Start 1: Regular polygon
    if time.time() - start_time < timeout - 10:
        cand = run_powell(init_points)
        if cand is not None:
            update_best(cand)

    # Generate other starts
    # Start 2: Perturbed regular polygon
    np.random.seed(42)
    perturbation = np.random.normal(0, 0.1, (N, 2))
    init2 = init_points + perturbation
    # Project back to convex position? Not strictly necessary for Powell, but helps.
    # Just ensure they are distinct.
    if time.time() - start_time < timeout - 10:
        cand = run_powell(init2)
        if cand is not None:
            update_best(cand)

    # Start 3: Random convex polygon
    # Generate random angles, sort them, use as vertices of a convex polygon.
    # Add some radial variation.
    if time.time() - start_time < timeout - 10:
        angles_rand = np.sort(np.random.uniform(0, 2*np.pi, N))
        # Ensure minimum angular separation?
        # Simple random convex hull points:
        # Pick random points, take convex hull, if hull has < N points, discard or project.
        # Better: Generate points on a circle with random radii close to 1.
        radii = np.random.uniform(0.8, 1.2, N)
        init3 = np.column_stack((np.cos(angles_rand) * radii, np.sin(angles_rand) * radii))
        # Ensure convexity? If not convex, some points are inside, which is suboptimal 
        # for this problem (usually optimal solutions are convex). 
        # Powell will likely push them out.
        cand = run_powell(init3)
        if cand is not None:
            update_best(cand)

    # Start 4: Another random start
    if time.time() - start_time < timeout - 10:
        np.random.seed(123)
        angles_rand2 = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii2 = np.random.uniform(0.5, 1.5, N)
        init4 = np.column_stack((np.cos(angles_rand2) * radii2, np.sin(angles_rand2) * radii2))
        cand = run_powell(init4)
        if cand is not None:
            update_best(cand)

    # Start 5: "Star" pattern or alternating radii
    if time.time() - start_time < timeout - 10:
        angles5 = 2 * np.pi * np.arange(N) / N
        radii5 = np.ones(N)
        radii5[::2] = 0.9 # Slight perturbation
        init5 = np.column_stack((np.cos(angles5) * radii5, np.sin(angles5) * radii5))
        cand = run_powell(init5)
        if cand is not None:
            update_best(cand)

    # If time remains, try Nelder-Mead on the best point found so far
    elapsed = time.time() - start_time
    if elapsed < timeout - 10 and best_points is not None:
        try:
            def _nm_time_check_callback(xk):
                if time.time() - start_time > timeout - 5:
                    raise TimeoutError("NM optimization timeout")
            
            res_nm = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8},
                callback=_nm_time_check_callback
            )
            
            if res_nm.x is not None:
                cand_points = res_nm.x.reshape(N, 2)
                update_best(cand_points)
                
        except TimeoutError:
            pass
        except Exception:
            pass

    # Final check: ensure we have a valid result
    if best_points is None:
        # Fallback to regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
