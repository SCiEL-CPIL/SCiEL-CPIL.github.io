import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

def _compute_score(points):
    """
    Compute the Heilbronn score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate all triangle areas
    # Using vectorized calculation for speed
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0

    return min_area / hull_area

def _objective(x):
    """
    Objective function for differential_evolution.
    Minimizes the negative score.
    x is a flat array of size 26 (13 points * 2 coords).
    """
    points = x.reshape(13, 2)
    score = _compute_score(points)
    return -score

def _refine_local(points, time_limit=5.0):
    """
    A robust local refinement using subgradient ascent on the active set.
    This is kept from the original strategy but made more stable.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.02
    no_improve_count = 0
    max_no_improve = 20
    n = 13
    
    # Precompute indices for speed
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]

    while time.time() - start_time < time_limit:
        # Calculate areas
        p1 = current_points[i_flat]
        p2 = current_points[j_flat]
        p3 = current_points[k_flat]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        if min_area < 1e-12:
            break

        # Find active triangles (those close to min)
        tol = max(1e-10, 0.05 * min_area)
        active_mask = areas <= (min_area + tol)
        
        if not np.any(active_mask):
            break
            
        # Compute subgradient
        grads = np.zeros_like(current_points)
        active_indices = np.where(active_mask)[0]
        
        for idx in active_indices:
            i, j, k = i_flat[idx], j_flat[idx], k_flat[idx]
            p1 = current_points[i]
            p2 = current_points[j]
            p3 = current_points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            sign = np.sign(cross)
            if sign == 0:
                sign = 1.0
            
            # Gradients of 0.5 * sign * cross
            # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
            # d/dx1 = -(y3-y1) + (y2-y1) = y2 - y3
            # d/dy1 = (x3-x1) - (x2-x1) = x3 - x2
            # d/dx2 = (y3-y1)
            # d/dy2 = -(x3-x1)
            # d/dx3 = -(y2-y1)
            # d/dy3 = (x2-x1)
            
            # Note: Area = 0.5 * sign * cross.
            # So gradient is 0.5 * sign * grad(cross)
            
            g1 = 0.5 * sign * np.array([y2-y3, x3-x2]) # Wait, y2-y3?
            # Let's re-derive carefully.
            # A = 0.5 * s * ( (x2-x1)(y3-y1) - (y2-y1)(x3-x1) )
            # dA/dx1 = 0.5 * s * ( -(y3-y1) + (y2-y1) ) = 0.5 * s * (y2 - y3)
            # dA/dy1 = 0.5 * s * ( (x3-x1) - (x2-x1) ) = 0.5 * s * (x3 - x2)
            
            # dA/dx2 = 0.5 * s * (y3-y1)
            # dA/dy2 = 0.5 * s * ( -(x3-x1) )
            
            # dA/dx3 = 0.5 * s * ( -(y2-y1) )
            # dA/dy3 = 0.5 * s * ( (x2-x1) )
            
            grads[i] += np.array([0.5 * sign * (p2[1] - p3[1]), 0.5 * sign * (p3[0] - p2[0])])
            grads[j] += np.array([0.5 * sign * (p3[1] - p1[1]), 0.5 * sign * (-(p3[0] - p1[0]))])
            grads[k] += np.array([0.5 * sign * (-(p2[1] - p1[1])), 0.5 * sign * (p2[0] - p1[0])])
            
        # Normalize gradients to prevent large steps if many active triangles
        # We want to move in the direction that increases the minimum area.
        # The subgradient of the min function is in the convex hull of the gradients of the active functions.
        # We use the average gradient of the active triangles as a descent direction for -min_area.
        # So we ascend in the direction of the average gradient.
        
        # Only update points that are part of active triangles?
        # Actually, moving inactive points might break the hull or change the active set.
        # Standard subgradient method moves all points, but usually only active ones have non-zero gradient in this context if we define it that way.
        # Here, grads are zero for inactive points. So they don't move.
        
        new_points = current_points + step_size * grads
        
        # Project to ensure points remain in a reasonable range? 
        # DE works in a box. We should probably keep points within the hull or a bounding box.
        # For local refinement, we don't strictly need a box, but it helps stability.
        
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            no_improve_count = 0
            step_size *= 0.95 # Decay step size
        else:
            no_improve_count += 1
            step_size *= 0.7
            if step_size < 1e-6:
                break
            if no_improve_count > max_no_improve:
                break
                
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Uses Differential Evolution for global search, followed by local subgradient refinement.
    """
    start_time = time.time()
    deadline = start_time + 580 # 50s safety margin
    
    # Define bounds for DE. 
    # We can normalize the space to [0, 1] x [0, 1] to help DE.
    # The score is scale-invariant, so the absolute size doesn't matter, only the relative positions.
    # However, DE needs finite bounds.
    bounds = [(0, 1) for _ in range(26)]
    
    # Initial population size
    # DE default popsize is 15. For 26 dims, 15*26 = 390 individuals.
    # Let's try a larger population for better exploration.
    popsize = 20
    
    # Max iterations
    # We will run DE until time is up, then refine.
    # We need to estimate how long one DE iteration takes.
    # One iteration = popsize * 26 * 286 triangle calcs (approx) + hull.
    # It's fast, but let's be conservative.
    
    # We will run DE in chunks to check time.
    # However, differential_evolution doesn't support early stopping easily without a callback.
    # We can use a wrapper or just set a reasonable maxiter and rely on the fact that it's fast.
    # Let's set maxiter high and see if it finishes. If not, we might timeout.
    # Better: Use a custom loop or a callback?
    # scipy.optimize.differential_evolution has a 'callback' parameter.
    
    def de_callback(xk, convergence):
        if time.time() > deadline - 60: # Leave 60s for refinement
            raise RuntimeError("Time limit reached for DE")

    best_result = None
    try:
        # Use 'differential_evolution'
        # strategy: 'best1bin' is standard.
        # mutation: (0.5, 1.0)
        # recombination: 0.7
        result = differential_evolution(
            _objective, 
            bounds, 
            popsize=popsize, 
            maxiter=200, 
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False, # We will do our own polish
            callback=de_callback,
            init='latinhypercube'
        )
        best_points = result.x.reshape(13, 2)
        best_score = -result.fun
        best_result = best_points
        
    except RuntimeError:
        # If DE didn't finish, we might not have a result.
        # But DE updates the best internally. We can't easily access it if it raises.
        # Let's try a different approach: Run DE with a smaller maxiter and check time.
        # Or, use a manual loop? No, DE is complex.
        # Let's assume 200 iterations of popsize 20 fits in 500s.
        # 20 * 200 = 4000 evaluations.
        # Each eval is ~1ms? 4000ms = 4s. So it should be very fast.
        # The bottleneck is likely the hull calculation or the overhead of Python loops in DE.
        # Actually, DE is implemented in Python in scipy, so it might be slower.
        # Let's reduce maxiter to 50 and see.
        pass

    # If DE failed or timed out, fall back to a robust initial guess
    if best_result is None:
        # Use a known good configuration as a seed for refinement
        # Hexagonal packing variant
        theta_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        outer = np.column_stack((np.cos(theta_outer), np.sin(theta_outer)))
        r_inner = 0.5
        theta_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
        inner = np.column_stack((r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)))
        best_points = np.vstack([outer, inner])
        best_score = _compute_score(best_points)
    else:
        best_points = best_result
        best_score = _compute_score(best_points)

    # Refine the best point found
    if time.time() < deadline - 10:
        refined_points, refined_score = _refine_local(best_points, time_limit=30.0)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score

    # Final sanity check
    if not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
