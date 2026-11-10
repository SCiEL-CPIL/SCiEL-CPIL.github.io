# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _unit_hex_vertices():
    """Vertices of a unit regular hexagon centered at origin, angle 0."""
    angles = np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _check_separation_and_containment(centers, angles_deg, outer_angle_deg):
    """
    Checks if the configuration is valid.
    Returns:
        s: The minimum side length of the outer hexagon required to contain all inner hexagons.
        is_valid: True if all inner hexagons are separated and contained in a hexagon of side s.
        penalty: A scalar penalty value to be minimized.
    """
    n = len(centers)
    unit_verts = _unit_hex_vertices()
    
    # Compute vertices for all inner hexagons
    # Shape: (12, 6, 2)
    inner_verts_list = []
    for i in range(n):
        a = np.deg2rad(angles_deg[i])
        angles = a + np.deg2rad(60.0 * np.arange(6))
        verts = centers[i] + np.column_stack([np.cos(angles), np.sin(angles)])
        inner_verts_list.append(verts)
    inner_verts = np.array(inner_verts_list)
    
    # Determine outer hexagon side length s
    # We need the smallest s such that all inner verts are inside the outer hexagon centered at (0,0) with angle outer_angle_deg.
    # Actually, we should allow the outer hexagon to move too? 
    # The task says "outer a (3,) array (x, y, angle_deg)". 
    # Usually, we can center the outer hexagon at the centroid of the inner ones to minimize s? 
    # No, the outer hexagon position is part of the output. 
    # However, for a given set of inner hexagons, the optimal outer hexagon position is such that the "support function" is minimized.
    # For a regular polygon, the optimal position is often the centroid or determined by the bounding box in the normal directions.
    # Let's assume the outer hexagon is centered at the centroid of the inner hexagons' centers for the optimization step, 
    # and we can refine the outer position later if needed. 
    # Actually, the problem allows the outer hexagon to be anywhere.
    # To keep the parameter space small, I will fix the outer hexagon center to the centroid of the inner centers.
    # This is a valid assumption because shifting the outer hexagon and inner hexagons together doesn't change the relative geometry, 
    # and the "size" s is invariant to translation. 
    # Wait, the outer hexagon size s depends on the relative position. 
    # If I fix the outer center to the centroid of the inner centers, I am constraining the relative position. 
    # Is it always optimal to center the outer hexagon on the centroid of the inner packing?
    # For a convex container and a set of convex objects, the minimum enclosing container of a fixed shape is often aligned with the centroid, 
    # but not always. However, for regular hexagons, it's a good heuristic.
    # Let's try fixing outer center to (0,0) and inner centers relative to it.
    # So, params are: 12 inner centers (24), 12 inner angles (12), 1 outer angle (1). Total 37.
    
    outer_center = np.array([0.0, 0.0])
    
    # Calculate required s
    # The outer hexagon has vertices at angles (outer_angle + 60k).
    # Its edges have normals at (outer_angle + 30 + 60k).
    # The apothem (inradius) is r = s * sqrt(3) / 2.
    # A point p is inside if for all 6 normals n_k, dot(p, n_k) <= r.
    # So r >= max_k max_p dot(p, n_k).
    # s = 2 * r / sqrt(3).
    
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # Project all inner vertices onto the 6 normals
    # inner_verts: (12, 6, 2)
    # normals: (6, 2)
    # We want max projection of any inner vertex onto any normal.
    # proj[i, j, k] = dot(inner_verts[i, j], normals[k])
    # This is (12, 6, 6)
    # We can do this by reshaping.
    # inner_verts_flat: (72, 2)
    # proj = inner_verts_flat @ normals.T -> (72, 6)
    # max_proj = max(proj)
    
    inner_verts_flat = inner_verts.reshape(-1, 2)
    projections = inner_verts_flat @ normals.T
    max_projection = np.max(projections)
    
    # The apothem must be at least max_projection
    apothem = max_projection
    s = 2.0 * apothem / np.sqrt(3.0)
    
    # Check Separation
    # For each pair of inner hexagons, check if they overlap.
    # Overlap if they share interior points.
    # SAT: if there is a separating axis, they don't overlap.
    # Axes: normals of all edges of both hexagons.
    
    penalty = 0.0
    is_valid = True
    
    # Precompute normals for each inner hexagon
    # Normal of edge k for hex i: angle = angles_deg[i] + 30 + 60k
    inner_normals = []
    for i in range(n):
        a = np.deg2rad(angles_deg[i])
        n_angles = a + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
        n_vecs = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
        inner_normals.append(n_vecs)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check SAT
            vi = inner_verts[i]
            vj = inner_verts[j]
            
            # Normals to check: inner_normals[i] and inner_normals[j]
            all_normals = np.vstack([inner_normals[i], inner_normals[j]])
            
            separated = False
            for k in range(12):
                n = all_normals[k]
                proj_i = vi @ n
                proj_j = vj @ n
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - 1e-8 or max_j < min_i - 1e-8:
                    separated = True
                    break
            
            if not separated:
                # They overlap
                # Calculate penetration depth for penalty
                # Approximate by min distance between vertices? 
                # Or just a fixed penalty.
                penalty += 1000.0
                is_valid = False
            
            # Early exit if penalty is too high
            if penalty > 10000:
                break
        if penalty > 10000:
            break
    
    return s, is_valid, penalty


def _objective(params):
    """
    Objective function for the optimizer.
    params: [cx0, cy0, a0, cx1, cy1, a1, ..., cx11, cy11, a11, outer_angle]
    """
    n = 12
    centers = params[:2*n].reshape(n, 2)
    angles_deg = params[2*n:3*n]
    outer_angle_deg = params[3*n]
    
    s, is_valid, penalty = _check_separation_and_containment(centers, angles_deg, outer_angle_deg)
    
    if penalty > 0:
        return s + penalty
    return s


def _get_initial_guess():
    """
    Generate a good initial guess based on the rigid cluster.
    """
    sqrt3 = np.sqrt(3.0)
    centers = np.array([[0.0, 0.0]])
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        centers = np.vstack([centers, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    for i in range(5):
        ang = np.deg2rad(30.0 + 60.0 * i)
        centers = np.vstack([centers, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    
    # Center the cluster
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    angles_deg = np.zeros(12)
    outer_angle = 0.0
    
    params = np.concatenate([centers.flatten(), angles_deg, [outer_angle]])
    return params


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    best_params = _get_initial_guess()
    best_s, best_valid, _ = _check_separation_and_containment(
        best_params[:24].reshape(12, 2), 
        best_params[24:36], 
        best_params[36]
    )
    
    # Define bounds for differential evolution
    # Centers: -10 to 10
    # Angles: 0 to 60 (due to symmetry)
    # Outer angle: 0 to 60
    bounds = [(-10, 10)] * 24 + [(0, 60)] * 12 + [(0, 60)]
    
    # Use Differential Evolution for global search
    # This is expensive, so we limit iterations.
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        try:
            # Use a subset of the population to save time?
            # Or just run DE with limited maxiter.
            # The objective is non-smooth due to penalties, but DE handles this well.
            
            # To speed up, we might skip DE if the initial guess is already very good?
            # The initial guess is a valid packing.
            # Let's try a local refinement first, which is faster.
            
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_s:
                best_params = result.x
                best_s, best_valid, _ = _check_separation_and_containment(
                    best_params[:24].reshape(12, 2), 
                    best_params[24:36], 
                    best_params[36]
                )
                
        except Exception as e:
            print(f"Local refinement failed: {e}")
        
    # If time remains, try Differential Evolution
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        try:
            # DE is stochastic. We need to be careful with the time.
            # Let's run a short DE.
            # Note: DE calls the objective many times.
            # We'll use a smaller population size.
            
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                seed=42,
                update_interval=1,
                polish=True
            )
            
            if de_result.fun < best_s:
                best_params = de_result.x
                best_s, best_valid, _ = _check_separation_and_containment(
                    best_params[:24].reshape(12, 2), 
                    best_params[24:36], 
                    best_params[36]
                )
                
        except Exception as e:
            print(f"DE failed: {e}")
            
    # Final local refinement
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        try:
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if result.fun < best_s:
                best_params = result.x
                best_s, best_valid, _ = _check_separation_and_containment(
                    best_params[:24].reshape(12, 2), 
                    best_params[24:36], 
                    best_params[36]
                )
        except Exception as e:
            print(f"Final refinement failed: {e}")

    # Extract final parameters
    centers = best_params[:24].reshape(12, 2)
    angles_deg = best_params[24:36]
    outer_angle_deg = best_params[36]
    
    # Compute final s
    s_final, valid_final, pen_final = _check_separation_and_containment(centers, angles_deg, outer_angle_deg)
    
    # If invalid, fall back to initial guess
    if not valid_final:
        centers = _get_initial_guess()[:24].reshape(12, 2)
        angles_deg = _get_initial_guess()[24:36]
        outer_angle_deg = _get_initial_guess()[36]
        s_final, valid_final, pen_final = _check_separation_and_containment(centers, angles_deg, outer_angle_deg)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = angles_deg[i]
    
    outer_hex_data = np.array([0.0, 0.0, outer_angle_deg])
    outer_hex_side_length = s_final
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
