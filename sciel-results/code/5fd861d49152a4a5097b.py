import math
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_normals(angle_deg):
    """
    Return the 6 outward unit normals of a regular hexagon with the given rotation.
    """
    base = np.deg2rad(angle_deg + 30)
    normals = []
    for k in range(6):
        theta = base + k * np.pi / 3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    return np.array(normals)

def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Compute the minimum side length of the outer hexagon (centered at origin) 
    such that all inner hexagons are contained.
    """
    normals = _outer_hex_normals(outer_angle_deg)
    max_proj = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        a = inner_angles[i]
        verts = _hex_vertices(cx, cy, a)
        dots = verts @ normals.T
        abs_dots = np.abs(dots)
        m = np.max(abs_dots)
        if m > max_proj:
            max_proj = m
    
    return 2.0 * max_proj / np.sqrt(3)

def _seed_configuration():
    """
    Return the standard concentric ring configuration:
    - 1 hexagon at center
    - 6 hexagons in first ring at distance sqrt(3)
    - 4 hexagons in second ring at distance 3 (filling gaps)
    All inner hexagons have the same orientation (30 degrees).
    """
    eps = 1e-10
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(30.0)
    
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    # Second ring: 4 hexagons at distance 3, at angles 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + eps
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    return np.array(centers), np.array(angles)

def _check_no_overlap(centers, min_dist=np.sqrt(3) - 1e-9):
    """Check that no two inner hexagons overlap (center distance >= sqrt(3))."""
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                return False
    return True

def _smooth_max(vals, beta=100.0):
    """
    Soft-max via log-sum-exp for smooth approximation of max.
    Returns smooth approximation of max(vals).
    """
    vals = np.asarray(vals, dtype=float)
    m = np.max(vals)
    if m > 1e6:
        return m
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return m + log_sum

def _deformation_objective(params, inner_angles, outer_angle_deg, beta_smooth=50.0):
    """
    Objective for deformation optimization.
    params: 22-vector = [dx1, dy1, dx2, dy2, ..., dx11, dy11]
    Returns: smooth max projection (proxy for side length) + overlap penalty
    """
    centers = _seed_configuration()[0] + params.reshape(11, 2)
    
    # Compute max projection (smooth)
    normals = _outer_hex_normals(outer_angle_deg)
    all_projs = []
    for i in range(11):
        cx, cy = centers[i]
        a = inner_angles[i]
        verts = _hex_vertices(cx, cy, a)
        dots = verts @ normals.T
        all_projs.append(np.abs(dots).flatten())
    
    all_projs = np.concatenate(all_projs)
    smooth_max_proj = _smooth_max(all_projs, beta_smooth)
    
    # Overlap penalty: for each pair, if distance < sqrt(3), add penalty
    penalty = 0.0
    min_dist = np.sqrt(3)
    for i in range(11):
        for j in range(i + 1, 11):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                # Quadratic penalty for violation
                violation = min_dist - d
                penalty += 1000.0 * violation * violation
    
    # Side length proxy
    side_proxy = 2.0 * smooth_max_proj / np.sqrt(3)
    
    return side_proxy + penalty

def _deformation_gradient(params, inner_angles, outer_angle_deg, beta_smooth=50.0, eps=1e-7):
    """
    Numerical gradient of the deformation objective.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        fp = _deformation_objective(params_plus, inner_angles, outer_angle_deg, beta_smooth)
        fm = _deformation_objective(params_minus, inner_angles, outer_angle_deg, beta_smooth)
        grad[i] = (fp - fm) / (2 * eps)
    return grad

def _optimize_deformation(inner_angles, outer_angle_deg, beta_smooth=50.0, max_iter=200):
    """
    Optimize the deformation parameters using BFGS with numerical gradient.
    Returns: (params, objective_value)
    """
    x0 = np.zeros(22)
    
    res = minimize(
        _deformation_objective,
        x0,
        args=(inner_angles, outer_angle_deg, beta_smooth),
        method='BFGS',
        jac=_deformation_gradient,
        options={'maxiter': max_iter, 'gtol': 1e-8}
    )
    
    return res.x, res.fun

def _validate_and_refine(centers, inner_angles, outer_angle_deg):
    """
    Validate that the configuration is feasible (no overlaps) and 
    refine the outer angle to minimize side length.
    Returns: (side_length, refined_outer_angle) or None if invalid
    """
    if not _check_no_overlap(centers):
        return None
    
    side = _min_enclosing_side(centers, inner_angles, outer_angle_deg)
    
    # Refine outer angle
    if time.time() < 145.0 - 3.0:
        try:
            def obj(angle_deg):
                return _min_enclosing_side(centers, inner_angles, angle_deg)
            
            res = minimize_scalar(obj, bounds=(0, 60), method='bounded',
                                   options={'xatol': 1e-10})
            if res.fun < side:
                side = res.fun
                outer_angle_deg = res.x
        except Exception:
            pass
    
    return side, outer_angle_deg

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    # Start with the seed configuration
    centers, inner_angles = _seed_configuration()
    outer_angle = 30.0
    
    # Compute initial side length
    best_side = _min_enclosing_side(centers, inner_angles, outer_angle)
    best_centers = centers.copy()
    best_outer_angle = outer_angle
    
    # Verify no overlap
    if not _check_no_overlap(best_centers):
        best_centers = best_centers * 1.001
        best_side = _min_enclosing_side(best_centers, inner_angles, best_outer_angle)
    
    # Strategy 1: Optimize outer angle for the rigid seed configuration
    if time.time() < DEADLINE - safety_margin:
        try:
            def objective(angle_deg):
                if time.time() > DEADLINE - 2:
                    return 1e9
                return _min_enclosing_side(best_centers, inner_angles, angle_deg)
            
            res = minimize_scalar(objective, bounds=(0, 60), method='bounded',
                                   options={'xatol': 1e-10})
            
            if res.fun < best_side:
                best_side = res.fun
                best_outer_angle = res.x
        except Exception:
            pass
    
    # Strategy 2: Deformation-based optimization
    # Try different smoothness parameters and inner orientations
    if time.time() < DEADLINE - safety_margin:
        for inner_rot in [30.0]:  # Keep same orientation as seed
            test_angles = np.array([inner_rot] * 11)
            
            for beta_smooth in [50.0, 100.0, 200.0]:
                if time.time() > DEADLINE - 10.0:
                    break
                
                try:
                    params, obj_val = _optimize_deformation(
                        test_angles, best_outer_angle, beta_smooth, max_iter=150
                    )
                    
                    # Extract centers from params
                    deformed_centers = _seed_configuration()[0] + params.reshape(11, 2)
                    
                    # Validate and refine
                    result = _validate_and_refine(deformed_centers, test_angles, best_outer_angle)
                    if result is not None:
                        side, refined_angle = result
                        if side < best_side - 1e-12:
                            best_side = side
                            best_centers = deformed_centers
                            best_outer_angle = refined_angle
                            inner_angles = test_angles
                except Exception:
                    pass
    
    # Strategy 3: Try different inner orientations with rigid cluster
    if time.time() < DEADLINE - safety_margin:
        for inner_rot in [0, 15, 45]:
            test_angles = np.array([inner_rot] * 11)
            test_side = _min_enclosing_side(best_centers, test_angles, best_outer_angle)
            
            if test_side < best_side - 1e-12:
                if time.time() < DEADLINE - 2:
                    try:
                        def obj2(angle_deg):
                            if time.time() > DEADLINE - 2:
                                return 1e9
                            return _min_enclosing_side(best_centers, test_angles, angle_deg)
                        
                        res2 = minimize_scalar(obj2, bounds=(0, 60), method='bounded',
                                                 options={'xatol': 1e-10})
                        if res2.fun < best_side:
                            best_side = res2.fun
                            best_outer_angle = res2.x
                            inner_angles = test_angles
                    except Exception:
                        pass
                else:
                    best_side = test_side
                    inner_angles = test_angles
    
    # Strategy 4: Small perturbations of the cluster
    if time.time() < DEADLINE - safety_margin:
        for dx in [-0.01, 0.0, 0.01]:
            for dy in [-0.01, 0.0, 0.01]:
                test_centers = best_centers + np.array([dx, dy])
                if not _check_no_overlap(test_centers):
                    continue
                
                test_side = _min_enclosing_side(test_centers, inner_angles, best_outer_angle)
                
                if test_side < best_side - 1e-12:
                    if time.time() < DEADLINE - 2:
                        try:
                            def obj3(angle_deg):
                                if time.time() > DEADLINE - 2:
                                    return 1e9
                                return _min_enclosing_side(test_centers, inner_angles, angle_deg)
                            
                            res3 = minimize_scalar(obj3, bounds=(0, 60), method='bounded',
                                                     options={'xatol': 1e-10})
                            if res3.fun < best_side:
                                best_side = res3.fun
                                best_outer_angle = res3.x
                                best_centers = test_centers
                        except Exception:
                            pass
                    else:
                        best_side = test_side
                        best_centers = test_centers
    
    # Final verification
    if not _check_no_overlap(best_centers):
        best_centers, inner_angles = _seed_configuration()
        best_side = _min_enclosing_side(best_centers, inner_angles, 30.0)
        best_outer_angle = 30.0
    
    # Recompute final side length to be safe
    final_side = _min_enclosing_side(best_centers, inner_angles, best_outer_angle)
    if final_side < best_side:
        best_side = final_side
    
    # Build the inner list: [x, y, angle_degrees]
    inner = []
    for i in range(11):
        inner.append([best_centers[i, 0], best_centers[i, 1], inner_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle
