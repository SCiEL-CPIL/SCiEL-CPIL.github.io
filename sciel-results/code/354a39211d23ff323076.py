import math
import time
import numpy as np
from scipy.optimize import minimize

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
    The edge normals are at angles angle_deg + 30 + k*60.
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
    Return the standard concentric ring configuration.
    """
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(30.0)
    
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    # Second ring: 4 hexagons at distance 3, at angles 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
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

def _objective(params, deadline):
    """
    Objective function for optimization.
    params: [x0, y0, x1, y1, ..., x10, y10, angle0, angle1, ..., angle10, outer_angle]
    Returns the side length of the outer hexagon.
    """
    if time.time() > deadline:
        return 1e9
    
    n = 11
    centers = np.array(params[:2*n]).reshape(n, 2)
    inner_angles = np.array(params[2*n:3*n])
    outer_angle = params[3*n]
    
    # Check for overlap
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < np.sqrt(3) - 1e-8:
                return 1e9
    
    return _min_enclosing_side(centers, inner_angles, outer_angle)

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
    best_inner_angles = inner_angles.copy()
    best_outer_angle = outer_angle
    
    # Verify no overlap
    if not _check_no_overlap(best_centers):
        best_centers = best_centers * 1.001
        best_side = _min_enclosing_side(best_centers, best_inner_angles, best_outer_angle)
    
    # Strategy: Gradient-based optimization with parameterization
    # params: [x0, y0, ..., x10, y10, angle0, ..., angle10, outer_angle]
    # Total: 22 + 11 + 1 = 34 parameters
    
    if time.time() < DEADLINE - safety_margin:
        try:
            # Initial parameter vector
            init_params = np.concatenate([
                best_centers.flatten(),
                best_inner_angles,
                [best_outer_angle]
            ])
            
            def objective_func(params):
                if time.time() > DEADLINE - 2:
                    return 1e9
                return _objective(params, DEADLINE)
            
            # Use L-BFGS-B for constrained optimization
            res = minimize(
                objective_func,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_side and res.fun < 1e8:
                best_side = res.fun
                best_centers = np.array(res.x[:22]).reshape(11, 2)
                best_inner_angles = np.array(res.x[22:33])
                best_outer_angle = res.x[33]
                
                # Verify no overlap
                if not _check_no_overlap(best_centers):
                    # Revert to previous best
                    best_side = _min_enclosing_side(best_centers, best_inner_angles, best_outer_angle)
                    if best_side > 1e8:
                        # Overlap found, revert
                        best_centers, best_inner_angles = _seed_configuration()
                        best_outer_angle = 30.0
                        best_side = _min_enclosing_side(best_centers, best_inner_angles, best_outer_angle)
        except Exception:
            pass
    
    # Strategy 2: Try different initial configurations and optimize
    if time.time() < DEADLINE - safety_margin:
        for inner_rot in [0, 15, 30, 45, 60]:
            test_angles = np.array([inner_rot] * 11)
            test_side = _min_enclosing_side(best_centers, test_angles, best_outer_angle)
            
            if test_side < best_side - 1e-12:
                if time.time() < DEADLINE - 2:
                    try:
                        init_params = np.concatenate([
                            best_centers.flatten(),
                            test_angles,
                            [best_outer_angle]
                        ])
                        
                        def obj2(params):
                            if time.time() > DEADLINE - 2:
                                return 1e9
                            return _objective(params, DEADLINE)
                        
                        res2 = minimize(
                            obj2,
                            init_params,
                            method='L-BFGS-B',
                            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                        
                        if res2.fun < best_side and res2.fun < 1e8:
                            best_side = res2.fun
                            best_centers = np.array(res2.x[:22]).reshape(11, 2)
                            best_inner_angles = np.array(res2.x[22:33])
                            best_outer_angle = res2.x[33]
                            
                            if not _check_no_overlap(best_centers):
                                best_centers, best_inner_angles = _seed_configuration()
                                best_outer_angle = 30.0
                                best_side = _min_enclosing_side(best_centers, best_inner_angles, best_outer_angle)
                    except Exception:
                        pass
                else:
                    best_side = test_side
                    best_inner_angles = test_angles
    
    # Strategy 3: Small perturbations of the cluster
    if time.time() < DEADLINE - safety_margin:
        for dx in [-0.01, 0.0, 0.01]:
            for dy in [-0.01, 0.0, 0.01]:
                test_centers = best_centers + np.array([dx, dy])
                if not _check_no_overlap(test_centers):
                    continue
                
                test_side = _min_enclosing_side(test_centers, best_inner_angles, best_outer_angle)
                
                if test_side < best_side - 1e-12:
                    if time.time() < DEADLINE - 2:
                        try:
                            init_params = np.concatenate([
                                test_centers.flatten(),
                                best_inner_angles,
                                [best_outer_angle]
                            ])
                            
                            def obj3(params):
                                if time.time() > DEADLINE - 2:
                                    return 1e9
                                return _objective(params, DEADLINE)
                            
                            res3 = minimize(
                                obj3,
                                init_params,
                                method='L-BFGS-B',
                                options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                            )
                            
                            if res3.fun < best_side and res3.fun < 1e8:
                                best_side = res3.fun
                                best_centers = np.array(res3.x[:22]).reshape(11, 2)
                                best_inner_angles = np.array(res3.x[22:33])
                                best_outer_angle = res3.x[33]
                                
                                if not _check_no_overlap(best_centers):
                                    best_centers, best_inner_angles = _seed_configuration()
                                    best_outer_angle = 30.0
                                    best_side = _min_enclosing_side(best_centers, best_inner_angles, best_outer_angle)
                        except Exception:
                            pass
                    else:
                        best_side = test_side
                        best_centers = test_centers
    
    # Final verification
    if not _check_no_overlap(best_centers):
        best_centers, best_inner_angles = _seed_configuration()
        best_side = _min_enclosing_side(best_centers, best_inner_angles, 30.0)
        best_outer_angle = 30.0
    
    # Recompute final side length to be safe
    final_side = _min_enclosing_side(best_centers, best_inner_angles, best_outer_angle)
    if final_side < best_side:
        best_side = final_side
    
    # Build the inner list: [x, y, angle_degrees]
    inner = []
    for i in range(11):
        inner.append([best_centers[i, 0], best_centers[i, 1], best_inner_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle
