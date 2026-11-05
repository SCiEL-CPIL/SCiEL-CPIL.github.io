import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

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
    A regular hexagon with side s has apothem a = s*sqrt(3)/2.
    The vertices are at angles angle_deg + k*60.
    The edge normals (perpendicular to edges, pointing outward) are at angles angle_deg + 30 + k*60.
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
    Uses the property that for a regular hexagon, containment is equivalent to
    each vertex's projection onto each of the 6 edge normals being <= apothem.
    Apothem = side * sqrt(3) / 2.
    So side = 2 * max_projection / sqrt(3).
    """
    normals = _outer_hex_normals(outer_angle_deg)
    max_proj = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        a = inner_angles[i]
        verts = _hex_vertices(cx, cy, a)
        # Project all vertices onto all normals
        dots = verts @ normals.T
        # For containment in a centered hexagon, we need |projection| <= apothem
        # Actually, the hexagon is defined by -apothem <= n_i . v <= apothem for all vertices v
        # So we need max over all vertices and normals of |n_i . v|
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

def _smooth_max(values, beta=50.0):
    """
    Smooth approximation of max using log-sum-exp.
    As beta -> infinity, this approaches the true max.
    """
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    if max_val < -1000:
        return max_val
    return max_val + np.log(np.sum(np.exp(beta * (values - max_val)))) / beta

def _deformation_objective(params, outer_angle_deg, beta=50.0):
    """
    Objective function for deformation optimization.
    params: [dx1, dy1, dx2, dy2, ..., dx11, dy11] - displacements from seed config
    Returns: (side_length, overlap_penalty)
    """
    start_time = time.time()
    DEADLINE = 145.0
    if time.time() > DEADLINE - 5:
        return (1e9, 1e9)
    
    # Get seed configuration
    seed_centers, seed_angles = _seed_configuration()
    
    # Apply displacements
    centers = seed_centers.copy()
    for i in range(N):
        centers[i, 0] += params[2*i]
        centers[i, 1] += params[2*i + 1]
    
    inner_angles = seed_angles.copy()
    
    # Compute side length
    side = _min_enclosing_side(centers, inner_angles, outer_angle_deg)
    
    # Compute overlap penalty using smooth max
    n = len(centers)
    overlap_terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            # Penalty when distance < sqrt(3)
            violation = np.sqrt(3) - d
            if violation > 0:
                overlap_terms.append(violation)
    
    if overlap_terms:
        overlap_penalty = _smooth_max(overlap_terms, beta=beta)
    else:
        overlap_penalty = 0.0
    
    return (side, overlap_penalty)

def _optimize_deformation(outer_angle_deg, max_iter=1000, deadline=145.0):
    """
    Optimize the deformation of the hexagon cluster using gradient-based methods.
    """
    start_time = time.time()
    
    # Initial guess: zero displacement
    x0 = np.zeros(2 * N)
    
    # Objective function for scipy.optimize
    def objective(x):
        if time.time() > deadline - 5:
            return 1e9
        side, penalty = _deformation_objective(x, outer_angle_deg)
        # Combined objective: minimize side, penalize overlap
        return side + 1000.0 * penalty
    
    # Try L-BFGS-B with bounds to keep displacements reasonable
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if time.time() > deadline - 5:
            return x0
        
        # Check if the result is valid
        side, penalty = _deformation_objective(result.x, outer_angle_deg)
        if penalty < 1e-6:  # No significant overlap
            return result.x
        else:
            # If there's overlap, try a smaller step
            return x0 * 0.1
    except Exception:
        return x0

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
        # Fallback: spread out slightly
        best_centers = best_centers * 1.001
        best_side = _min_enclosing_side(best_centers, inner_angles, best_outer_angle)
    
    # Strategy 1: Optimize outer rotation angle (1D search)
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
    
    # Strategy 2: Deformation optimization
    # Allow the hexagons to move independently to find a tighter packing
    if time.time() < DEADLINE - safety_margin:
        try:
            # First, optimize deformation for the current best outer angle
            displacements = _optimize_deformation(best_outer_angle, max_iter=500, deadline=DEADLINE)
            
            if np.any(displacements != 0):
                # Apply displacements to seed configuration
                seed_centers, seed_angles = _seed_configuration()
                new_centers = seed_centers.copy()
                for i in range(N):
                    new_centers[i, 0] += displacements[2*i]
                    new_centers[i, 1] += displacements[2*i + 1]
                
                # Check for overlap
                if _check_no_overlap(new_centers):
                    new_side = _min_enclosing_side(new_centers, seed_angles, best_outer_angle)
                    
                    if new_side < best_side - 1e-12:
                        # Now optimize outer angle for this new configuration
                        if time.time() < DEADLINE - 2:
                            try:
                                def obj_new(angle_deg):
                                    if time.time() > DEADLINE - 2:
                                        return 1e9
                                    return _min_enclosing_side(new_centers, seed_angles, angle_deg)
                                
                                res_new = minimize_scalar(obj_new, bounds=(0, 60), method='bounded',
                                                         options={'xatol': 1e-10})
                                if res_new.fun < best_side:
                                    best_side = res_new.fun
                                    best_outer_angle = res_new.x
                                    best_centers = new_centers
                                    inner_angles = seed_angles
                            except Exception:
                                best_side = new_side
                                best_centers = new_centers
                                inner_angles = seed_angles
                        else:
                            best_side = new_side
                            best_centers = new_centers
                            inner_angles = seed_angles
        except Exception:
            pass
    
    # Strategy 3: Try different inner orientations
    if time.time() < DEADLINE - safety_margin:
        for inner_rot in [0, 15, 30, 45, 60]:
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
    
    # Strategy 4: Fine-tune with small perturbations
    if time.time() < DEADLINE - safety_margin:
        for dx in [-0.005, 0.0, 0.005]:
            for dy in [-0.005, 0.0, 0.005]:
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
        # Fallback to seed
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
