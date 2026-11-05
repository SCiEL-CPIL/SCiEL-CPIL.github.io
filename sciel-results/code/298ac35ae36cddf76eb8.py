import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

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
    
    # Strategy: 1D golden-section search over outer rotation angle
    # The inner hexagons maintain their relative positions (rigid cluster)
    # We only optimize the global rotation of the outer hexagon
    
    if time.time() < DEADLINE - safety_margin:
        try:
            # Search over outer angle in range [0, 60] due to 6-fold symmetry
            def objective(angle_deg):
                if time.time() > DEADLINE - 2:
                    return 1e9
                return _min_enclosing_side(best_centers, inner_angles, angle_deg)
            
            # Use minimize_scalar with golden-section method
            res = minimize_scalar(objective, bounds=(0, 60), method='bounded',
                                   options={'xatol': 1e-10})
            
            if res.fun < best_side:
                best_side = res.fun
                best_outer_angle = res.x
        except Exception:
            pass
    
    # Strategy 2: Try different inner orientations
    # The inner hexagons can be rotated. Try a few discrete orientations
    # and see if any gives a better outer side length
    
    if time.time() < DEADLINE - safety_margin:
        for inner_rot in [0, 15, 30, 45, 60]:
            test_angles = np.array([inner_rot] * 11)
            test_side = _min_enclosing_side(best_centers, test_angles, best_outer_angle)
            
            if test_side < best_side - 1e-12:
                # This orientation is better, now optimize outer angle for this orientation
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
    
    # Strategy 3: Small perturbations of the cluster
    # Try shifting the entire cluster slightly and see if it helps
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
