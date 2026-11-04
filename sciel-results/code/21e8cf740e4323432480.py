import numpy as np
import time
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _compute_required_side(centers, angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute minimum outer side length to contain all hexagons.
    The outer hexagon is defined by its center, side length, and rotation.
    For a regular hexagon with side L, the distance from center to a side (apothem) is L * sqrt(3) / 2.
    The distance from center to a vertex (circumradius) is L.
    To contain a point P, we project P onto the 6 normal axes of the outer hexagon.
    The maximum projection distance R_max must be <= L * sqrt(3) / 2.
    Thus L >= R_max * 2 / sqrt(3).
    """
    max_proj = 0.0
    oc = np.array(outer_center)
    oa = np.deg2rad(outer_angle)
    
    # Normals to the 6 sides of the outer hexagon
    # Side k is perpendicular to angle oa + k * 60 deg
    for k in range(6):
        angle = oa + k * np.pi / 3
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v - oc, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _objective(params):
    """
    Objective function for optimization.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    Returns: side_length + overlap_penalty + containment_penalty
    """
    centers = params[0::3].reshape(-1, 2)
    angles = params[1::3]
    
    # 1. Compute required side length
    side = _compute_required_side(centers, angles)
    
    # 2. Compute overlap penalty using SAT
    # We use a smooth approximation of the penetration depth or distance.
    # For SAT, if projections overlap on all axes, polygons intersect.
    # We sum the squared overlap depth for each axis where they intersect.
    penalty = 0.0
    
    for i in range(N):
        v1 = _hex_vertices(centers[i], 1.0, angles[i])
        for j in range(i + 1, N):
            v2 = _hex_vertices(centers[j], 1.0, angles[j])
            
            # Check axes from both polygons
            # To be fast, we can just check the 6 axes of v1 and 6 of v2.
            # If they are far apart, we can skip.
            dist_centers = np.linalg.norm(centers[i] - centers[j])
            if dist_centers > 3.5:
                continue
                
            overlap_sum = 0.0
            for k in range(6):
                # Axes from v1
                edge1 = v1[(k+1) % 6] - v1[k]
                normal1 = np.array([-edge1[1], edge1[0]])
                norm1 = np.linalg.norm(normal1)
                if norm1 < 1e-12: continue
                normal1 /= norm1
                
                proj1 = np.dot(v1, normal1)
                proj2 = np.dot(v2, normal1)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check separation
                if max1 < min2 or max2 < min1:
                    # Separated on this axis, no overlap contribution from this axis
                    # But for penalty, we want to know if they are close.
                    # If separated, contribution is 0.
                    pass
                else:
                    # Overlapping on this axis
                    # Depth of overlap
                    depth = min(max1, max2) - max(min1, min2)
                    if depth > 0:
                        overlap_sum += depth ** 2
                        
                # Axes from v2 (only needed if not already separated by v1 axes? 
                # Actually for SAT, if separated on ANY axis, they don't overlap.
                # If they overlap on all axes of v1, we must check v2 axes too to be sure.
                # However, for penalty, we can just sum overlaps on all 12 axes.
                # This is a valid penalty that is 0 iff they are disjoint (separated on at least one axis).
                
            for k in range(6):
                edge2 = v2[(k+1) % 6] - v2[k]
                normal2 = np.array([-edge2[1], edge2[0]])
                norm2 = np.linalg.norm(normal2)
                if norm2 < 1e-12: continue
                normal2 /= norm2
                
                proj1 = np.dot(v1, normal2)
                proj2 = np.dot(v2, normal2)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    pass
                else:
                    depth = min(max1, max2) - max(min1, min2)
                    if depth > 0:
                        overlap_sum += depth ** 2
            
            if overlap_sum > 0:
                penalty += overlap_sum * 10.0 # Scale factor

    # 3. Containment penalty
    # Check if any vertex is outside the outer hexagon defined by 'side'
    # Actually, since 'side' is computed to *contain* them, this penalty should be 0 by definition
    # of the side calculation IF the side calculation is correct.
    # Wait, _compute_required_side finds the MINIMUM side. So the hexagons are exactly touching or inside.
    # So containment penalty is naturally handled by the side length term.
    # However, numerical errors might cause issues. We'll rely on the side term.
    
    return side + penalty

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Initial configurations
    configs = []
    
    # Config 1: Honeycomb ring (1 center, 6 neighbors, 5 more)
    # Actually, let's use a standard hexagonal lattice patch.
    # 1 center, 6 around it, then 6 more around the gaps.
    centers1 = np.zeros((N, 2))
    angles1 = np.zeros(N)
    
    # Center at 0,0
    centers1[0] = [0, 0]
    # Ring 1: 6 hexes at distance sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        centers1[i+1] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    # Ring 2: 6 hexes at distance 2*sqrt(3), offset by 30 deg
    for i in range(5): # We only need 5 more to make 12
        angle = (i + 0.5) * np.pi / 3
        centers1[i+7] = [2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)]
    configs.append((centers1, angles1))
    
    # Config 2: Linear strip
    centers2 = np.zeros((N, 2))
    angles2 = np.zeros(N)
    for i in range(N):
        centers2[i] = [i * np.sqrt(3), 0]
    configs.append((centers2, angles2))
    
    # Config 3: 2x6 Grid
    centers3 = np.zeros((N, 2))
    angles3 = np.zeros(N)
    for i in range(6):
        centers3[i] = [i * np.sqrt(3), 0]
        centers3[i+6] = [i * np.sqrt(3), 3.0] # Height of hex is 2*sqrt(3)/2 * 2 = sqrt(3)*2? 
        # Hex height (flat to flat) is sqrt(3). So offset by sqrt(3) for touching?
        # Side 1 hex: width 2, height sqrt(3).
        # If aligned, vertical distance for touching is sqrt(3).
        centers3[i+6] = [i * np.sqrt(3), np.sqrt(3)]
    configs.append((centers3, angles3))
    
    # Config 4: Random
    centers4 = np.random.uniform(-3, 3, (N, 2))
    angles4 = np.random.uniform(0, 360, N)
    configs.append((centers4, angles4))
    
    # Run Nelder-Mead on each config
    for idx, (centers, angles) in enumerate(configs):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        params = np.column_stack((centers, angles)).flatten()
        
        # Nelder-Mead settings
        options = {
            'maxiter': 2000,
            'maxfev': 2000,
            'xatol': 1e-4,
            'fatol': 1e-4,
            'disp': False
        }
        
        try:
            result = minimize(_objective, params, method='Nelder-Mead', options=options)
        except Exception as e:
            continue
            
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        final_params = result.x
        final_centers = final_params[0::3].reshape(-1, 2)
        final_angles = final_params[1::3]
        
        final_side = _compute_required_side(final_centers, final_angles)
        
        # Verify validity
        valid = True
        for i in range(N):
            v1 = _hex_vertices(final_centers[i], 1.0, final_angles[i])
            # Check containment
            for v in v1:
                # Check if v is inside outer hexagon of side final_side
                # Outer hexagon center (0,0), angle 0
                # Simplified check: distance to origin < final_side is necessary but not sufficient for hex.
                # Proper check:
                inside = True
                for k in range(6):
                    angle = k * np.pi / 3
                    normal = np.array([np.cos(angle), np.sin(angle)])
                    # Apothem is final_side * sqrt(3) / 2
                    apothem = final_side * np.sqrt(3) / 2
                    proj = np.dot(v, normal)
                    if proj > apothem + 1e-9:
                        inside = False
                        break
                if not inside:
                    valid = False
                    break
            if not valid:
                break
                
            # Check overlap with others
            if valid:
                for j in range(i + 1, N):
                    v2 = _hex_vertices(final_centers[j], 1.0, final_angles[j])
                    # SAT check
                    separated = False
                    for k in range(6):
                        edge1 = v1[(k+1) % 6] - v1[k]
                        normal1 = np.array([-edge1[1], edge1[0]])
                        norm1 = np.linalg.norm(normal1)
                        if norm1 < 1e-12: continue
                        normal1 /= norm1
                        proj1 = np.dot(v1, normal1)
                        proj2 = np.dot(v2, normal1)
                        if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                            separated = True
                            break
                    if not separated:
                        for k in range(6):
                            edge2 = v2[(k+1) % 6] - v2[k]
                            normal2 = np.array([-edge2[1], edge2[0]])
                            norm2 = np.linalg.norm(normal2)
                            if norm2 < 1e-12: continue
                            normal2 /= norm2
                            proj1 = np.dot(v1, normal2)
                            proj2 = np.dot(v2, normal2)
                            if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                                separated = True
                                break
                    if not separated:
                        valid = False
                        break
            if not valid:
                break
                
        if valid and final_side < best_side:
            best_side = final_side
            best_inner = [[final_centers[i][0], final_centers[i][1], final_angles[i]] for i in range(N)]
            
    # Fallback
    if best_inner is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return (best_inner, best_outer_center, best_side, best_outer_angle)
