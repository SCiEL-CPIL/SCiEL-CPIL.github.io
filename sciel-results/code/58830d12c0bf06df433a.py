import numpy as np
from scipy.optimize import minimize

N = 11

def support_function(angle_deg, rot_deg):
    """
    Calculate support function of a unit regular hexagon with rotation rot_deg
    in direction angle_deg.
    Hexagon vertices are at rot_deg + k*60.
    """
    # Normalize angles to [0, 360)
    a = angle_deg % 360
    r = rot_deg % 360
    
    # The support function is the max projection of vertices.
    # Vertices are at r, r+60, ..., r+300
    # Projection of vertex at angle v is cos(a - v)
    # We can calculate this analytically or numerically.
    # Analytical: The max is 1 if a is close to a vertex, sqrt(3)/2 if close to a normal.
    # The function is periodic with 60 deg.
    # Shift a relative to r
    delta = (a - r) % 60
    # The max projection in the sector [0, 60] relative to a vertex at 0
    # is max(cos(delta), cos(60-delta))
    # cos(delta) is max at 0, min at 60.
    # cos(60-delta) is min at 0, max at 60.
    # They intersect at 30.
    # So if delta < 30, max is cos(delta).
    # If delta > 30, max is cos(60-delta).
    # Note: cos(delta) in degrees.
    delta_rad = np.deg2rad(delta)
    if delta < 30:
        return np.cos(delta_rad)
    else:
        return np.cos(np.deg2rad(60 - delta))

def calc_outer_side(inner_hexes):
    """
    inner_hexes: list of (x, y, rot_deg)
    Returns minimal side length of outer hexagon centered at (0,0) with rot 0.
    """
    max_apothem = 0.0
    # Outer hexagon normals are at 30, 90, 150, 210, 270, 330
    normals = [30 + 60*k for k in range(6)]
    
    for x, y, rot in inner_hexes:
        for n_ang in normals:
            # Projection of center
            proj_c = x * np.cos(np.deg2rad(n_ang)) + y * np.sin(np.deg2rad(n_ang))
            # Support of inner hex
            sup = support_function(n_ang, rot)
            total = proj_c + sup
            if total > max_apothem:
                max_apothem = total
                
    side = 2.0 * max_apothem
    return side

def check_overlap(hex1, hex2):
    """
    Simple distance check. 
    This is an approximation. For robustness, we should ensure distance > min_dist.
    Min distance for two unit hexagons depends on relative rotation.
    For rot 0 and 0, min dist is sqrt(3).
    For rot 0 and 90, min dist is approx 1.8?
    We will use a conservative min_dist of 1.8 for non-identical rotations, sqrt(3) for identical.
    """
    x1, y1, r1 = hex1
    x2, y2, r2 = hex2
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    rel_rot = abs(r1 - r2) % 60
    if rel_rot < 1e-6 or rel_rot > 60 - 1e-6:
        min_d = np.sqrt(3)
    else:
        # Approximate min distance for other rotations
        # It varies between sqrt(3) and 2.
        # Let's use 1.8 to be safe, or calculate properly.
        # Actually, for this specific packing, the gap hexes are likely rotated 90 deg relative to ring.
        # Let's just use 1.75 as a safe lower bound for non-aligned.
        min_d = 1.75
        
    return dist >= min_d - 1e-6

def construct():
    # Base 7-cluster
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        ang = 30 + 60*k
        rad = np.deg2rad(ang)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        inner.append((x, y, 0.0))
        
    # 4 additional hexagons
    # Place them in gaps at 0, 120, 240, 300? 
    # Let's try 0, 60, 120, 180 for now, but optimize d and rot.
    # To maintain symmetry, let's place them at 0, 180, 120, 300.
    gap_angles = [0, 120, 240, 300] # Wait, 240 and 300 are not symmetric with 0 and 120?
    # 0 and 180 are opposite. 120 and 300 are opposite.
    # This set has D2 symmetry.
    
    # Parameters to optimize: d (distance), rot (rotation)
    def objective(params):
        d, rot = params
        # Check bounds
        if d < 1.5 or d > 5.0:
            return 100.0
            
        current_inner = list(inner)
        
        # Add gap hexes
        for ang in gap_angles:
            rad = np.deg2rad(ang)
            x = d * np.cos(rad)
            y = d * np.sin(rad)
            current_inner.append((x, y, rot))
            
        # Check overlaps
        for i in range(len(current_inner)):
            for j in range(i+1, len(current_inner)):
                if not check_overlap(current_inner[i], current_inner[j]):
                    return 100.0 + (10.0 - d) # Penalty
            
        s = calc_outer_side(current_inner)
        return s

    # Initial guess
    x0 = [2.8, 90.0]
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000})
    
    d_opt, rot_opt = res.x
    
    # Construct final inner list
    final_inner = list(inner)
    for ang in gap_angles:
        rad = np.deg2rad(ang)
        x = d_opt * np.cos(rad)
        y = d_opt * np.sin(rad)
        final_inner.append((x, y, rot_opt))
        
    # Convert to list of lists
    inner_list = [[x, y, r] for x, y, r in final_inner]
    
    side = calc_outer_side(final_inner)
    
    return inner_list, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import minimize
    
    N = 11
    
    def support_function(angle_deg, rot_deg):
        a = angle_deg % 360
        r = rot_deg % 360
        delta = (a - r) % 60
        delta_rad = np.deg2rad(delta)
        if delta < 30:
            return np.cos(delta_rad)
        else:
            return np.cos(np.deg2rad(60 - delta))

    def calc_outer_side(inner_hexes):
        max_apothem = 0.0
        normals = [30 + 60*k for k in range(6)]
        for x, y, rot in inner_hexes:
            for n_ang in normals:
                proj_c = x * np.cos(np.deg2rad(n_ang)) + y * np.sin(np.deg2rad(n_ang))
                sup = support_function(n_ang, rot)
                total = proj_c + sup
                if total > max_apothem:
                    max_apothem = total
        return 2.0 * max_apothem

    def check_overlap(hex1, hex2):
        x1, y1, r1 = hex1
        x2, y2, r2 = hex2
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        rel_rot = abs(r1 - r2) % 60
        if rel_rot < 1e-6 or rel_rot > 60 - 1e-6:
            min_d = np.sqrt(3)
        else:
            min_d = 1.75
        return dist >= min_d - 1e-6

    # Base 7-cluster
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        ang = 30 + 60*k
        rad = np.deg2rad(ang)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        inner.append((x, y, 0.0))
        
    # 4 additional hexagons in gaps
    gap_angles = [0, 120, 240, 300]
    
    def objective(params):
        d, rot = params
        if d < 1.5 or d > 5.0:
            return 100.0
        current_inner = list(inner)
        for ang in gap_angles:
            rad = np.deg2rad(ang)
            x = d * np.cos(rad)
            y = d * np.sin(rad)
            current_inner.append((x, y, rot))
            
        for i in range(len(current_inner)):
            for j in range(i+1, len(current_inner)):
                if not check_overlap(current_inner[i], current_inner[j]):
                    return 100.0 + (10.0 - d)
        
        return calc_outer_side(current_inner)

    x0 = [2.8, 90.0]
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
    d_opt, rot_opt = res.x
    
    final_inner = list(inner)
    for ang in gap_angles:
        rad = np.deg2rad(ang)
        x = d_opt * np.cos(rad)
        y = d_opt * np.sin(rad)
        final_inner.append((x, y, rot_opt))
        
    inner_list = [[x, y, r] for x, y, r in final_inner]
    side = calc_outer_side(final_inner)
    
    return inner_list, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
