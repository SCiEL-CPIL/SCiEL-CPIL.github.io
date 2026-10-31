import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def construct():
    # Axial coordinates for 11 hexagons
    # Center + 6 surrounding + 4 from next layer
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
        
    pts = np.array(pts)
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid # Center at origin

    # We will optimize the rotation of the container.
    # We assume all inner hexagons are aligned (rotation 0) for simplicity,
    # but we can try to optimize their rotations if time permits.
    # For now, assume all inner hexagons have rotation 0.
    
    def calc_S(beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Container normals
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_S = 0.0
        for nx, ny in normals:
            # Project centers
            projections = pts[:, 0] * nx + pts[:, 1] * ny
            # Add max projection of hexagon in this direction
            # For a hexagon with rotation 0, the support function h(phi) is:
            # We need the angle of the normal relative to the hexagon's axes.
            # Hexagon vertices are at 0, 60, ...
            # Normal angle is alpha = atan2(ny, nx)
            # Relative angle to vertex: delta = alpha mod 60
            # h(delta) = cos(delta) / cos(pi/6) ? No.
            # h(delta) = R * cos(delta) / cos(pi/6) is for the side?
            # Actually, h(delta) = 1 / cos(pi/6) * cos(delta) is wrong.
            # h(delta) = 1 / cos(pi/6) * cos(delta - delta_vertex)?
            # Let's use the formula: h(theta) = R * cos(theta - theta_v) / cos(pi/6) is not right.
            # The support function of a regular hexagon with circumradius R=1 is:
            # h(phi) = 1 / cos(pi/6) * cos(phi - phi_0) where phi_0 is angle to vertex?
            # No, h(phi) = a / cos(phi - phi_a) where phi_a is angle to apothem?
            # h(phi) = 1 / cos(pi/6) * cos(phi) is for the side?
            # Let's just compute it:
            # The hexagon is defined by 6 sides.
            # h(phi) = max( dot(v_i, u) )
            # Vertices are at angles 0, 60, ...
            # h(phi) = max( cos(phi - k*pi/3) )
            # This is the support function for a polygon with vertices at unit distance?
            # Yes, if vertices are at distance 1.
            # So h(phi) = max_{k} cos(phi - k*pi/3)
            
            # For each point, we need max( p_i . n + h(angle(n) - rot_i) )
            # Since rot_i = 0, h(angle(n))
            # We can precompute h for the 6 normals? No, beta changes.
            
            # For a given normal n, the angle is alpha.
            # h(alpha) = max(cos(alpha), cos(alpha-pi/3), ...)
            # This is periodic with pi/3.
            # h(alpha) = cos( (alpha mod pi/3) - pi/6 ) / cos(pi/6) ?
            # Let's just use the max of cos.
            
            alpha = math.atan2(ny, nx)
            # Reduce alpha to [0, pi/3]
            alpha_mod = alpha % (math.pi / 3.0)
            # The support function is maximized when the normal aligns with a vertex.
            # h(alpha) = cos(alpha_mod - pi/6) / cos(pi/6) is incorrect.
            # h(alpha) = 1 / cos(pi/6) * cos(alpha_mod) is incorrect.
            # h(alpha) = cos(alpha_mod) / cos(pi/6) is for the apothem?
            # Let's use the explicit max.
            h_val = max(math.cos(alpha - k * math.pi / 3.0) for k in range(6))
            
            # Max projection of any hexagon
            max_proj = np.max(projections) + h_val
            abs_max_proj = max(abs(max_proj), abs(-np.min(projections) + h_val)) # Symmetric?
            # Since we centered, and shape is roughly symmetric, we can just take max abs projection + h_val
            # But h_val is always positive.
            # We need max( p_i . n + h_val, - (p_i . n) + h_val )?
            # The container is centered at 0.
            # The condition is -A <= p_i . n + h_val <= A
            # So A >= max( max_i (p_i . n + h_val), max_i (-p_i . n + h_val) )
            
            pos_max = np.max(projections) + h_val
            neg_max = -np.min(projections) + h_val
            required_A = max(pos_max, neg_max)
            
            S = (2.0 / math.sqrt(3.0)) * required_A
            if S > max_S:
                max_S = S
        
        return max_S

    try:
        res = minimize_scalar(calc_S, bounds=(0, math.pi / 6), method='bounded', options={'maxiter': 100})
        best_beta = res.x
        best_S = res.fun
    except Exception:
        best_beta = 0.0
        best_S = calc_S(0.0)
        
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        # We assumed rot_i = 0. But we can rotate them to best fit?
        # If we rotate them, we might need to change the container rotation.
        # For now, keep rot_i = 0.
        inner.append([float(x), float(y), 0.0])
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
