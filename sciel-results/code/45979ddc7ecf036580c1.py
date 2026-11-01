import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Flower packing axial coords
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    pts = axial_to_cartesian(axial)
    
    # Center the points
    cx_mean = np.mean(pts[:, 0])
    cy_mean = np.mean(pts[:, 1])
    pts_centered = pts - np.array([cx_mean, cy_mean])

    best_S = float('inf')
    best_angle = 0.0
    best_center = [0.0, 0.0]
    best_pts = pts_centered

    # Try several rotations
    for angle_deg in range(0, 60, 5):
        beta = math.radians(angle_deg)
        
        # Calculate required S for this rotation
        # Outer hexagon normals
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            # Project centered points
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            # The hexagon extends by APOTHEM
            # So max distance from center to boundary of inner hex is max_proj + APOTHEM
            # This must be <= Outer Apothem = S * APOTHEM
            # S * APOTHEM >= max_proj + APOTHEM
            # S >= (max_proj + APOTHEM) / APOTHEM
            current_max = np.max(projs)
            current_min = np.min(projs)
            # We need to cover both positive and negative projections
            # Actually, max_proj should be the max of abs(projections)
            # Because the outer hexagon is centered at 0.
            # The condition is: for all normals n, max(p.n) <= S*APOTHEM - APOTHEM
            # AND min(p.n) >= -(S*APOTHEM - APOTHEM)
            # So S*APOTHEM - APOTHEM >= max(|max(p.n)|, |min(p.n)|)
            # S >= (max_abs_proj + APOTHEM) / APOTHEM
            
            max_abs = max(abs(current_max), abs(current_min))
            if max_abs > max_proj:
                max_proj = max_abs
        
        S = (max_proj + APOTHEM) / APOTHEM
        
        if S < best_S:
            best_S = S
            best_angle = angle_deg
            best_pts = pts_centered

    # To ensure disjointness, scale the inner points by 1.001
    # This creates a gap between hexagons.
    scale_factor = 1.001
    final_pts = best_pts * scale_factor
    
    # Recalculate S for the scaled points
    beta = math.radians(best_angle)
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj_final = 0.0
    for nx, ny in normals:
        projs = final_pts[:, 0] * nx + final_pts[:, 1] * ny
        current_max = np.max(projs)
        current_min = np.min(projs)
        max_abs = max(abs(current_max), abs(current_min))
        if max_abs > max_proj_final:
            max_proj_final = max_abs
            
    final_S = (max_proj_final + APOTHEM) / APOTHEM
    
    # Prepare output
    inner = []
    for x, y in final_pts:
        inner.append([float(x), float(y), float(best_angle)])
        
    center = [0.0, 0.0]
    side = float(final_S)
    rotation = float(best_angle)
    
    return inner, center, side, rotation
