import numpy as np

def construct():
    # Define the 11 hexagons
    # 1 Center
    centers = [[0.0, 0.0]]
    angles = [0.0]
    
    # 6 Ring 1
    # Neighbors are at distance sqrt(3) at angles 30, 90, 150, 210, 270, 330
    d1 = np.sqrt(3)
    for i in range(6):
        ang = np.deg2rad(30 + i * 60)
        x = d1 * np.cos(ang)
        y = d1 * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
        
    # 4 Ring 2 "edge" hexagons
    # These are at distance 3 from center, at angles 30, 150, 210, 330
    # (Skipping 90 and 270 to keep it compact/symmetric? Or maybe pick 4 that minimize max extent)
    # Actually, picking any 4 from the 6 "edge" ones results in similar max extent.
    # Let's pick 30, 150, 210, 330.
    d2 = 3.0
    indices = [0, 2, 3, 5] # Corresponds to 30, 150, 210, 330
    for i in indices:
        ang = np.deg2rad(30 + i * 60)
        x = d2 * np.cos(ang)
        y = d2 * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
        
    # Apply gap factor
    gap = 1.001
    for i in range(len(centers)):
        centers[i][0] *= gap
        centers[i][1] *= gap
        
    # Calculate minimal outer side length S
    # The outer hexagon is flat-top, aligned with the inner ones.
    # We need to find the smallest S such that all inner hexagons are inside.
    # This is equivalent to finding the max "radius" of the cluster in the hex metric.
    # For a flat-top hexagon, the constraint is determined by the distance to the boundary.
    # Boundary distance in direction theta: d_out(theta) = (S * sqrt(3) / 2) / cos(theta % 60 - 30)
    # We need d_inner(theta) <= d_out(theta) for all vertices of inner hexagons.
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for cx, cy in centers:
        for j in range(6):
            v_ang = np.deg2rad(j * 60)
            vx = cx + np.cos(v_ang)
            vy = cy + np.sin(v_ang)
            all_vertices.append((vx, vy))
            
    # For each vertex, calculate the required S
    max_s = 0.0
    for vx, vy in all_vertices:
        r = np.sqrt(vx*vx + vy*vy)
        theta = np.arctan2(vy, vx)
        if theta < 0:
            theta += 2 * np.pi
            
        # Normalize theta to 0-60 degrees
        theta_mod = theta % (np.pi / 3)
        # The angle relative to the nearest normal (which is at 30 deg, 90 deg, etc)
        # The normal to the edge at 30 deg is at 30 deg.
        # The distance to the edge is r_out = S * sqrt(3) / 2.
        # The projection of the vertex onto the normal direction must be <= r_out.
        # Normal direction is theta_mod + 30? No.
        # If theta is in [0, 60], the closest normal is at 30.
        # Angle diff = theta - 30?
        # Let's use the formula: S_req = (2 * r * cos(theta_mod - 30)) / sqrt(3) ?
        # Wait, if theta = 30, cos(0) = 1, S = 2r/sqrt(3).
        # If theta = 0, cos(-30) = sqrt(3)/2, S = 2r(sqrt(3)/2)/sqrt(3) = r.
        # This matches: at theta=0 (vertex direction), the boundary is at distance S.
        # So S_req = r / cos(theta - 30) ?
        # Let's check.
        # Distance to boundary in direction theta is d = r_out / cos(theta - 30).
        # We need d >= r.
        # r_out / cos(theta - 30) >= r
        # (S * sqrt(3) / 2) / cos(theta - 30) >= r
        # S >= 2 * r * cos(theta - 30) / sqrt(3)
        
        # theta_mod is in [0, 60]. The normal is at 30.
        # So the angle difference is theta_mod - 30.
        # But we need to be careful with the range.
        # If theta is in [0, 60], the normal is at 30.
        # If theta is in [60, 120], the normal is at 90.
        # So we should map theta to the nearest multiple of 60 + 30.
        
        # Let's just compute the distance to the 6 edges.
        # Edge 1: Normal at 30 deg.
        # Project vertex onto normal.
        # n = (cos(30), sin(30))
        # proj = vx * cos(30) + vy * sin(30)
        # If proj > r_out, it's outside.
        # r_out = S * sqrt(3) / 2
        # S >= 2 * proj / sqrt(3)
        
        # We need to check all 6 edges.
        # The required S is the max over all edges of (2 * proj_i / sqrt(3))
        # where proj_i is the projection of the vertex onto the normal of edge i.
        # If the projection is negative, it's inside that half-plane, so no constraint (or constraint is 0).
        
        s_req = 0.0
        for k in range(6):
            ang_n = np.deg2rad(30 + k * 60)
            nx = np.cos(ang_n)
            ny = np.sin(ang_n)
            proj = vx * nx + vy * ny
            if proj > 0:
                s_val = 2 * proj / np.sqrt(3)
                if s_val > s_req:
                    s_req = s_val
        if s_req > max_s:
            max_s = s_req
            
    return centers, (0.0, 0.0), max_s, 0.0
