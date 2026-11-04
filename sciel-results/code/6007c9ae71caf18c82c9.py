import numpy as np

N = 11

def construct():
    # Generate hexagonal lattice points
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    # Spacing is sqrt(3) for neighbors
    
    points = []
    max_radius = 10
    for r in range(-max_radius, max_radius + 1):
        for q in range(-max_radius, max_radius + 1):
            # Hex distance
            dist = 0.5 * (abs(q) + abs(r) + abs(q + r))
            if dist <= max_radius:
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                points.append((x, y))
    
    # Sort by distance from origin to get compact cluster
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take N points
    selected = points[:N]
    
    # Center the points
    xs = [p[0] for p in selected]
    ys = [p[1] for p in selected]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    # Centered points
    pts = [(x - cx, y - cy) for x, y in selected]
    
    # Optimize rotation angle theta
    # S(theta) = 1 + (1/sqrt(3)) * max_k Width_k(theta)
    # Width_k is the projection of pts onto normal k
    
    best_theta = 0.0
    best_S = float('inf')
    
    # Normals for a regular hexagon with side S, apothem A = S*sqrt(3)/2
    # The normals are at angles theta, theta+60, ...
    # We search theta in [0, pi/3)
    
    num_steps = 200
    for i in range(num_steps):
        theta = i * (np.pi / 3.0) / num_steps
        
        max_width = 0.0
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            nx = np.cos(angle)
            ny = np.sin(angle)
            
            projections = [x * nx + y * ny for x, y in pts]
            w = max(projections) - min(projections)
            if w > max_width:
                max_width = w
        
        S = 1.0 + max_width / np.sqrt(3.0)
        
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Refine with finer search around best_theta
    # (Optional, but good for precision)
    # For now, 200 steps is likely sufficient for this scale
    
    inner = []
    for x, y in selected:
        # Angle in degrees
        angle_deg = np.deg2rad(best_theta) * 180.0 / np.pi
        inner.append([x, y, angle_deg])
        
    center = (cx, cy)
    side = best_S
    angle = best_theta
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    N = 11
    
    # Generate hexagonal lattice points
    points = []
    max_radius = 10
    for r in range(-max_radius, max_radius + 1):
        for q in range(-max_radius, max_radius + 1):
            # Hex distance
            dist = 0.5 * (abs(q) + abs(r) + abs(q + r))
            if dist <= max_radius:
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                points.append((x, y))
    
    # Sort by distance from origin to get compact cluster
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take N points
    selected = points[:N]
    
    # Center the points
    xs = [p[0] for p in selected]
    ys = [p[1] for p in selected]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    # Centered points for width calculation
    pts = [(x - cx, y - cy) for x, y in selected]
    
    best_theta = 0.0
    best_S = float('inf')
    
    # Search for optimal rotation
    # S(theta) = 1 + (1/sqrt(3)) * max_width(theta)
    num_steps = 360
    for i in range(num_steps):
        theta = i * (np.pi / 3.0) / num_steps
        
        max_width = 0.0
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            nx = np.cos(angle)
            ny = np.sin(angle)
            
            min_proj = float('inf')
            max_proj = float('-inf')
            for x, y in pts:
                proj = x * nx + y * ny
                if proj < min_proj:
                    min_proj = proj
                if proj > max_proj:
                    max_proj = proj
            
            w = max_proj - min_proj
            if w > max_width:
                max_width = w
        
        S = 1.0 + max_width / np.sqrt(3.0)
        
        if S < best_S:
            best_S = S
            best_theta = theta
            
    inner = []
    angle_deg = np.rad2deg(best_theta)
    for x, y in selected:
        inner.append([x, y, angle_deg])
        
    center = (cx, cy)
    side = best_S
    angle = best_theta
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
