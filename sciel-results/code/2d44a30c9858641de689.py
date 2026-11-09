import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    hex_centers = []
    
    # Row 0: 3 hexagons
    for i in range(3):
        hex_centers.append((i * sqrt3, 0.0))
    
    # Row 1: 4 hexagons
    for i in range(4):
        hex_centers.append((i * sqrt3 + sqrt3 / 2, 1.5))
    
    # Row 2: 3 hexagons
    for i in range(3):
        hex_centers.append((i * sqrt3 + sqrt3 / 2, 3.0))
    
    # Row 3: 2 hexagons
    for i in range(2):
        hex_centers.append((i * sqrt3 + sqrt3, 4.5))
    
    # Center the arrangement
    cx = np.mean([c[0] for c in hex_centers])
    cy = np.mean([c[1] for c in hex_centers])
    hex_centers = [(x - cx, y - cy) for x, y in hex_centers]
    
    # Generate all vertices of all inner hexagons (flat-top, angle=0)
    all_vertices = []
    for (x, y) in hex_centers:
        for k in range(6):
            angle = np.radians(60 * k)
            vx = x + np.cos(angle)
            vy = y + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # Find minimal enclosing regular hexagon
    best_s = float('inf')
    best_theta = 0.0
    best_hx = 0.0
    best_hy = 0.0
    
    # Coarse search over rotation
    for theta_deg in range(0, 60, 2):
        theta = np.radians(theta_deg)
        normals = np.array([
            [np.cos(theta + np.radians(60 * k)), np.sin(theta + np.radians(60 * k))]
            for k in range(6)
        ])
        
        projections = np.abs(all_vertices @ normals.T)
        max_proj = projections.max()
        s = max_proj / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_theta = theta_deg
    
    # Fine search over rotation
    for theta_deg in np.arange(best_theta - 2, best_theta + 2, 0.25):
        theta = np.radians(theta_deg)
        normals = np.array([
            [np.cos(theta + np.radians(60 * k)), np.sin(theta + np.radians(60 * k))]
            for k in range(6)
        ])
        
        projections = np.abs(all_vertices @ normals.T)
        max_proj = projections.max()
        s = max_proj / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_theta = theta_deg
    
    # Fine search over center
    for dx in np.arange(-0.5, 0.5, 0.1):
        for dy in np.arange(-0.5, 0.5, 0.1):
            theta = np.radians(best_theta)
            normals = np.array([
                [np.cos(theta + np.radians(60 * k)), np.sin(theta + np.radians(60 * k))]
                for k in range(6)
            ])
            
            shifted = all_vertices - np.array([dx, dy])
            projections = np.abs(shifted @ normals.T)
            max_proj = projections.max()
            s = max_proj / (np.sqrt(3) / 2)
            
            if s < best_s:
                best_s = s
                best_hx = dx
                best_hy = dy
    
    inner = [[float(x), float(y), 0.0] for x, y in hex_centers]
    return inner, (float(best_hx), float(best_hy)), float(best_s), float(best_theta)
# EVOLVE-BLOCK-END
