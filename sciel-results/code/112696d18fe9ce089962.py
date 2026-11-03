import numpy as np

N = 12

def construct():
    # Generate candidate packings
    candidates = []
    
    # Candidate 1: 4x3 grid
    centers = []
    for j in range(3):
        for i in range(4):
            x = i * np.sqrt(3) + (j % 2) * np.sqrt(3) / 2
            y = j * 1.5
            centers.append((x, y))
    candidates.append(centers)
    
    # Candidate 2: 3x4 grid
    centers = []
    for j in range(4):
        for i in range(3):
            x = i * np.sqrt(3) + (j % 2) * np.sqrt(3) / 2
            y = j * 1.5
            centers.append((x, y))
    candidates.append(centers)
    
    # Candidate 3: Hexagonal 7 + 5 (manual tweak or similar)
    # Let's try a "cross" or other shape if needed, but grids are good.
    
    best_side = float('inf')
    best_inner = None
    best_angle = 0.0
    
    for centers in candidates:
        # Center the configuration
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        centered_centers = [(c[0]-cx, c[1]-cy) for c in centers]
        
        # Generate all vertices
        points = []
        for (cx_i, cy_i) in centered_centers:
            for k in range(6):
                angle = np.pi/3 * k # Pointy top vertices? 
                # Standard hex vertex at angle 0 is (1,0).
                # If we use pointy-top, vertices are at 30, 90, ...
                # Let's use flat-top vertices: 0, 60, 120...
                # It doesn't matter as long as consistent.
                # Let's use vertices at angles 0, 60, 120, 180, 240, 300.
                vx = cx_i + np.cos(np.pi/3 * k)
                vy = cy_i + np.sin(np.pi/3 * k)
                points.append((vx, vy))
        
        points = np.array(points)
        
        # Find min enclosing hexagon
        min_s = float('inf')
        min_phi = 0.0
        # Search phi in [0, pi/3]
        phis = np.linspace(0, np.pi/3, 500)
        for phi in phis:
            # Normals for outer hexagon
            # Outer hexagon sides are perpendicular to normals.
            # If rotation is phi, normals are at phi + k*pi/3
            normals = []
            for k in range(6):
                ang = phi + k * np.pi / 3
                normals.append((np.cos(ang), np.sin(ang)))
            normals = np.array(normals)
            
            # Project points onto normals
            # points: (N, 2), normals: (6, 2)
            # projections: (N, 6)
            projections = points @ normals.T
            max_proj = np.max(projections)
            # Apothem a = max_proj
            # Side S = 2 * a / sqrt(3)
            s = 2 * max_proj / np.sqrt(3)
            if s < min_s:
                min_s = s
                min_phi = phi
        
        if min_s < best_side:
            best_side = min_s
            best_inner = [[cx_i, cy_i, 0.0] for (cx_i, cy_i) in centered_centers]
            best_angle = min_phi
    
    return best_inner, (0.0, 0.0), best_side, best_angle
