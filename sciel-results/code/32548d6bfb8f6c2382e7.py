import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        return x, y

    centers = []
    centers.append(axial_to_cartesian(0, 0))
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in neighbors:
        centers.append(axial_to_cartesian(q, r))
    extended = [(0,2), (-1,2), (-2,0), (1,-2)]
    for q, r in extended:
        centers.append(axial_to_cartesian(q, r))
        
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Precompute vertices of inner hexagons
    all_vertices = []
    for x, y, _ in inner:
        for v in range(6):
            v_angle = np.radians(60 * v)
            vx = x + np.cos(v_angle)
            vy = y + np.sin(v_angle)
            all_vertices.append((vx, vy))
            
    all_vertices = np.array(all_vertices)
    
    best_S = float('inf')
    best_angle = 0.0
    
    # Try a few angles
    for theta_deg in np.linspace(0, 60, 13):
        theta = np.radians(theta_deg)
        max_proj = 0
        for k in range(6):
            angle = theta + np.radians(60 * k)
            nx = np.cos(angle)
            ny = np.sin(angle)
            projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_proj = max(max_proj, np.max(projs))
        S = max_proj * 2 / sqrt3
        if S < best_S:
            best_S = S
            best_angle = theta_deg
            
    return inner, (0.0, 0.0), best_S, best_angle
# EVOLVE-BLOCK-END
