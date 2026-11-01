import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Candidate 1: 4-4-3 block
    c1 = []
    for i in range(4): c1.append((i * sqrt3, 0))
    for i in range(4): c1.append((i * sqrt3 + sqrt3/2, 1.5))
    for i in range(3): c1.append((i * sqrt3, 3.0))
    
    # Candidate 2: 3-5-3 block
    c2 = []
    for i in range(3): c2.append((i * sqrt3, 0))
    for i in range(5): c2.append((i * sqrt3 + sqrt3/2, 1.5))
    for i in range(3): c2.append((i * sqrt3, 3.0))
    
    # Candidate 3: 2-5-4 block?
    c3 = []
    for i in range(2): c3.append((i * sqrt3, 0))
    for i in range(5): c3.append((i * sqrt3 + sqrt3/2, 1.5))
    for i in range(4): c3.append((i * sqrt3, 3.0))
    
    candidates = [c1, c2, c3]
    
    def get_min_side(points):
        points = np.array(points)
        center = points.mean(axis=0)
        points_centered = points - center
        
        # 6 normals for hexagon
        angles = np.deg2rad(np.arange(0, 360, 60))
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        min_inradius = float('inf')
        # Try rotating the container hexagon
        for rot in np.linspace(0, np.pi/6, 7):
            c, s = np.cos(rot), np.sin(rot)
            R = np.array([[c, -s], [s, c]])
            normals_rot = normals @ R.T
            proj = points_centered @ normals_rot.T
            max_proj = np.max(proj)
            if max_proj < min_inradius:
                min_inradius = max_proj
                
        # Side length S = 2/sqrt(3) * inradius + 1
        side = 2.0 / np.sqrt(3) * min_inradius + 1.0
        return side, center
        
    best_side = float('inf')
    best_center = (0,0)
    best_points = c1
    
    for cand in candidates:
        side, center = get_min_side(cand)
        if side < best_side:
            best_side = side
            best_center = center
            best_points = cand
            
    inner = [[p[0], p[1], 0.0] for p in best_points]
    return inner, (best_center[0], best_center[1]), best_side, 0.0
# EVOLVE-BLOCK-END
