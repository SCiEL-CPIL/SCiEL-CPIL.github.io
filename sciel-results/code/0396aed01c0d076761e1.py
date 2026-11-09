import numpy as np

N = 11

def construct():
    # Generate centers for N=11 in a compact hexagonal cluster
    # Center (0,0)
    centers = [(0.0, 0.0)]
    
    # Layer 1: 6 neighbors
    # Using pointy-top hex grid coordinates (q, r)
    # x = sqrt(3) * (q + r/2), y = 1.5 * r
    # Neighbors of (0,0) are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    layer1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in layer1:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
        
    # Layer 2: 12 neighbors. We need 4 more.
    # To minimize the bounding box, pick 4 symmetrically.
    # The 12 positions in layer 2 are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-2)? No.
    # Let's generate all layer 2 points and pick 4 that are symmetric.
    # Symmetric choice: 0, 90, 180, 270 degrees? Or 30, 120, 210, 300?
    # Let's calculate the angles of the 12 layer 2 points.
    layer2_candidates = [
        (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
        (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-2) # Duplicates?
    ]
    # Correct layer 2 list:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
    # (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2) is wrong.
    # (q,r) with max(|q|,|r|,|q+r|) = 2
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
    # (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2) -> (2,-2) has q+r=0, max=2.
    # (1,-2) has q+r=-1, max=2.
    # So the 12 are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
    # (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-2) is duplicate?
    # (1,-2) and (2,-2) are distinct.
    # Let's just pick 4 that are 90 degrees apart in angle.
    # Calculate angles for all 12.
    pts = []
    for q, r in [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-2)]:
        # Remove duplicates
        pass
    
    # Simpler: Just pick (2,0), (0,2), (-2,0), (0,-2) which are at 0, 60, 180, 240?
    # (2,0) -> angle 0
    # (0,2) -> angle 60
    # (-2,0) -> angle 180
    # (0,-2) -> angle 240
    # This is not 90 deg apart.
    # Let's pick (2,0), (-1,2), (-2,0), (1,-2)?
    # (2,0) angle 0.
    # (-1,2) angle 120.
    # (-2,0) angle 180.
    # (1,-2) angle 240?
    # This is symmetric.
    
    selected_layer2 = [(2,0), (-1,2), (-2,0), (1,-2)]
    for q, r in selected_layer2:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
        
    # Now we have 11 centers.
    # Generate vertices of all inner hexagons.
    # Inner hexagons are unit side, aligned (angle 0).
    # Vertices of a unit hexagon at (cx, cy) with angle 0 (pointy top? or flat top?)
    # The problem says "regular hexagons of side length 1".
    # The orientation of the inner hexagons can be chosen.
    # To minimize the outer hexagon, we should align the inner hexagons with the outer one?
    # Or rotate them?
    # If we align all inner hexagons with each other and the outer one, it's usually good.
    # Let's assume all inner hexagons have angle 0.
    
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.pi / 3 * k
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
            
    V = np.array(all_vertices)
    
    # Find minimal enclosing regular hexagon
    # Rotate the outer hexagon by alpha.
    # The sides are at angles alpha, alpha+60, ...
    # The normal vectors are u_i.
    # The required apothem A is max(v . u_i).
    # L = A * 2 / sqrt(3).
    
    def calc_L(alpha):
        A = 0
        for i in range(6):
            theta = alpha + i * np.pi / 3
            u = np.array([np.cos(theta), np.sin(theta)])
            # Project all vertices
            proj = V @ u
            max_proj = np.max(proj)
            if max_proj > A:
                A = max_proj
        return A * 2 / np.sqrt(3)
        
    # Optimize alpha
    # The function is periodic with 60 degrees.
    # We can sample alpha in [0, 30 degrees] and find the minimum.
    alphas = np.linspace(0, np.pi/6, 1000)
    Ls = [calc_L(a) for a in alphas]
    min_L = min(Ls)
    
    # Return inner hexagons (x, y, angle)
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    # Outer center (0,0), side length min_L, angle 0 (or optimal alpha)
    # The angle of the outer hexagon is the alpha that gave min_L.
    # But the interface asks for (inner, center, side, angle).
    # The angle is the rotation of the outer hexagon.
    # We can just return 0.0 for angle if we assume the optimal alpha is 0?
    # Or return the optimal alpha.
    # Let's return the optimal alpha.
    best_alpha = alphas[np.argmin(Ls)]
    
    return inner, (0.0, 0.0), min_L, best_alpha

# EVOLVE-BLOCK-START
def construct():
    # Re-define to avoid global scope issues if needed, but the above is inside EVOLVE-BLOCK?
    # The prompt says "Keep the function names and signatures of the seed program."
    # The seed program has construct() inside EVOLVE-BLOCK.
    # I will put the code inside the block.
    import numpy as np
    
    N = 11
    centers = [(0.0, 0.0)]
    
    layer1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in layer1:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
        
    selected_layer2 = [(2,0), (-1,2), (-2,0), (1,-2)]
    for q, r in selected_layer2:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        centers.append((x, y))
        
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.pi / 3 * k
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
            
    V = np.array(all_vertices)
    
    def calc_L(alpha):
        A = 0
        for i in range(6):
            theta = alpha + i * np.pi / 3
            u = np.array([np.cos(theta), np.sin(theta)])
            proj = V @ u
            max_proj = np.max(proj)
            if max_proj > A:
                A = max_proj
        return A * 2 / np.sqrt(3)
        
    alphas = np.linspace(0, np.pi/6, 1000)
    Ls = [calc_L(a) for a in alphas]
    min_L = min(Ls)
    best_alpha = alphas[np.argmin(Ls)]
    
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    return inner, (0.0, 0.0), min_L, best_alpha
# EVOLVE-BLOCK-END
