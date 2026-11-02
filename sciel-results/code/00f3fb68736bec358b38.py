import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Lattice parameters
    # 3x4 grid: i in 0..3, j in 0..2
    # Base vectors for hex centers (distance sqrt(3))
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    
    def get_centers(shear):
        centers = []
        sqrt3 = np.sqrt(3)
        for i in range(4):
            for j in range(3):
                x = i * sqrt3 + j * (sqrt3 / 2.0) + shear * j
                y = j * 1.5
                centers.append((x, y))
        return np.array(centers)

    def check_overlap(centers):
        # Min distance between centers must be >= sqrt(3)
        # To be safe, use slightly larger due to floating point
        min_dist = np.sqrt(3) - 1e-9
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    return False
        return True

    def cost(params):
        shear, alpha, beta = params
        
        centers = get_centers(shear)
        
        if not check_overlap(centers):
            return 100.0

        # Generate vertices of 12 hexagons
        # Each hexagon has 6 vertices
        # Rotation alpha
        # Side length 1, so radius 1
        vertices = []
        for cx, cy in centers:
            for k in range(6):
                angle = alpha + k * np.pi / 3.0
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                vertices.append((vx, vy))
        
        vertices = np.array(vertices)
        
        # Centroid of all vertices
        centroid = np.mean(vertices, axis=0)
        
        # Calculate required side length S for outer hexagon
        # Outer hexagon orientation beta
        # Normals are at beta + k * pi/3
        max_proj = 0.0
        for k in range(6):
            normal_angle = beta + k * np.pi / 3.0
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            
            # Project all vertices onto this normal
            # We want the max distance from centroid in this direction
            # Since centroid is center, we check max positive projection
            # Actually, the hexagon is centered at centroid.
            # The condition is n . (v - C) <= apothem
            # So apothem >= max(n . (v - C))
            # But we must consider both sides?
            # The max distance from center to a face in direction n is max(n.(v-C)).
            # If we take all 6 normals, we cover the whole hexagon.
            projs = vertices[:, 0] * nx + vertices[:, 1] * ny - (centroid[0] * nx + centroid[1] * ny)
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        # Apothem = S * sqrt(3) / 2
        # So S = 2 * apothem / sqrt(3)
        S = 2.0 * max_proj / np.sqrt(3)
        
        return S

    # Initial guess
    # Shear 0, Alpha 0, Beta 0
    # Try a few starting points to avoid local minima
    best_result = None
    best_cost = float('inf')
    
    starts = [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [-0.1, 0.2, 0.0],
        [0.0, 0.0, np.pi/6],
        [0.2, 0.0, 0.0]
    ]
    
    for start in starts:
        try:
            res = minimize(cost, start, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 200})
            if res.success and res.fun < best_cost:
                best_cost = res.fun
                best_result = res.x
        except Exception:
            continue

    if best_result is None:
        # Fallback to standard lattice
        shear, alpha, beta = 0.0, 0.0, 0.0
    else:
        shear, alpha, beta = best_result

    centers = get_centers(shear)
    
    # Center the inner hexagons around (0,0) for the return value
    centroid = np.mean(centers, axis=0)
    inner = []
    for cx, cy in centers:
        inner.append([cx - centroid[0], cy - centroid[1], np.degrees(alpha)])
    
    # Outer hexagon
    # Center is (0,0) because we centered the inner ones
    # Side length is best_cost
    # Angle is beta
    return inner, (0.0, 0.0), best_cost, np.degrees(beta)

# EVOLVE-BLOCK-END
