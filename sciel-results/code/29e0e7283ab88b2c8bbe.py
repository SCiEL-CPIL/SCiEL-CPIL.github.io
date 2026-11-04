import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial configuration: 3x4 hexagonal grid
    # Lattice vectors for unit hexagons (side 1)
    # Distance between centers is sqrt(3)
    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    centers = []
    angles = []
    # 3 rows of 4
    for i in range(3):
        for j in range(4):
            c = i * u + j * v
            centers.append(c)
            angles.append(0.0)
            
    centers = np.array(centers)
    angles = np.array(angles)
    
    # Center the configuration
    centers -= np.mean(centers, axis=0)
    
    # Function to compute vertices of a hexagon
    def get_hex_vertices(center, angle, side=1.0):
        # Regular hexagon with flat top if angle=0?
        # Standard orientation: vertex at (1,0) rotated by angle?
        # Let's use flat top for angle=0.
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center?
        # If flat top, vertices are at 30, 90, 150, 210, 270, 330?
        # Let's stick to a consistent definition.
        # Let's say angle 0 means a vertex is at (1, 0).
        # Then vertices are at k * 60 + angle.
        theta = np.deg2rad(angle)
        vs = np.zeros((6, 2))
        for k in range(6):
            a = theta + k * np.pi / 3
            vs[k, 0] = center[0] + side * math.cos(a)
            vs[k, 1] = center[1] + side * math.sin(a)
        return vs

    # Function to compute minimal outer side length L for given centers, angles, and outer angle Phi
    def calc_L(centers, angles, Phi_deg):
        Phi = np.deg2rad(Phi_deg)
        # Normals to the sides of the outer hexagon
        # If outer hexagon has flat top, normals are at 0, 60, 120...
        # Let's define outer hexagon such that its normals are at Phi + k*60.
        normals = []
        for k in range(6):
            a = Phi + k * np.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            vs = get_hex_vertices(centers[i], angles[i])
            all_verts.append(vs)
        all_verts = np.vstack(all_verts)
        
        max_L = 0.0
        for n in normals:
            # Project vertices onto normal
            projections = all_verts @ n
            w = np.max(projections) - np.min(projections)
            # Required side length for this width
            # Width of regular hexagon side L is L * sqrt(3)
            L_req = w / math.sqrt(3)
            if L_req > max_L:
                max_L = L_req
        return max_L

    # Function to check overlaps
    # Two hexagons overlap if the distance between centers is less than the sum of their support functions?
    # Simpler: Check if any vertex of one is inside the other?
    # Or use the Minkowski sum property.
    # For efficiency, we can check distance between centers.
    # If distance < 2 * apothem? No, depends on relative angle.
    # Let's use a simple point-in-polygon check for vertices, or just rely on the optimizer to keep them apart.
    # Given the initial grid is valid, and we do small steps, we can check validity at the end.
    # But if we move them, they might overlap.
    # Let's implement a simple overlap check using scipy.spatial or manual.
    # Manual: For each pair, check if distance between centers < min_dist(angle_diff).
    # min_dist for two unit hexagons with relative angle alpha.
    # This is complex. Let's just check if the convex hull of one intersects the other?
    # Or simply: The problem statement says "disjoint".
    # We will assume the optimizer stays in a valid region or we add a penalty.
    
    # Optimization
    # We will optimize centers and outer angle.
    # Variables: x0...x11, y0...y11, Phi
    # Total 25 variables.
    
    def objective(vars):
        if time.time() > deadline:
            return 1e9
        cs = vars[:24].reshape(12, 2)
        Phi = vars[24]
        return calc_L(cs, angles, Phi)
        
    # Initial variables
    init_vars = np.zeros(25)
    init_vars[:24] = centers.flatten()
    init_vars[24] = 0.0
    
    # Use scipy.optimize.minimize
    try:
        from scipy.optimize import minimize
        # Nelder-Mead is robust for non-smooth functions
        res = minimize(objective, init_vars, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        best_vars = res.x
    except Exception as e:
        best_vars = init_vars
        
    # Extract best solution
    best_centers = best_vars[:24].reshape(12, 2)
    best_Phi = best_vars[24]
    
    # Refine: We can also optimize angles if time permits
    # For now, return with optimized centers and Phi.
    # The outer hexagon center is the mean of the bounding box?
    # Actually, the outer hexagon center should be chosen to center the inner cluster.
    # Let's compute the center of the bounding box of the inner vertices.
    all_verts = []
    for i in range(N):
        vs = get_hex_vertices(best_centers[i], angles[i])
        all_verts.append(vs)
    all_verts = np.vstack(all_verts)
    center_x = (np.max(all_verts[:, 0]) + np.min(all_verts[:, 0])) / 2
    center_y = (np.max(all_verts[:, 1]) + np.min(all_verts[:, 1])) / 2
    outer_center = (center_x, center_y)
    
    # Recalculate L with this center?
    # The calc_L function assumed outer center at (0,0) effectively by using width.
    # Width is translation invariant. So L is correct.
    # The outer center returned should be the center of the hexagon.
    
    # Final check: ensure no overlaps.
    # If overlaps exist, the score might be invalid or penalized?
    # The task says "Place 12 disjoint...".
    # If they overlap, it's an invalid result.
    # Let's add a simple check. If overlap, revert to initial grid.
    
    def check_overlap(c1, a1, c2, a2):
        # Check if hexagon 1 and 2 overlap
        # Approximation: distance between centers
        d = np.linalg.norm(c1 - c2)
        # Min distance for non-overlap of two unit hexagons
        # If angles are 0, min dist is sqrt(3) (touching sides) or 2 (touching vertices)?
        # In a grid, dist is sqrt(3).
        # If d < sqrt(3) - epsilon, likely overlap.
        # This is a rough check.
        return d < math.sqrt(3) - 1e-3

    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap(best_centers[i], angles[i], best_centers[j], angles[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Revert to initial
        best_centers = centers
        best_Phi = 0.0
        outer_center = (0.0, 0.0)
        # Recalc L for initial
        L = calc_L(centers, angles, 0.0)
        return [[c[0], c[1], a] for c, a in zip(centers, angles)], outer_center, L, 0.0
        
    L = calc_L(best_centers, angles, best_Phi)
    
    inner = [[c[0], c[1], a] for c, a in zip(best_centers, angles)]
    return inner, outer_center, L, best_Phi

# EVOLVE-BLOCK-END
