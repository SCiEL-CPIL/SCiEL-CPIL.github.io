import numpy as np

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def min_enclosing_hexagon(points):
    """Find the minimal side length regular hexagon (any rotation) enclosing all points.
    Returns (center_x, center_y, side, angle_deg)."""
    # We'll try different rotations and for each, find the minimal side
    best_side = float('inf')
    best_params = None
    
    # Try various rotation angles
    for angle_deg in np.linspace(0, 60, 121):  # 0 to 60 degrees, 121 steps
        angle = np.radians(angle_deg)
        
        # For a given hexagon orientation, the constraint for a point (px, py) to be inside
        # a hexagon centered at (cx, cy) with side s and rotation angle is:
        # The hexagon has 6 sides. For each side direction, the projection of (px-cx, py-cy)
        # onto the normal of that side must be <= s * sqrt(3)/2 (the inradius)
        
        # Actually, let's think of it differently. A regular hexagon with side s and rotation angle
        # can be described by 6 half-planes. The normals to the sides are at angles:
        # angle + pi/6, angle + pi/6 + pi/3, ..., angle + pi/6 + 5*pi/3
        
        # For the hexagon to contain all points, for each of the 6 side normals n_i:
        # max over points of dot(point - center, n_i) <= s * sqrt(3)/2
        # and min over points of dot(point - center, n_i) >= -s * sqrt(3)/2
        
        # So for a given center and orientation, the required side is:
        # s = max over i of (max_j dot(p_j - c, n_i) - min_j dot(p_j - c, n_i)) / (2 * sqrt(3)/2)
        #   = max over i of (range_i) / sqrt(3)
        
        # To minimize s, we need to choose center c optimally.
        # For a fixed orientation, the optimal center is the midpoint of the bounding box
        # in each of the 6 directions, but they must be consistent.
        
        # Actually, for a fixed orientation, the minimal enclosing hexagon with that orientation
        # has center at the centroid of the "dual" problem. Let me just compute it numerically.
        
        # For a given orientation, the 6 normal directions are:
        normals = []
        for i in range(6):
            theta = angle + np.pi / 6 + i * np.pi / 3
            normals.append((np.cos(theta), np.sin(theta)))
        
        # For the hexagon to contain all points with center c and side s:
        # For each normal n_i: max_j dot(p_j - c, n_i) <= s * sqrt(3)/2
        #                    min_j dot(p_j - c, n_i) >= -s * sqrt(3)/2
        # This means: max_j dot(p_j, n_i) - dot(c, n_i) <= s * sqrt(3)/2
        #            dot(c, n_i) - min_j dot(p_j, n_i) <= s * sqrt(3)/2
        # So: max_j dot(p_j, n_i) - min_j dot(p_j, n_i) <= s * sqrt(3)
        # Therefore: s >= (max_j dot(p_j, n_i) - min_j dot(p_j, n_i)) / sqrt(3)
        
        # This is independent of c! So for a fixed orientation, the minimal side is:
        # s = max over i of (range of projections onto n_i) / sqrt(3)
        
        # And the center can be anywhere in the intersection of the feasible regions,
        # but the side length is determined solely by the orientation and the points.
        
        # Wait, that's not right. The center matters. Let me redo this.
        
        # For a hexagon centered at c with side s and orientation angle:
        # A point p is inside if for all 6 side normals n_i:
        #   dot(p - c, n_i) <= s * sqrt(3)/2
        #   dot(c - p, n_i) <= s * sqrt(3)/2  (i.e., dot(p - c, -n_i) <= s*sqrt(3)/2)
        
        # Actually, a regular hexagon with side s has 6 sides. The distance from center to each side is s*sqrt(3)/2.
        # The outward normals to the sides are at angles: angle + pi/6 + k*pi/3 for k=0..5
        
        # For point p to be inside: for each outward normal n_k:
        #   dot(p - c, n_k) <= s * sqrt(3)/2
        
        # So for all points p_j and all k:
        #   dot(p_j - c, n_k) <= s * sqrt(3)/2
        #   => dot(p_j, n_k) - dot(c, n_k) <= s * sqrt(3)/2
        #   => dot(c, n_k) >= dot(p_j, n_k) - s * sqrt(3)/2
        
        # And also the opposite constraint isn't needed because the hexagon is bounded by these 6 half-planes.
        
        # Hmm, actually for a convex hexagon, the 6 half-planes fully define it. So:
        # For each k: dot(p_j - c, n_k) <= s*sqrt(3)/2 for all j
        # => max_j dot(p_j, n_k) - dot(c, n_k) <= s*sqrt(3)/2
        # => dot(c, n_k) >= max_j dot(p_j, n_k) - s*sqrt(3)/2
        
        # Similarly, considering the inward normals (or equivalently, the other 6 constraints from the opposite sides):
        # Actually, the 6 normals n_k already cover all 6 sides. The hexagon is the intersection of:
        # {p : dot(p - c, n_k) <= s*sqrt(3)/2} for k=0..5
        
        # So we need: for all k, j: dot(p_j, n_k) - dot(c, n_k) <= s*sqrt(3)/2
        # => dot(c, n_k) >= max_j dot(p_j, n_k) - s*sqrt(3)/2
        
        # To find the minimal s, we need to find c and s such that there exists c with:
        # dot(c, n_k) >= max_j dot(p_j, n_k) - s*sqrt(3)/2 for all k
        
        # This is a linear feasibility problem in c and s. For a given s, we check if there exists c.
        
        # Alternatively, we can observe that for a fixed orientation, the minimal enclosing hexagon
        # has its center at the centroid of the points (or some optimal point), and we can binary search on s.
        
        # Let me just do a simple approach: for a given orientation, try centers on a grid and compute the required s.
        
        # Actually, let me think about this more simply. For a fixed orientation, the problem reduces to:
        # Find c that minimizes max_k (max_j dot(p_j - c, n_k)) / (sqrt(3)/2)
        # = 2/sqrt(3) * min_c max_k max_j dot(p_j - c, n_k)
        
        # The inner term max_j dot(p_j - c, n_k) = max_j dot(p_j, n_k) - dot(c, n_k)
        
        # So we want to minimize max_k (M_k - dot(c, n_k)) where M_k = max_j dot(p_j, n_k)
        
        # This is equivalent to maximizing min_k dot(c, n_k) - M_k... no.
        
        # Let me just do a numerical optimization. For a given orientation, I'll use scipy to find the optimal center.
        
        from scipy.optimize import minimize
        
        points_arr = np.array(points)
        
        def objective(c):
            cx, cy = c
            max_val = 0
            for nx, ny in normals:
                proj = (points_arr[:, 0] - cx) * nx + (points_arr[:, 1] - cy) * ny
                max_val = max(max_val, proj.max())
            return max_val
        
        # Initial guess: centroid
        c0 = [points_arr[:, 0].mean(), points_arr[:, 1].mean()]
        
        result = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 1000})
        c_opt = result.x
        max_proj = objective(c_opt)
        side = max_proj / (np.sqrt(3) / 2)
        
        if side < best_side:
            best_side = side
            best_params = (c_opt[0], c_opt[1], side, angle_deg)
    
    return best_params

def construct():
    # Generate candidate arrangements of 12 unit hexagons
    # Try hexagonal grid patterns
    
    candidates = []
    
    # Pattern 1: 3x3 grid minus corners (7 hexagons) + 5 more
    # Let's try various hexagonal grid arrangements
    
    # Hexagonal grid coordinates: use axial coordinates
    # Generate a 4x4 hex grid and pick 12
    
    # Actually, let me try a simple approach: place hexagons in a hexagonal lattice
    # and try different subsets or full small hexagons
    
    # For a hexagonal arrangement with "radius" r (number of hexagons from center to edge):
    # Total hexagons = 3r^2 + 3r + 1... no, it's 3r(r+1)+1
    
    # r=0: 1 hexagon
    # r=1: 7 hexagons  
    # r=2: 19 hexagons
    
    # We need 12, so let's try r=1 (7 hexagons) plus 5 more, or a rectangular-ish arrangement.
    
    # Let me try: a 3x4 rectangular arrangement in hex grid coordinates
    
    # Hex grid positions (axial coords converted to cartesian):
    # For a hex with side 1, the center-to-center distance is sqrt(3)
    # Axial (q, r) to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        return (x, y)
    
    # Try various arrangements:
    
    # Arrangement A: 4 columns of 3 each (12 hexagons)
    # Axial coords for a 4x3 block
    arrangement_A = []
    for col in range(4):
        for row in range(3):
            arrangement_A.append((col, row))
    
    # Arrangement B: hexagonal shape with 12 hexagons
    # Center + 6 around + 5 more
    arrangement_B = []
    # Center
    arrangement_B.append((0, 0))
    # 6 neighbors
    for dq, dr in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
        arrangement_B.append((dq, dr))
    # 5 more: extend in some directions
    arrangement_B.extend([(2,0), (0,2), (-2,0), (2,-2), (-2,2)])
    
    # Arrangement C: 3x4 in a different orientation
    arrangement_C = []
    for col in range(3):
        for row in range(4):
            arrangement_C.append((col, row))
    
    for arr in [arrangement_A, arrangement_B, arrangement_C]:
        if len(arr) != N:
            continue
        
        # Convert to cartesian
        points = [axial_to_cartesian(q, r) for q, r in arr]
        
        # Compute minimal enclosing hexagon
        params = min_enclosing_hexagon(points)
        if params is not None:
            cx, cy, side, angle_deg = params
            # Shift so that the outer hexagon center is at (cx, cy)
            # The inner hexagons are at the computed points
            inner = [[p[0], p[1], 0.0] for p in points]
            candidates.append((side, inner, (cx, cy), angle_deg))
    
    # Pick the best candidate
    candidates.sort(key=lambda x: x[0])
    
    if candidates:
        side, inner, center, angle_deg = candidates[0]
        return inner, center, side, angle_deg
    else:
        # Fallback: linear arrangement
        inner = [[i * np.sqrt(3), 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 20.0, 0.0

