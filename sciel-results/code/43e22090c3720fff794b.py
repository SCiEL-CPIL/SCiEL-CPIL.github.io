"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a trust-region reflective optimization with penalty for overlap.
    """
    # Parameters
    n_inner = N
    # Each inner hexagon: (x, y, angle) -> 3 params per hexagon
    # Outer hexagon: (cx, cy, side, angle) -> 4 params
    total_params = 3 * n_inner + 4
    
    def hexagon_vertices(cx, cy, side, angle_rad):
        """Return vertices of a regular hexagon centered at (cx, cy) with given side and angle."""
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    def hexagon_corners(cx, cy, side, angle_deg):
        """Return corners of a regular hexagon centered at (cx, cy) with side length and angle in degrees."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    def polygon_distance(corners1, corners2):
        """
        Compute minimum distance between two convex polygons.
        If they overlap, return a negative value indicating penetration depth (approximate).
        """
        # For two convex polygons, we can use the separating axis theorem approach
        # to determine if they overlap and compute a penalty.
        # We'll compute the minimum distance from each vertex of one polygon to the edges of the other.
        
        def point_to_segment_distance(p, a, b):
            """Distance from point p to segment ab."""
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        
        def polygon_overlap_penalty(c1, c2):
            """
            Return a penalty if polygons overlap.
            Use SAT-like check: if any separating axis exists, no overlap.
            If no separating axis, compute approximate penetration.
            """
            # Collect all edges from both polygons
            edges = []
            for i in range(len(c1)):
                edges.append((c1[i], c1[(i+1) % len(c1)]))
            for i in range(len(c2)):
                edges.append((c2[i], c2[(i+1) % len(c2)]))
            
            # Check separating axes
            for a, b in edges:
                normal = np.array([-(b[1] - a[1]), b[0] - a[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                
                proj1 = np.dot(c1, normal)
                proj2 = np.dot(c2, normal)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                
                if max1 < min2 or max2 < min1:
                    # Separating axis found, no overlap
                    gap = max(min2 - max1, min1 - max2)
                    return -gap  # negative means no overlap, gap is distance
                    
            # No separating axis, polygons overlap
            # Approximate penetration depth
            # Find minimum distance from vertices of one to edges of other
            min_pen = float('inf')
            for p in c1:
                for i in range(len(c2)):
                    a, b = c2[i], c2[(i+1) % len(c2)]
                    # Check if p is inside the half-plane defined by edge
                    ab = b - a
                    normal = np.array([ab[1], -ab[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12:
                        continue
                    normal = normal / norm
                    # Point should be on the same side as the polygon interior
                    # For a convex polygon, interior is to the left of each directed edge
                    # So check if p is to the right of edge (outside)
                    dist = np.dot(p - a, normal)
                    if dist > 0:
                        # p is outside this edge's half-plane
                        min_pen = min(min_pen, dist)
            
            for p in c2:
                for i in range(len(c1)):
                    a, b = c1[i], c1[(i+1) % len(c1)]
                    ab = b - a
                    normal = np.array([ab[1], -ab[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12:
                        continue
                    normal = normal / norm
                    dist = np.dot(p - a, normal)
                    if dist > 0:
                        min_pen = min(min_pen, dist)
            
            if min_pen == float('inf'):
                min_pen = 0.0
            
            return min_pen  # positive means overlap
        
        return polygon_overlap_penalty(corners1, corners2)
    
    def point_in_hexagon(p, cx, cy, side, angle_deg):
        """Check if point p is inside hexagon centered at (cx, cy) with given side and angle."""
        corners = hexagon_corners(cx, cy, side, angle_deg)
        # Use winding number or ray casting
        x, y = p
        inside = False
        n = len(corners)
        for i in range(n):
            x1, y1 = corners[i]
            x2, y2 = corners[(i+1) % n]
            if ((y1 > y) != (y2 > y)):
                x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < x_intersect:
                    inside = not inside
        return inside
    
    def objective(params):
        """
        Objective: minimize outer side length + penalty for overlaps.
        params: [x1, y1, a1, x2, y2, a2, ..., xn, yn, an, cx, cy, side, outer_angle]
        """
        inner_params = params[:3 * n_inner]
        outer_params = params[3 * n_inner:]
        
        cx, cy, side, outer_angle = outer_params
        
        # Penalty for negative side length
        if side < 0.5:
            return 1000.0 + (0.5 - side) ** 2
        
        # Compute inner hexagon corners
        inner_corners = []
        for i in range(n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            corners = hexagon_corners(x, y, 1.0, angle)
            inner_corners.append(corners)
        
        outer_corners = hexagon_corners(cx, cy, side, outer_angle)
        
        total_penalty = 0.0
        
        # Check inner-inner overlaps
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                pen = polygon_distance(inner_corners[i], inner_corners[j])
                if pen > 0:
                    total_penalty += pen ** 2
        
        # Check inner-outer containment
        for i in range(n_inner):
            for p in inner_corners[i]:
                if not point_in_hexagon(p, cx, cy, side, outer_angle):
                    # Compute distance to outer boundary
                    # Find minimum distance from p to outer hexagon edges
                    min_dist = float('inf')
                    for k in range(6):
                        a = outer_corners[k]
                        b = outer_corners[(k+1) % 6]
                        min_dist = min(min_dist, point_to_segment_distance(p, a, b))
                    total_penalty += min_dist ** 2
        
        # Objective: side length + penalty
        return side + 10.0 * total_penalty
    
    # Initial guess
    # Place inner hexagons in a grid pattern
    initial_params = np.zeros(total_params)
    
    # Spread inner hexagons in a hexagonal grid
    spacing = 2.2  # Slightly more than 2 to avoid initial overlap
    for i in range(n_inner):
        # Hexagonal grid positions
        row = i // 4
        col = i % 4
        x = (col - 1.5) * spacing
        y = row * spacing * np.sqrt(3) / 2
        if row % 2 == 1:
            x += spacing / 2
        initial_params[3*i] = x
        initial_params[3*i+1] = y
        initial_params[3*i+2] = 0.0
    
    # Outer hexagon initial guess
    initial_params[3*n_inner] = 0.0    # cx
    initial_params[3*n_inner+1] = 0.0  # cy
    initial_params[3*n_inner+2] = 15.0 # side
    initial_params[3*n_inner+3] = 0.0  # angle
    
    # Optimize using trust-region reflective
    result = minimize(objective, initial_params, method='trust-constr',
                      options={'maxiter': 200, 'gtol': 1e-5, 'xtol': 1e-5})
    
    # Extract solution
    if result.success or result.fun < 100:
        final_params = result.x
    else:
        final_params = initial_params
    
    inner = []
    for i in range(n_inner):
        x = final_params[3*i]
        y = final_params[3*i+1]
        angle = final_params[3*i+2]
        inner.append([float(x), float(y), float(angle)])
    
    cx = float(final_params[3*n_inner])
    cy = float(final_params[3*n_inner+1])
    side = float(final_params[3*n_inner+2])
    outer_angle = float(final_params[3*n_inner+3])
    
    # Ensure side is positive
    if side < 1.0:
        side = 1.0
    
    return inner, (cx, cy), side, outer_angle
# EVOLVE-BLOCK-END
