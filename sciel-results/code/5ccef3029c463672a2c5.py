"""Honeycomb Lattice Deformation for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    """
    Place 12 disjoint regular hexagons of side length 1 inside a regular hexagon,
    minimizing the outer side length.
    
    Strategy: Use a 3x4 rectangular sub-lattice of the hexagonal tiling.
    Apply a global affine shear to the lattice basis vectors to minimize the
    bounding hexagon side length.
    """
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.pi * np.arange(6) / 3.0
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside or on boundary of hexagon."""
        # Transform to hexagon local coordinates
        angle_rad = np.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        # Rotate back
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        lx = cos_a * dx - sin_a * dy
        ly = sin_a * dx + cos_a * dy
        
        # For a regular hexagon with side s, centered at origin, flat-top orientation (0 deg rotation):
        # The hexagon is defined by |ly| <= s*sqrt(3)/2 and |lx| <= s and |lx| + 2*|ly|/sqrt(3) <= 2*s
        # Actually, let's use the standard: vertices at angles 0, 60, 120, 180, 240, 300
        # For flat-top (pointy-top? let's be consistent):
        # With angle_deg=0, vertices are at 0, 60, 120, 180, 240, 300 degrees
        # This is a "pointy-top" hexagon? No, vertex at 0 degrees means pointy right.
        # Let's use the condition: max over 6 directions of dot product <= side
        # Directions: 0, 60, 120, 180, 240, 300
        dirs = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])
        dots = np.abs(dirs @ [lx, ly])
        return np.all(dots <= side * 1.0 + 1e-9)
    
    def min_outer_side(inner_positions, outer_center, outer_angle):
        """
        Find minimum side length of regular hexagon centered at outer_center with rotation outer_angle
        that contains all inner hexagons.
        """
        cx, cy = outer_center
        oa = outer_angle
        
        # For each inner hexagon, find its vertices
        all_vertices = []
        for (ix, iy, iangle) in inner_positions:
            verts = hexagon_vertices(ix, iy, 1.0, iangle)
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        # The outer hexagon with center (cx, cy), side s, rotation oa contains a point (px, py)
        # iff for all 6 directions (oa + k*60 degrees), the projection of (px-cx, py-cy) 
        # onto that direction is <= s
        # Actually, for a regular hexagon with side s, the distance from center to each side is s*sqrt(3)/2
        # The 6 normal directions are at angles oa + 30 + k*60 degrees? Let me think again.
        
        # A regular hexagon with "flat top" has sides horizontal. Vertices at 30, 90, 150, 210, 270, 330.
        # With rotation angle oa (applied to vertices), the normals to the sides are at oa + 0, oa + 60, ...
        # Wait, if vertices are at angles theta_k = oa + k*60, then the side between vertex k and k+1
        # has normal at angle oa + k*60 + 30.
        
        # The condition for point p to be inside hexagon(center c, side s, rotation oa):
        # For each k in 0..5: dot(p-c, n_k) <= s * sqrt(3)/2
        # where n_k = (cos(oa + 30 + k*60), sin(oa + 30 + k*60))
        
        normals = np.array([[np.cos(np.radians(oa + 30 + k * 60)), np.sin(np.radians(oa + 30 + k * 60))] for k in range(6)])
        diffs = all_vertices - np.array([cx, cy])
        projections = diffs @ normals.T  # shape (n_points, 6)
        max_projections = np.max(projections, axis=0)  # max projection in each normal direction
        # Required side: max_projections[k] <= s * sqrt(3)/2 for all k
        # So s >= max_k(max_projections[k]) / (sqrt(3)/2)
        required_s = np.max(max_projections) / (np.sqrt(3) / 2.0)
        return required_s
    
    def objective(params):
        """
        params: [shear_x, shear_y, outer_cx, outer_cy, outer_angle_deg]
        Apply shear to hexagonal lattice, place 12 hexagons, compute outer side.
        """
        shear_x = params[0]
        shear_y = params[1]
        outer_cx = params[2]
        outer_cy = params[3]
        outer_angle = params[4]
        
        # Base hexagonal lattice: 
        # For a hexagonal tiling with unit hexagons (side 1), the lattice vectors are:
        # v1 = (3, 0) for flat-top? Let's use pointy-top.
        # Actually, let's define the lattice such that adjacent hexagons touch.
        # For unit hexagons, center-to-center distance for adjacent hexagons is 2 * (sqrt(3)/2) * 1 = sqrt(3) for pointy-top?
        # No. For a regular hexagon with side s, the distance from center to a vertex is s, and center to side midpoint is s*sqrt(3)/2.
        # Adjacent hexagons share an edge, so center-to-center distance is 2 * s * sqrt(3)/2 = s * sqrt(3).
        
        # Let's use a lattice with basis:
        # b1 = (sqrt(3), 0) * 1  -- horizontal spacing
        # b2 = (sqrt(3)/2, 3/2) * 1 -- diagonal
        # Wait, this depends on orientation. Let's use the standard hex grid.
        
        # For pointy-top hexagons (vertex at top):
        # Horizontal spacing: sqrt(3) * s
        # Vertical spacing: 1.5 * s
        # Offset for odd rows: sqrt(3)/2 * s
        
        # For flat-top hexagons (flat side at top):
        # Horizontal spacing: 1.5 * s
        # Vertical spacing: sqrt(3) * s
        # Offset for odd columns: sqrt(3)/2 * s
        
        # Let's use flat-top. Lattice points for a 3x4 grid (3 columns, 4 rows):
        # Column j (0..2), Row i (0..3)
        # x = j * 1.5 * s
        # y = i * sqrt(3) * s + (j % 2) * sqrt(3)/2 * s
        
        # Apply shear transformation to the lattice basis vectors.
        # Original basis: e1 = (1.5, 0), e2 = (0, sqrt(3)) with offset for parity.
        # Shear: new_e1 = (1.5 + shear_x, shear_y), new_e2 = (shear_x2, sqrt(3) + shear_y2)
        # To keep it simple, use a single shear parameter that skews the lattice.
        
        # Let's define the lattice with a shear:
        # pos(i, j) = i * v1 + j * v2 + (j % 2) * offset
        # where v1, v2 are the deformed basis vectors.
        
        # Base: v1_base = (0, sqrt(3)), v2_base = (1.5, 0)
        # With shear: v1 = (shear_x, sqrt(3) + shear_y), v2 = (1.5, 0)
        # Offset for odd j: (shear_x/2, sqrt(3)/2)
        
        positions = []
        for i in range(4):  # 4 rows
            for j in range(3):  # 3 columns
                x = i * (shear_x) + j * 1.5
                y = i * (np.sqrt(3) + shear_y) + (j % 2) * (np.sqrt(3) / 2)
                # Note: the offset for odd columns should also be sheared
                if j % 2 == 1:
                    x += shear_x / 2
                    y += shear_y / 2
                positions.append((x, y, 0.0))  # No individual rotation
        
        # Center the configuration
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx_mean = (max(xs) + min(xs)) / 2
        cy_mean = (max(ys) + min(ys)) / 2
        positions = [(x - cx_mean, y - cy_mean, a) for (x, y, a) in positions]
        
        outer_center = (outer_cx, outer_cy)
        return min_outer_side(positions, outer_center, outer_angle)
    
    # Initial guess
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'maxfev': 5000})
    
    best_params = result.x
    shear_x, shear_y, outer_cx, outer_cy, outer_angle = best_params
    
    # Reconstruct positions
    positions = []
    for i in range(4):
        for j in range(3):
            x = i * shear_x + j * 1.5
            y = i * (np.sqrt(3) + shear_y) + (j % 2) * (np.sqrt(3) / 2)
            if j % 2 == 1:
                x += shear_x / 2
                y += shear_y / 2
            positions.append((x, y, 0.0))
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    positions = [(x - cx_mean, y - cy_mean, a) for (x, y, a) in positions]
    
    # The outer hexagon should be centered at the centroid of the configuration
    outer_center = (0.0, 0.0)
    
    # Compute the side length
    side = min_outer_side(positions, outer_center, outer_angle)
    
    inner = [[pos[0], pos[1], pos[2]] for pos in positions]
    return inner, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
