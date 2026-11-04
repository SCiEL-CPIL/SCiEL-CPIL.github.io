"""Program for packing 11 unit regular hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We need to pack 11 unit regular hexagons inside a regular hexagon.
    # The seed places them in a line: positions at x = 4*k for k in range(-5, 6)
    # That gives 11 hexagons in a row, each separated by 4 (which is 2*side*2 = distance for non-overlap along x when aligned).
    # Wait, for unit hexagons (side=1), the width across flats is sqrt(3), and across corners is 2.
    # If we place them in a row along x-axis with angle 0, the distance between centers for non-overlap is 2 (across corners).
    # But the seed uses 4*k, which is very sparse. Let's think about a denser packing.
    
    # Actually, let's reconsider the geometry. A unit regular hexagon has:
    # - Circumradius (center to vertex) = 1
    # - Inradius (center to edge midpoint) = sqrt(3)/2
    # - Width across corners = 2
    # - Width across flats = sqrt(3)
    
    # For two unit hexagons to be disjoint, the distance between centers must be at least 2 if they are aligned the same way (vertex-to-vertex contact).
    
    # Let's try a more compact arrangement. A common efficient packing is to arrange hexagons in a honeycomb pattern.
    # For 11 hexagons, we can try a 3-4-3 or 4-3-4 arrangement, or a linear chain with some offset.
    
    # Let's first try the simple linear arrangement but with optimal spacing:
    # Place hexagons at x = 2*k for k in range(-5, 6), y=0, angle=0.
    # This gives 11 hexagons in a row, touching each other.
    # The total span is from x=-10 to x=10, so width 20 in x-direction.
    # The height is just the height of one hexagon = sqrt(3) ≈ 1.732.
    
    # Now we need to find the minimal enclosing regular hexagon.
    # The enclosing hexagon has center (cx, cy), side length s, and rotation angle theta.
    # A point (x, y) is inside the enclosing hexagon if it satisfies certain inequalities.
    
    # For a regular hexagon with center (0,0), side length s, and rotation 0:
    # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees, at distance s from center.
    # Wait, for a regular hexagon with side length s, the circumradius is s.
    # So vertices are at (s*cos(k*60°), s*sin(k*60°)) for k=0..5.
    # The hexagon is the intersection of 6 half-planes.
    
    # Actually, let's define the enclosing hexagon more carefully.
    # A regular hexagon with center (cx, cy), side length s, and rotation angle phi has:
    # - 6 vertices at (cx + s*cos(phi + k*pi/3), cy + s*sin(phi + k*pi/3)) for k=0..5
    # - 6 edges, each edge is a line segment
    # - The hexagon is the intersection of 6 half-planes, one for each edge
    
    # For each edge, the normal vector points outward. The distance from center to each edge is the inradius = s*sqrt(3)/2.
    # So the hexagon can be described as: for each direction u_k (unit normal to edge k), the dot product (p - center) · u_k <= s*sqrt(3)/2.
    
    # The 6 outward normals for a hexagon with rotation phi are at angles phi + k*pi/3 + pi/6 for k=0..5? Let me think.
    # If vertices are at angles phi + k*pi/3, then edges are between consecutive vertices.
    # The midpoint of the edge between vertex k and vertex k+1 is at angle phi + (k+0.5)*pi/3, at distance s*sqrt(3)/2 from center.
    # So the outward normal for edge k is at angle phi + (k+0.5)*pi/3.
    
    # So the condition for point p to be inside is:
    # For k=0..5: (p - center) · (cos(phi + (k+0.5)*pi/3), sin(phi + (k+0.5)*pi/3)) <= s*sqrt(3)/2
    
    # We want to minimize s such that all 11 hexagon points (all vertices of all inner hexagons) are inside.
    
    # This is a linear programming problem in s for fixed phi and center, but we also need to optimize over phi and center.
    
    # Let's start with the linear arrangement and optimize over phi, cx, cy.
    
    # First, get all vertices of the 11 inner hexagons.
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get the 6 vertices of a regular hexagon with center (cx,cy), rotation angle_deg, side length."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append((x, y))
        return vertices
    
    # Linear arrangement: 11 hexagons at (2*k, 0, 0) for k in range(-5, 6)
    inner_positions = [(2.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
    
    # Get all vertices
    all_vertices = []
    for (hx, hy, ha) in inner_positions:
        verts = get_hex_vertices(hx, hy, ha)
        all_vertices.extend(verts)
    all_vertices = np.array(all_vertices)
    
    # Now optimize over phi (rotation of enclosing hexagon), cx, cy (center), and s (side length).
    # For fixed phi, cx, cy, the minimal s is determined by the maximum over all vertices of:
    # s_k = ((vx - cx) * cos(nk) + (vy - cy) * sin(nk)) / (sqrt(3)/2)
    # where nk = phi + (k+0.5)*pi/3 for k=0..5, and we take the max over k and over all vertices.
    # Actually, s = max over all vertices p and all k of ((p - center) · u_k) / (sqrt(3)/2)
    # where u_k are the 6 outward normals.
    
    # So s(phi, cx, cy) = max_{p in vertices, k in 0..5} ((p - c) · u_k(phi)) / (sqrt(3)/2)
    # We want to minimize this over phi, cx, cy.
    
    # Let's use scipy.optimize to minimize over phi, cx, cy.
    
    def objective(params):
        phi, cx, cy = params
        # 6 outward normal directions
        normals = []
        for k in range(6):
            angle = phi + (k + 0.5) * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)  # shape (6, 2)
        
        # Compute (p - center) · u_k for all p and all k
        # all_vertices shape: (M, 2), normals shape: (6, 2)
        # We want max over all M*6 values of (p - c) · u_k / (sqrt(3)/2)
        diff = all_vertices - np.array([cx, cy])  # (M, 2)
        # For each normal, compute dot products with all vertices
        # dot products: (M, 6)
        dots = diff @ normals.T  # (M, 6)
        max_dot = np.max(dots)
        s = max_dot / (np.sqrt(3) / 2)
        return s
    
    # Initial guess
    phi0 = 0.0
    cx0 = 0.0
    cy0 = 0.0
    
    # Use Nelder-Mead or similar
    result = minimize_scalar(lambda phi: objective([phi, 0.0, 0.0]), bounds=(0, np.pi/3), method='bounded')
    phi_best = result.x
    
    # Now optimize cx, cy for fixed phi
    def obj_xy(cx, cy):
        return objective([phi_best, cx, cy])
    
    from scipy.optimize import minimize
    res_xy = minimize(obj_xy, [0.0, 0.0], method='Nelder-Mead')
    cx_best, cy_best = res_xy.x
    
    s_best = objective([phi_best, cx_best, cy_best])
    
    # Return the configuration
    inner = [[pos[0], pos[1], pos[2]] for pos in inner_positions]
    return inner, (cx_best, cy_best), s_best, np.degrees(phi_best)
# EVOLVE-BLOCK-END
