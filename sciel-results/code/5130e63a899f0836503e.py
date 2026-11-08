"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Approach:
    - Use a lattice-like arrangement of hexagons that tiles efficiently.
    - The support function / projection method: for a given outer hexagon (center, side, angle),
      the minimal side length enclosing all inner hexagons is determined by the maximum
      projection of all inner hexagon vertices onto the 6 normal directions of the outer hexagon.
    - We optimize the positions and orientations of the inner hexagons to minimize this maximum
      projection, which directly gives the outer side length.
    
    Key insight: For a regular hexagon of side s centered at origin with orientation theta,
    the 6 outward normals are at angles theta + k*pi/3 for k=0..5. A point p is inside if
    dot(p - center, n_k) <= s * sqrt(3)/2 for all k (since the apothem of a regular hexagon
    with side s is s*sqrt(3)/2).
    
    So the outer side length needed is: max over all vertices v of all inner hexagons,
    over all 6 normals n_k, of dot(v - center, n_k) / (sqrt(3)/2).
    
    We'll set up a configuration where the 12 hexagons are arranged in a compact pattern,
    then optimize their positions and rotations to minimize the enclosing hexagon side.
    """
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon with given center, side, and rotation."""
        angle_rad = np.radians(angle_deg)
        # Vertices of a regular hexagon with side length 'side' centered at (cx, cy)
        # The hexagon has flat top/bottom when angle=0 (vertices at 0, 60, 120, 180, 240, 300 degrees)
        # Actually, let's use the convention where angle=0 means one vertex points right
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)
    
    def enclosing_hexagon_side(inner_hexes, outer_center):
        """
        Compute the minimum side length of a regular hexagon centered at outer_center
        (with optimal rotation) that encloses all inner hexagons.
        
        Actually, we need to also optimize the outer hexagon's rotation. For a given
        set of points, the minimal enclosing regular hexagon's side length for a given
        rotation theta is:
            s(theta) = max_k max_p dot(p - center, n_k(theta)) / (sqrt(3)/2)
        where n_k(theta) are the 6 normals at angles theta + k*pi/3.
        
        We minimize over theta.
        """
        # Collect all vertices from all inner hexagons
        all_vertices = []
        for cx, cy, ang in inner_hexes:
            verts = hexagon_vertices(cx, cy, 1.0, ang)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # Translate to center
        ocx, ocy = outer_center
        all_vertices = all_vertices - np.array([ocx, ocy])
        
        def side_for_rotation(theta):
            """Compute required side length for given outer hexagon rotation."""
            max_proj = 0.0
            for k in range(6):
                angle = theta + k * np.pi / 3
                n = np.array([np.cos(angle), np.sin(angle)])
                proj = np.max(all_vertices @ n)
                # The apothem is s * sqrt(3)/2, so s = proj / (sqrt(3)/2) = 2*proj/sqrt(3)
                s_needed = proj / (np.sqrt(3) / 2)
                if s_needed > max_proj:
                    max_proj = s_needed
            return max_proj
        
        # Optimize over theta
        # The function is periodic with period pi/3, so we can search in [0, pi/3)
        result = minimize(lambda t: side_for_rotation(t), [0.0], bounds=[(0.0, np.pi/3)], method='L-BFGS-B')
        best_theta = result.x[0]
        best_side = side_for_rotation(best_theta)
        
        return best_side, best_theta
    
    def total_cost(params):
        """
        Parameters: for each of N hexagons, (x, y, angle_deg)
        Total: 3*N parameters.
        """
        inner_hexes = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            ang = params[3*i + 2]
            inner_hexes.append((x, y, ang))
        
        # For simplicity, keep outer center at origin
        outer_center = (0.0, 0.0)
        side, theta = enclosing_hexagon_side(inner_hexes, outer_center)
        return side
    
    # Initial configuration: arrange 12 hexagons in a compact pattern
    # A hexagonal arrangement: rows of hexagons
    # Let's try a 3-4-3-2 arrangement or similar compact layout
    # Actually, for 12 hexagons, a good initial guess is a roughly hexagonal cluster
    
    # Hexagonal packing: distance between centers of adjacent unit hexagons is sqrt(3)
    # (since they share an edge, the center-to-center distance is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3))
    d = np.sqrt(3)
    
    # Let's place them in a pattern that forms a larger hexagonal shape
    # Row-based layout with offset rows
    # For 12 hexagons: maybe 4+4+4 or 3+4+3+2 etc.
    # Let's try a 2-3-4-3 arrangement (like a diamond)
    
    # Better: use a known efficient packing
    # Place in a grid with hexagonal offsets
    positions = []
    # Let's do 3 rows of 4, with offsets
    # Row 0: 4 hexagons
    for j in range(4):
        positions.append((j * d, 0.0, 0.0))
    # Row 1: 4 hexagons, offset by d/2 in x and d*sqrt(3)/2 in y
    for j in range(4):
        positions.append((j * d + d/2, d * np.sqrt(3)/2, 0.0))
    # Row 2: 4 hexagons
    for j in range(4):
        positions.append((j * d, d * np.sqrt(3), 0.0))
    
    # Center the configuration
    cx_mean = np.mean([p[0] for p in positions])
    cy_mean = np.mean([p[1] for p in positions])
    positions = [(p[0] - cx_mean, p[1] - cy_mean, p[2]) for p in positions]
    
    # Initial parameters
    x0 = []
    for p in positions:
        x0.extend([p[0], p[1], p[2]])
    
    # Optimize
    result = minimize(total_cost, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    # Extract optimized positions
    inner = []
    for i in range(N):
        x = result.x[3*i]
        y = result.x[3*i + 1]
        ang = result.x[3*i + 2]
        inner.append([x, y, ang])
    
    # Compute the final enclosing hexagon
    outer_center = (0.0, 0.0)
    side, theta = enclosing_hexagon_side(inner, outer_center)
    angle_deg = np.degrees(theta)
    
    return inner, outer_center, side, angle_deg
# EVOLVE-BLOCK-END
