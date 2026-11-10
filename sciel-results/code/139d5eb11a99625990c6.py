# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    
    Uses a hexagonal lattice representation where the 12 unit hexagons are defined by
    integer coordinates in a dual triangular grid, ensuring non-overlap by construction.
    Optimizes only the affine transformation (rotation and scale) that maps this fixed
    lattice configuration into the minimal containing regular hexagon.
    
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Define 12 hexagons on a hexagonal lattice
    # The 12 hexagons are arranged in a pattern that fits well in a hexagon
    # Using axial coordinates (q, r) for hexagonal grid
    # Each hexagon at (q, r) has center at:
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + q/2)
    
    sqrt3 = np.sqrt(3)
    
    # 12 hexagons in a compact arrangement
    # A common compact arrangement for 12 hexagons:
    # Row -2: q = -1, 0  (2 hexagons)
    # Row -1: q = -2, -1, 0, 1  (4 hexagons)
    # Row 0:  q = -1, 0, 1  (3 hexagons) -- wait, let me think more carefully
    #
    # Let's use a known good arrangement. For 12 hexagons in a hexagonal shape:
    # The hex number 12 doesn't form a perfect hexagon, so we use a slightly irregular shape.
    #
    # A good arrangement:
    # Center at (0,0)
    # Ring 1: 6 hexagons around center
    # Ring 2: 5 more hexagons to make 12 total (not a full ring of 12)
    #
    # Actually, let's just define 12 positions that are known to work well.
    # Using axial coordinates:
    
    axial_coords = [
        (0, 0),      # center
        (1, 0),      # right
        (0, 1),      # top-right
        (-1, 1),     # top-left
        (-1, 0),     # left
        (0, -1),     # bottom-left
        (1, -1),     # bottom-right
        # Second ring (partial):
        (2, 0),      # far right
        (1, 1),      # far top-right
        (0, 2),      # far top
        (-1, 2),     # far top-left
        (-2, 1),     # far left-top
    ]
    
    # Convert axial to pixel coordinates
    centers = np.zeros((12, 2))
    for i, (q, r) in enumerate(axial_coords):
        centers[i, 0] = 1.5 * q
        centers[i, 1] = sqrt3 * (r + q / 2.0)
    
    # Now we need to find the optimal rotation and scale to fit these in a minimal regular hexagon
    # The outer hexagon will be centered at the origin (we can shift later)
    
    # First, center the configuration
    center_of_mass = np.mean(centers, axis=0)
    centered = centers - center_of_mass
    
    # We'll optimize over rotation angle theta and scale factor s
    # The outer hexagon has side length S, and we want to minimize S
    # A regular hexagon with side S has inradius = S * sqrt(3)/2
    # The hexagon can be represented as the intersection of 6 half-planes
    
    def compute_required_side(theta, scale):
        """
        Given rotation angle theta and scale factor, compute the minimum side length
        of a regular hexagon (centered at origin) that contains all scaled+rotated centers.
        """
        # Rotate and scale the centers
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        transformed = (centered * scale) @ rot_mat.T
        
        # For a regular hexagon centered at origin with side length S,
        # the vertices are at distance S from center, at angles 0, 60, 120, 180, 240, 300 degrees
        # The inradius (distance from center to edge midpoint) is S * sqrt(3) / 2
        #
        # A point (x, y) is inside a regular hexagon with side S and flat-top orientation if:
        # |x| <= S * sqrt(3) / 2  (for the left/right edges)
        # |y| <= S * sqrt(3) / 2  (for the top/bottom edges) -- no, this is for a square
        #
        # For a regular hexagon with vertices at angles 0, 60, ..., 300:
        # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330 degrees
        # The distance from center to each edge is the inradius = S * sqrt(3) / 2
        #
        # For a point to be inside, its projection onto each edge normal must be <= inradius
        # Edge normals (outward) are at angles 30, 90, 150, 210, 270, 330 degrees
        
        edge_normals = []
        for i in range(6):
            angle = np.radians(30 + 60 * i)
            edge_normals.append([np.cos(angle), np.sin(angle)])
        edge_normals = np.array(edge_normals)
        
        # Project all points onto each edge normal
        # projections[i, j] = dot(transformed[i], edge_normals[j])
        projections = transformed @ edge_normals.T
        
        # For each edge normal, the maximum projection gives the required inradius for that edge
        max_projections = np.max(projections, axis=0)
        
        # The inradius must be at least the maximum of all max_projections
        required_inradius = np.max(max_projections)
        
        # Also need to ensure non-negative (points could be on the other side)
        # Actually, since the hexagon is centered at origin and we're looking at outward normals,
        # we need the max projection to be <= inradius
        # But we also need to handle the case where points are in the "negative" direction
        # For a convex shape centered at origin, we need |projection| <= inradius for all edges
        # Wait, no. For a regular hexagon centered at origin with outward normals n_i,
        # a point p is inside if dot(p, n_i) <= inradius for all i
        # But this only works if the hexagon is symmetric about the origin, which it is.
        # However, we also need dot(p, -n_i) <= inradius, which is equivalent to dot(p, n_i) >= -inradius
        # So we need |dot(p, n_i)| <= inradius for all i
        
        abs_max_projections = np.max(np.abs(projections), axis=0)
        required_inradius = np.max(abs_max_projections)
        
        # Side length S = inradius * 2 / sqrt(3)
        S = required_inradius * 2.0 / sqrt3
        
        return S
    
    # Objective function: minimize the side length
    def objective(params):
        theta, log_scale = params
        scale = np.exp(log_scale)
        return compute_required_side(theta, scale)
    
    # Try multiple initial guesses for theta
    best_S = np.inf
    best_theta = 0
    best_scale = 1.0
    
    for theta_init in np.linspace(0, np.pi / 3, 7):
        log_scale_init = np.log(1.0)
        result = minimize(objective, [theta_init, log_scale_init], method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.success:
            theta_opt, log_scale_opt = result.x
            S_opt = compute_required_side(theta_opt, np.exp(log_scale_opt))
            if S_opt < best_S:
                best_S = S_opt
                best_theta = theta_opt
                best_scale = np.exp(log_scale_opt)
    
    # Apply the optimal transformation
    cos_t = np.cos(best_theta)
    sin_t = np.sin(best_theta)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    # Transform centers: scale then rotate
    final_centers = (centered * best_scale) @ rot_mat.T
    
    # The outer hexagon is centered at origin with side length best_S
    # But we need to verify that the hexagons (with side length 1) actually fit
    # The centers are placed, but the hexagons have extent. We need to check if the
    # actual hexagon vertices fit inside the outer hexagon.
    
    # For a unit hexagon centered at (cx, cy) with angle 0, the vertices are at:
    # (cx + cos(60*i*pi/180), cy + sin(60*i*pi/180)) for i=0..5
    # The maximum extent from center is 1 (the circumradius)
    
    # So we need the outer hexagon to contain all vertices of all inner hexagons
    # Let's recompute the required side length considering the full hexagon extent
    
    def compute_required_side_with_extent(theta, scale):
        """
        Compute the minimum side length of a regular hexagon that contains
        all vertices of the 12 unit hexagons after transformation.
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # For each center, generate the 6 vertices of the unit hexagon
        # Unit hexagon with angle 0 has vertices at angles 0, 60, 120, 180, 240, 300
        vertex_angles = np.radians([0, 60, 120, 180, 240, 300])
        vertex_offsets = np.column_stack([np.cos(vertex_angles), np.sin(vertex_angles)])
        
        all_vertices = []
        for i in range(12):
            # Scale and rotate the center
            scaled_center = centered[i] * scale
            rotated_center = scaled_center @ rot_mat.T
            
            # The hexagon orientation: we keep all inner hexagons at angle 0
            # So vertices are at rotated_center + vertex_offsets
            verts = rotated_center + vertex_offsets
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        # Edge normals for the outer hexagon
        edge_normals = []
        for i in range(6):
            angle = np.radians(30 + 60 * i)
            edge_normals.append([np.cos(angle), np.sin(angle)])
        edge_normals = np.array(edge_normals)
        
        projections = all_vertices @ edge_normals.T
        abs_max_projections = np.max(np.abs(projections), axis=0)
        required_inradius = np.max(abs_max_projections)
        
        S = required_inradius * 2.0 / sqrt3
        return S
    
    # Re-optimize considering the full hexagon extent
    def objective_with_extent(params):
        theta, log_scale = params
        scale = np.exp(log_scale)
        return compute_required_side_with_extent(theta, scale)
    
    best_S2 = np.inf
    best_theta2 = 0
    best_scale2 = 1.0
    
    for theta_init in np.linspace(0, np.pi / 3, 13):
        log_scale_init = np.log(1.0)
        result = minimize(objective_with_extent, [theta_init, log_scale_init], method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.success:
            theta_opt, log_scale_opt = result.x
            S_opt = compute_required_side_with_extent(theta_opt, np.exp(log_scale_opt))
            if S_opt < best_S2:
                best_S2 = S_opt
                best_theta2 = theta_opt
                best_scale2 = np.exp(log_scale_opt)
    
    # Use the better solution
    if best_S2 < best_S:
        best_theta = best_theta2
        best_scale = best_scale2
        outer_side = best_S2
    else:
        outer_side = best_S
    
    # Apply final transformation
    cos_t = np.cos(best_theta)
    sin_t = np.sin(best_theta)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    final_centers = (centered * best_scale) @ rot_mat.T
    
    # Create inner hexagon data
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = 0.0  # all hexagons at angle 0
    
    # Outer hexagon centered at origin
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_side


# EVOLVE-BLOCK-END
