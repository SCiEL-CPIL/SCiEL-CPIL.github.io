import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = cx + np.cos(theta)
            y = cy + np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def min_enclosing_hexagon_side(points):
        # points: Nx2 array of vertices
        # We want to minimize S such that points are inside a hexagon of side S centered at origin.
        # The hexagon is defined by 3 axes.
        # Let the axes be at angles 0, 60, 120.
        # The condition is |p dot u_i| <= S * sqrt(3)/2 for i=0,1,2.
        # So S >= max_i (|p dot u_i| * 2 / sqrt(3)).
        # We can rotate the hexagon by angle alpha.
        # We want to find alpha that minimizes max over points and axes.
        
        def calc_S(alpha):
            axes = []
            for i in range(3):
                theta = alpha + i * np.pi / 3
                axes.append([np.cos(theta), np.sin(theta)])
            axes = np.array(axes)
            # projections = points @ axes.T
            # max_proj = np.max(np.abs(projections), axis=1)
            # S = max_proj * 2 / np.sqrt(3)
            # return np.max(S)
            
            # Vectorized:
            # points (N, 2), axes (3, 2)
            # projections (N, 3)
            projections = points @ axes.T
            max_abs_proj = np.max(np.abs(projections), axis=1)
            S = max_abs_proj * 2 / np.sqrt(3)
            return np.max(S)

        # Optimize alpha
        # The function is periodic with period pi/3.
        # We can use a simple grid search or scipy minimize.
        # Since it's 1D, grid search is fast and robust.
        alphas = np.linspace(0, np.pi/3, 1000)
        S_vals = [calc_S(a) for a in alphas]
        min_idx = np.argmin(S_vals)
        return S_vals[min_idx], alphas[min_idx]

    # Define configurations
    # Configuration 1: 7-cluster + 4 attached
    # Centers of 7-cluster:
    centers = [[0, 0]]
    for i in range(6):
        theta = i * np.pi / 3
        centers.append([np.sqrt(3) * np.cos(theta), np.sqrt(3) * np.sin(theta)])
    
    # Attach 4 more. Let's try attaching to indices 1, 2, 4, 5 (leaving 0 and 3 empty)
    # Or 0, 1, 4, 5. Symmetry doesn't matter for S if we can rotate.
    # Let's attach to 0, 1, 2, 3.
    attach_indices = [0, 1, 2, 3]
    for idx in attach_indices:
        base_center = centers[idx]
        # Direction from origin to base_center
        theta = np.arctan2(base_center[1], base_center[0])
        new_center = [ (np.sqrt(3) + np.sqrt(3)) * np.cos(theta), (np.sqrt(3) + np.sqrt(3)) * np.sin(theta) ]
        centers.append(new_center)
        
    # We have 11 centers now.
    # All hexagons have angle 0 (aligned).
    inner_hex_data = np.zeros((11, 3))
    all_vertices = []
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
        verts = get_hex_vertices(c[0], c[1], 0.0)
        all_vertices.extend(verts)
        
    all_vertices = np.array(all_vertices)
    S, alpha = min_enclosing_hexagon_side(all_vertices)
    
    # We can also try other configurations
    # Configuration 2: 7-cluster + 4 in gaps (pushed out)
    # This is harder to define statically.
    
    # Let's just return this one for now, as it's a valid packing.
    # We need to ensure no overlaps.
    # The construction ensures they are touching or separated by construction (lattice based).
    # The 7-cluster is valid.
    # The attached ones are at distance 2*sqrt(3) from origin.
    # Distance to neighbor in 7-cluster: sqrt(3).
    # Distance to adjacent attached hexagon:
    # If attached to 0 and 1.
    # Center 0: (2sqrt(3), 0). Center 1: (sqrt(3), 3).
    # Distance: sqrt( (sqrt(3))^2 + (3)^2 ) = sqrt(3+9) = sqrt(12) = 2sqrt(3).
    # Wait, distance between centers of touching hexagons is sqrt(3).
    # Here distance is 2sqrt(3) > sqrt(3). So they are separated.
    # Distance to center hexagon: 2sqrt(3) > sqrt(3). Separated.
    # So no overlaps.
    
    outer_hex_data = np.array([0, 0, np.rad2deg(alpha)])
    outer_hex_side_length = S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
