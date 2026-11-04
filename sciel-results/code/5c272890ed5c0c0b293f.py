import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define centers for 11 unit hexagons
    # Using a compact cluster
    centers = [
        (0.0, 0.0),
        (np.sqrt(3), 0.0),
        (-np.sqrt(3), 0.0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (np.sqrt(3)/2, -1.5),
        (-np.sqrt(3)/2, -1.5),
        (2*np.sqrt(3), 0.0),
        (-2*np.sqrt(3), 0.0),
        (np.sqrt(3), 3.0),
        (-np.sqrt(3), 3.0)
    ]
    
    # Generate vertices for each hexagon
    # Unit hexagon vertices relative to center
    # Flat-top orientation
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    rel_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_vertices = []
    for cx, cy in centers:
        verts = rel_vertices + np.array([cx, cy])
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Find minimal bounding hexagon side length
    # We can rotate the outer hexagon.
    # The condition is that the projection of all vertices onto the 3 axes of the hexagon
    # must be within the apothem h = S * sqrt(3) / 2.
    # We want to minimize h.
    # Let the axes be at angles theta, theta+60, theta+120.
    
    def max_projection(theta):
        h_max = 0
        for k in range(3):
            alpha = theta + k * np.pi / 3
            proj = all_vertices[:, 0] * np.cos(alpha) + all_vertices[:, 1] * np.sin(alpha)
            h_max = max(h_max, np.max(np.abs(proj)))
        return h_max
    
    # Optimize theta
    # The function is periodic and smooth. We can sample it.
    thetas = np.linspace(0, np.pi / 3, 100)
    h_vals = [max_projection(t) for t in thetas]
    min_h = min(h_vals)
    
    # Side length S
    S = min_h * 2 / np.sqrt(3)
    
    # Construct inner list
    # The task asks for (x, y, angle_degrees) for inner hexagons.
    # We assumed flat-top (0 deg) for all inner hexagons.
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # Outer hexagon center is (0,0) if we centered the cluster?
    # The cluster centroid is not necessarily (0,0).
    # Let's compute the centroid of the vertices to center the outer hexagon properly.
    centroid = np.mean(all_vertices, axis=0)
    
    # Re-center vertices relative to centroid
    centered_vertices = all_vertices - centroid
    
    # Re-optimize theta with centered vertices
    def max_projection_centered(theta):
        h_max = 0
        for k in range(3):
            alpha = theta + k * np.pi / 3
            proj = centered_vertices[:, 0] * np.cos(alpha) + centered_vertices[:, 1] * np.sin(alpha)
            h_max = max(h_max, np.max(np.abs(proj)))
        return h_max

    h_vals_c = [max_projection_centered(t) for t in thetas]
    min_h_c = min(h_vals_c)
    S_c = min_h_c * 2 / np.sqrt(3)
    
    # Shift inner centers by -centroid
    inner_shifted = []
    for cx, cy in centers:
        inner_shifted.append([cx - centroid[0], cy - centroid[1], 0.0])
        
    return inner_shifted, (0.0, 0.0), S_c, 0.0
# EVOLVE-BLOCK-END
