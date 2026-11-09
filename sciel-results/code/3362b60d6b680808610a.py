<<<<<<< SEARCH
def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    inner = np.array(inner, dtype=float)
    outer_center = np.array(outer_center, dtype=float)
    
    # Shift inner centers by minus outer center
    shifted_centers = inner[:, :2] - outer_center
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a regular hexagon centered at origin with angle a and side s,
    # the constraint is that for each edge, the cross product test must pass.
    # We want to find minimum s such that all vertices are inside.
    
    # The hexagon with center at origin, angle a, side s has edges defined by:
    # For each edge k, the inward normal points from the edge toward center.
    # A point p is inside if for all k: dot(normal_k, p) <= s * apothem_factor
    
    # For a regular hexagon with side s, the apothem (distance from center to edge) is s * sqrt(3)/2
    # The distance from center to vertex is s
    
    # Let's use the constraint: for each outer edge (p1, p2), cross(p2-p1, v-p1) >= -1e-6
    # This means v is on the correct side of each edge.
    
    # For a hexagon centered at origin with angle a, the vertices are at:
    # (s*cos(a + 60*i), s*sin(a + 60*i)) for i=0..5
    
    # The edges connect consecutive vertices. For edge from vertex i to vertex i+1,
    # the inward normal direction can be computed.
    
    # Let's parameterize: outer hexagon has vertices at angle a + 60*i with radius s.
    # Edge from v_i to v_{i+1}. The inward normal is perpendicular to edge, pointing toward center.
    
    # For a regular hexagon, the apothem (inradius) = s * cos(30°) = s * sqrt(3)/2
    # The distance from center to any edge is the apothem.
    
    # So the constraint is: for each edge k, the projection of each inner vertex onto the 
    # inward normal of edge k must be <= apothem = s * sqrt(3)/2
    
    # Compute inward normals for each edge of the outer hexagon (centered at origin, angle a)
    a_rad = np.radians(outer_angle)
    outer_verts_origin = np.column_stack([
        s_dummy * np.cos(a_rad + np.radians(60) * np.arange(6)),
        s_dummy * np.sin(a_rad + np.radians(60) * np.arange(6))
    ]) if False else None  # We'll compute normally without s
    
    # Actually, let's just compute the required s directly.
    # For each inner vertex v (relative to outer center), and for each outer edge direction,
    # we need the projection onto the inward normal to be <= s * sqrt(3)/2
    
    # The inward normal for edge k (from vertex k to k+1) of a hexagon at angle a:
    # The edge direction is v_{k+1} - v_k (for unit s). The inward normal is perpendicular, pointing inward.
    
    # Let's compute for unit hexagon (s=1) and scale.
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        # Inward normal: perpendicular to edge, pointing toward center
        # For a convex polygon, the inward normal for edge from p1 to p2 is:
        # Rotate edge by -90 degrees (clockwise) if the polygon is CCW
        # Let's check orientation: vertices are in CCW order for increasing angle
        # So inward normal = rotate edge by -90 degrees
        normal = np.array([edge[1], -edge[0]])  # Rotate -90 degrees
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # For unit hexagon, apothem = sqrt(3)/2
        # The projection of point v onto normal should be <= apothem * s
        # i.e., dot(v, normal) <= s * sqrt(3)/2
        # So s >= dot(v, normal) / (sqrt(3)/2) for all v
        
        for v in all_verts:
            proj = np.dot(v, normal)
            if proj > max_proj:
                max_proj = proj
    
    # s * sqrt(3)/2 >= max_proj
    # s >= max_proj * 2 / sqrt(3)
    min_s = max_proj * 2.0 / np.sqrt(3)
    
    return max(min_s, np.sqrt(12))
=======
def _compute_min_outer_side(inner, outer_center, outer_angle, smooth_beta=0.0):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons.
    
    If smooth_beta > 0, uses a soft-max surrogate for smoother optimization.
    """
    inner = np.array(inner, dtype=float)
    outer_center = np.array(outer_center, dtype=float)
    
    # Shift inner centers by minus outer center
    shifted_centers = inner[:, :2] - outer_center
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    # Collect all projections
    all_projs = []
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        projs = np.dot(all_verts, normal)
        all_projs.append(projs)
    
    all_projs = np.vstack(all_projs)  # shape (72, )
    
    if smooth_beta > 0:
        # Soft-max: (1/beta) * log(sum(exp(beta * x)))
        # This approximates max(x) as beta -> infinity
        beta = smooth_beta
        max_val = np.max(all_projs)
        # Numerically stable soft-max
        shifted = beta * (all_projs - max_val)
        soft_max = max_val + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    else:
        soft_max = np.max(all_projs)
    
    min_s = soft_max * 2.0 / np.sqrt(3)
    
    return max(min_s, np.sqrt(12))
>>>>>>> REPLACE

