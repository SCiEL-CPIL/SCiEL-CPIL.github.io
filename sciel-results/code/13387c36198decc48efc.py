def objective(params):
    # ... same setup ...
    
    all_verts = np.vstack(polys)
    max_req = 0
    for n in normals:
        projections = np.dot(all_verts, n)
        max_proj = np.max(projections)
        min_proj = np.min(projections)
        # Distance from center to furthest point in this direction
        # We need the outer hexagon to contain the shape.
        # The outer hexagon centered at (0,0) with side L has apothem A = L*sqrt(3)/2.
        # So we need max_proj <= A and min_proj >= -A.
        # So A >= max(max_proj, -min_proj).
        req_a = max(max_proj, -min_proj)
        if req_a > max_req:
            max_req = req_a
            
    return max_req * 2 / np.sqrt(3)
