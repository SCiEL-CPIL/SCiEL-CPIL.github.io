# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution


def _hexagon_vertices(cx, cy, d, a_deg):
    """Return the 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _edge_normals(vertices):
    """Return unit outward normals for each edge of a convex polygon (vertices in CCW order)."""
    n = len(vertices)
    normals = np.zeros_like(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # For CCW polygon, the outward normal is (edge_y, -edge_x) normalized
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals


def _check_packing(inner_hex_data, outer_hex_data, s, tol_edge=1e-8, tol_sep=1e-6, tol_inside=1e-6):
    """Validate the packing configuration."""
    n_inner = len(inner_hex_data)
    
    # Check finite values
    if not np.all(np.isfinite(inner_hex_data)):
        return False
    if not np.all(np.isfinite(outer_hex_data)):
        return False
    if not np.isfinite(s) or s < np.sqrt(12) - 1e-12:
        return False
    
    outer_center = np.array(outer_hex_data[:2])
    outer_angle = outer_hex_data[2]
    
    # Get outer hexagon vertices
    outer_verts = _hexagon_vertices(outer_center[0], outer_center[1], s, outer_angle)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Get inner hexagons' vertices
    inner_verts = []
    for i in range(n_inner):
        cx, cy, angle = inner_hex_data[i]
        verts = _hexagon_vertices(cx, cy, 1.0, angle)
        inner_verts.append(verts)
    
    # Check unit edges
    for i in range(n_inner):
        verts = inner_verts[i]
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j + 1) % 6]
            length = np.linalg.norm(p2 - p1)
            if abs(length - 1.0) > tol_edge:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            verts_i = inner_verts[i]
            verts_j = inner_verts[j]
            normals_i = _edge_normals(verts_i)
            normals_j = _edge_normals(verts_j)
            
            all_normals = np.vstack([normals_i, normals_j])
            
            separated = False
            for normal in all_normals:
                proj_i = verts_i @ normal
                proj_j = verts_j @ normal
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                gap = min_j - max_i
                if gap > tol_sep:
                    separated = True
                    break
            
            if not separated:
                # Check the other direction too (though SAT should handle it)
                # Actually, SAT says if there's ANY separating axis, they're separated
                # We need to check if they overlap
                # If no axis gives a gap > tol_sep, they might be touching or overlapping
                # For safety, we consider them invalid if gap <= tol_sep for all axes
                # But we need to be careful: if gap is negative, they definitely overlap
                # If gap is positive but <= tol_sep, it's borderline
                # The spec says "more than 1e-6 apart" is valid, so <= 1e-6 is invalid?
                # Re-reading: "two inner hexagons not separated: some edge normal ... must give projection intervals more than 1e-6 apart"
                # This phrasing is tricky. It says the condition for "not separated" is that some normal gives intervals more than 1e-6 apart?
                # No, that can't be right. That would mean if they are far apart, they are "not separated"?
                # Let's re-read: "Invalid: ... two inner hexagons not separated: some edge normal of either one (separating axis test) must give projection intervals more than 1e-6 apart"
                # This implies: If they are NOT separated, then there EXISTS an axis where the gap is > 1e-6.
                # Contrapositive: If for ALL axes the gap is <= 1e-6, then they ARE separated?
                # That doesn't make sense. SAT says: Polygons are separated iff there EXISTS an axis where the projections are disjoint.
                # If they are disjoint (separated), there is an axis with gap > 0.
                # The condition "projection intervals more than 1e-6 apart" means gap > 1e-6.
                # So "not separated" means: For ALL axes, gap <= 1e-6.
                # So the check is: If for ALL axes, gap <= 1e-6, then it's invalid.
                # Which means we need: There EXISTS an axis with gap > 1e-6.
                
                # So my loop above checks if ANY axis has gap > tol_sep.
                # If it does, they are "separated" (valid).
                # If it doesn't, they are "not separated" (invalid).
                # So the logic is correct. If `separated` is False, return False.
                return False
    
    # Check all inner vertices are inside the outer hexagon
    # Outer hexagon is convex. For each edge (p1, p2), check cross(p2-p1, v-p1) >= -tol
    # Note: This assumes CCW ordering of outer vertices.
    for v in [verts[i] for verts in inner_verts for i in range(6)]:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            vec = v - p1
            cross = edge[0] * vec[1] - edge[1] * vec[0]
            if cross < -tol_inside:
                return False
    
    return True


def _packing_cost(params):
    """Calculate cost for a given packing configuration."""
    # params: 12*3 (inner centers and angles) + 3 (outer center and angle)
    # But we fix outer center to (0,0) for simplicity? Or let it vary?
    # Let's fix outer center to (0,0) to reduce parameters.
    # We also need to determine s.
    # Given the configuration, s is determined by the minimal enclosing hexagon.
    # But the outer hexagon's angle is a parameter.
    # Let's parameterize:
    # params[0:36] = inner hexagons (x, y, angle)
    # params[36] = outer angle
    # We fix outer center at (0,0).
    
    n_inner = 12
    inner_params = params[:n_inner * 3].reshape(n_inner, 3)
    outer_angle = params[n_inner * 3]
    
    # Shift inner centers so their centroid is at origin?
    # No, the outer hexagon is at origin. We can just place them.
    # But to keep the search space compact, let's shift inner centers by their centroid.
    centers = inner_params[:, :2]
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Now, for this configuration and outer angle, find minimal s.
    # s is such that all inner hexagons are inside the outer hexagon.
    # Outer hexagon: center (0,0), side s, angle outer_angle.
    # Vertices of outer hexagon: (s*cos(theta), s*sin(theta)) for theta in outer_angle + k*60 deg.
    # Edges of outer hexagon: between consecutive vertices.
    # For a point p to be inside, for each edge, the cross product check must pass.
    # Alternatively, we can compute the required s for each inner hexagon vertex.
    # For a given outer angle theta_out, the "radius" in direction phi is:
    # r(phi) = s * sqrt(3) / (2 * cos(phi - theta_out + 30 deg)) ? No.
    # Let's use the distance to edge method.
    # For a regular hexagon centered at origin with angle theta, the distance from center to the edge with normal n_k is s * sqrt(3) / 2.
    # The normal directions are theta + 30 deg + k * 60 deg.
    # For a point p, the condition to be inside is:
    # p . n_k <= s * sqrt(3) / 2 for all k.
    # So s >= 2 / sqrt(3) * max_k (p . n_k) for all p.
    
    theta_out = np.deg2rad(outer_angle)
    normals = []
    for k in range(6):
        angle = theta_out + np.deg2rad(30 + k * 60)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)  # shape (6, 2)
    
    # Compute max projection for all inner vertices
    max_proj = 0
    for i in range(n_inner):
        cx, cy, angle = inner_params[i, 0] - centroid[0], inner_params[i, 1] - centroid[1], inner_params[i, 2]
        # Wait, I already shifted centers. So cx, cy are relative to centroid.
        # But the inner hexagons also have angles.
        # We need to check all 6 vertices of each inner hexagon.
        verts = _hexagon_vertices(cx, cy, 1.0, angle)
        for v in verts:
            projs = verts @ normals.T  # shape (6,)
            max_proj = max(max_proj, np.max(projs))
    
    s = 2 * max_proj / np.sqrt(3)
    
    # Also need to check if inner hexagons are separated.
    # For now, let's just return s as the cost. Separation will be checked separately or penalized.
    # If we don't enforce separation, the optimizer might overlap them.
    # We can add a penalty for overlap.
    
    # Check separation
    penalty = 0
    inner_verts = []
    for i in range(n_inner):
        cx, cy, angle = inner_params[i, 0] - centroid[0], inner_params[i, 1] - centroid[1], inner_params[i, 2]
        verts = _hexagon_vertices(cx, cy, 1.0, angle)
        inner_verts.append(verts)
    
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            verts_i = inner_verts[i]
            verts_j = inner_verts[j]
            normals_i = _edge_normals(verts_i)
            normals_j = _edge_normals(verts_j)
            all_normals = np.vstack([normals_i, normals_j])
            
            min_gap = np.inf
            for normal in all_normals:
                proj_i = verts_i @ normal
                proj_j = verts_j @ normal
                gap = np.min(proj_j) - np.max(proj_i)
                # We want gap > 1e-6. If gap < 0, they overlap.
                # Penalize if gap < 1e-6.
                if gap < 1e-6:
                    penalty += max(0, 1e-6 - gap) ** 2
    
    return s + penalty


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s).
    """
    # Use differential evolution to find a good packing
    # Parameters: 12 inner hexagons (x, y, angle) + 1 outer angle
    # We fix outer center at (0,0).
    # We fix inner angles to 0 for simplicity? Or let them vary?
    # Let's let them vary.
    # Bounds for inner centers: [-5, 5]
    # Bounds for angles: [-180, 180]
    
    n_inner = 12
    bounds = []
    for i in range(n_inner):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((-180, 180))
    bounds.append((-180, 180))  # outer angle
    
    # Run differential evolution
    # This might be slow, so we use a small population and few iterations
    result = differential_evolution(
        _packing_cost,
        bounds,
        maxiter=100,
        popsize=10,
        tol=1e-4,
        seed=42
    )
    
    params = result.x
    n_inner = 12
    inner_params = params[:n_inner * 3].reshape(n_inner, 3)
    outer_angle = params[n_inner * 3]
    
    # Shift inner centers so centroid is at origin
    centers = inner_params[:, :2].copy()
    centroid = np.mean(centers, axis=0)
    centers -= centroid
    
    # Construct inner hex data
    inner_hex_data = np.zeros((n_inner, 3))
    inner_hex_data[:, :2] = centers
    inner_hex_data[:, 2] = inner_params[:, 2]
    
    # Compute s
    theta_out = np.deg2rad(outer_angle)
    normals = []
    for k in range(6):
        angle = theta_out + np.deg2rad(30 + k * 60)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    max_proj = 0
    for i in range(n_inner):
        cx, cy, angle = inner_hex_data[i, 0], inner_hex_data[i, 1], inner_hex_data[i, 2]
        verts = _hexagon_vertices(cx, cy, 1.0, angle)
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    s = 2 * max_proj / np.sqrt(3)
    
    outer_hex_data = np.array([0, 0, outer_angle])
    
    # Verify packing
    if not _check_packing(inner_hex_data, outer_hex_data, s):
        # Fallback to a simple valid packing
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        s = 8
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
