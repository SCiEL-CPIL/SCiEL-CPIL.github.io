# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])


def _min_separation(v1, v2):
    """
    Compute the minimum separation between two convex hexagons using SAT.
    Returns the minimum gap (positive if separated, negative if overlapping).
    """
    # Get edge normals for both hexagons
    def edge_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        # Normal perpendicular to each edge (pointing inward for convex)
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return normals / norms

    n1 = edge_normals(v1)
    n2 = edge_normals(v2)

    min_gap = np.inf

    for n in np.vstack([n1, n2]):
        proj1 = v1 @ n
        proj2 = v2 @ n
        gap = min(proj2.max(), proj1.max()) - max(proj1.min(), proj2.min())
        # gap > 0 means separated, gap < 0 means overlapping
        min_gap = min(min_gap, gap)

    return min_gap


def _all_separated(inner_verts_list):
    """Check if all inner hexagons are separated. Returns min gap."""
    min_gap = np.inf
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            gap = _min_separation(inner_verts_list[i], inner_verts_list[j])
            min_gap = min(min_gap, gap)
    return min_gap


def _max_vertex_dist_from_center(outer_verts):
    """Find the vertex of outer hexagon that is farthest... no, we need the min s needed."""
    pass


def _required_s(inner_verts_list, ocx, ocy, oangle):
    """
    Compute the minimum side length s of an outer hexagon centered at (ocx, ocy) 
    with angle oangle that contains all inner hexagon vertices.
    """
    # Outer hexagon with side s, center (ocx, ocy), angle oangle
    # Vertices: (ocx + s*cos(oangle + 60i), ocy + s*sin(oangle + 60i))
    # For a point to be inside, it must be on the correct side of each edge.
    # Edge i goes from vertex i to vertex i+1.
    # The constraint is: for each edge, cross(edge_vec, point - vertex_i) >= 0 (for CCW)
    
    # Actually, let's think about this differently.
    # A regular hexagon with side s and angle a has its vertices at distance s from center.
    # The distance from center to each edge (apothem) is s * cos(30°) = s * sqrt(3)/2.
    # The distance from center to each vertex is s.
    
    # For a point P to be inside the hexagon, for each of the 6 edges, 
    # the projection of (P - center) onto the outward normal of that edge must be <= apothem.
    
    # The outward normal of edge i (between vertex i and vertex i+1) points in direction 
    # oangle + 60*i + 30 degrees (the bisector of the angle at the center for that edge).
    # Wait, let me reconsider.
    
    # Vertex i is at angle oangle + 60*i from center.
    # Edge i connects vertex i and vertex i+1.
    # The outward normal of edge i points in the direction of the midpoint of the arc from vertex i to vertex i+1,
    # which is at angle oangle + 60*i + 30 degrees.
    
    # The apothem (distance from center to edge) is s * cos(30°) = s * sqrt(3)/2.
    
    # For point P, let d = P - center.
    # The projection of d onto the outward normal of edge i is d · n_i where n_i = (cos(oangle + 60i + 30), sin(oangle + 60i + 30)).
    # We need this projection <= apothem = s * sqrt(3)/2 for all i.
    
    # So s >= 2/sqrt(3) * max_i (d · n_i)
    
    angles = np.radians(oangle) + np.radians(60 * np.arange(6)) + np.radians(30)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # For each inner vertex, compute the max projection onto any normal
    # Then s >= 2/sqrt(3) * max over all vertices and all normals
    
    max_proj = 0
    for verts in inner_verts_list:
        # verts is (6, 2) array of absolute coordinates
        # d = verts - center
        d = verts - np.array([ocx, ocy])
        # Projections: d @ normals.T gives (6, 6) matrix
        projs = d @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    return 2.0 / np.sqrt(3.0) * max_proj


def hexagon_packing_12():
    """
    Optimizes packing of 12 unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    n = 12
    
    # Initial guess: hexagonal grid arrangement
    # Unit hexagons in a honeycomb pattern
    # Axial coordinates to Cartesian: x = 1.5 * q, y = sqrt(3)/2 * (q + 2*r)
    # For side length 1, adjacent centers are distance sqrt(3) apart
    
    sqrt3 = np.sqrt(3)
    
    # 12 hexagons in a compact arrangement
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # This gives a 3x4 block, but let's use a more compact hexagonal shape
    
    # Better: 1+2+3+2+1 = 9... need 12
    # Let's use 2+3+3+2+2 = 12 or similar
    
    # Actually, let's start with a simple grid and let optimization fix it
    # Initial positions in a loose grid
    init_positions = []
    # 4 columns x 3 rows
    for row in range(3):
        for col in range(4):
            x = col * 1.5
            y = row * sqrt3 + (sqrt3 / 2 if col % 2 == 1 else 0)
            init_positions.append([x, y, 0.0])
    
    init_positions = np.array(init_positions[:12])
    
    # Optimize using gradient-free method (Nelder-Mead or similar)
    # Variables: for each inner hex: (x, y, angle) = 36 variables
    # Plus outer hex: (ocx, ocy, oangle) = 3 variables
    # Total: 39 variables
    
    # We'll use a simple simulated annealing / random restart approach
    # with local refinement
    
    best_result = None
    best_score = -np.inf
    
    # Use scipy's minimize if available
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    def objective(params):
        """
        Objective: minimize s (outer hex side length)
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, ocx, ocy, oangle]
        """
        inner_params = params[:36].reshape(12, 3)
        ocx, ocy, oangle = params[36:39]
        
        # Compute inner hexagon vertices
        inner_verts_list = []
        for i in range(12):
            cx, cy, angle = inner_params[i]
            verts = _hex_vertices(cx, cy, 1.0, angle)
            inner_verts_list.append(verts)
        
        # Check separation
        min_sep = _all_separated(inner_verts_list)
        if min_sep < -1e-6:
            return 1e6 + (min_sep + 1e-6) * 1e4  # Penalty for overlap
        
        # Compute required s
        s = _required_s(inner_verts_list, ocx, ocy, oangle)
        
        # Add penalty if s is too small (shouldn't happen if separation is ok)
        if s < np.sqrt(12) - 1e-6:
            return 1e6 + (np.sqrt(12) - s) * 1e4
        
        return s
    
    if has_scipy:
        # Try multiple random restarts
        np.random.seed(42)
        
        for restart in range(50):
            if time.time() > deadline - 60:
                break
            
            # Perturb initial guess
            noise_scale = 0.5 if restart == 0 else 1.0 + restart * 0.1
            x0 = init_positions + np.random.randn(12, 3) * noise_scale
            x0[:, 2] = np.random.randn(12) * 30  # Random angles
            outer_init = np.random.randn(3) * 2.0
            x0_full = np.concatenate([x0.flatten(), outer_init])
            
            result = minimize(objective, x0_full, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
            
            if time.time() > deadline - 30:
                break
    
    else:
        # Fallback: simple random search
        np.random.seed(42)
        
        for iteration in range(100000):
            if time.time() > deadline - 30:
                break
            
            # Random perturbation around best or initial
            if best_result is not None and np.random.rand() < 0.5:
                base = best_result
                scale = 0.3
            else:
                base = np.concatenate([init_positions.flatten(), np.zeros(3)])
                scale = 1.0
            
            candidate = base + np.random.randn(39) * scale
            
            score = objective(candidate)
            if score < best_score:
                best_score = score
                best_result = candidate.copy()
    
    # Extract best result
    if best_result is None:
        # Fallback to initial
        best_result = np.concatenate([init_positions.flatten(), np.zeros(3)])
        best_score = objective(best_result)
    
    inner_params = best_result[:36].reshape(12, 3)
    ocx, ocy, oangle = best_result[36:39]
    
    # Compute final s
    inner_verts_list = []
    for i in range(12):
        cx, cy, angle = inner_params[i]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        inner_verts_list.append(verts)
    
    s = _required_s(inner_verts_list, ocx, ocy, oangle)
    
    # Ensure s is at least sqrt(12)
    s = max(s, np.sqrt(12))
    
    inner_hex_data = inner_params
    outer_hex_data = np.array([ocx, ocy, oangle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
