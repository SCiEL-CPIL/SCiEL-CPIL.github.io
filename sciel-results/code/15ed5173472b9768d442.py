import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def get_hex_vertices_batch(xys_rots):
    """
    Batch compute vertices for all hexagons. xys_rots is (N, 3) array.
    Returns (N, 6, 2) array.
    """
    N_pts = len(xys_rots)
    # Base angles for a regular hexagon with a vertex at 0 degrees
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    
    # angles shape: (N, 6)
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)  # (N, 6)
    sin_a = np.sin(angles)  # (N, 6)
    
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    
    # Return shape (N, 6, 2)
    return np.stack((x, y), axis=-1)

def sat_overlap_batch(vertices):
    """
    Vectorized SAT overlap check for all pairs of hexagons.
    vertices: (N, 6, 2)
    Returns penalty (sum of squared penetration depths) and has_overlap boolean.
    """
    N_pts = len(vertices)
    if N_pts < 2:
        return 0.0, False
        
    # Precompute edge normals for all hexagons
    # Edges are v[i] to v[i+1]
    v_next = np.roll(vertices, -1, axis=1)
    edges = v_next - vertices # (N, 6, 2)
    
    # Normal to edge (dx, dy) is (-dy, dx)
    normals = np.stack((-edges[:,:,1], edges[:,:,0]), axis=-1) # (N, 6, 2)
    norms = np.linalg.norm(normals, axis=2, keepdims=True) # (N, 6, 1)
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    normals = normals / norms # (N, 6, 2)
    
    has_overlap = False
    penalty = 0.0
    
    # For each pair (i, j), check all 12 axes (6 from i, 6 from j)
    for i in range(N_pts):
        v1 = vertices[i]
        n1 = normals[i]
        
        for j in range(i+1, N_pts):
            v2 = vertices[j]
            n2 = normals[j]
            
            # Combine axes
            # We can just loop over the 12 axes for simplicity as N=12 is small
            # But vectorizing the projection is faster
            
            # Project v1 and v2 onto all 12 axes
            # Axes shape: (12, 2)
            all_axes = np.vstack((n1, n2))
            
            # Projections: (12, 6)
            proj1 = v1 @ all_axes.T
            proj2 = v2 @ all_axes.T
            
            min1 = np.min(proj1, axis=1)
            max1 = np.max(proj1, axis=1)
            min2 = np.min(proj2, axis=1)
            max2 = np.max(proj2, axis=1)
            
            # Check separation
            # If max1 < min2 or max2 < min1, they are separated on this axis
            sep1 = max1 < min2
            sep2 = max2 < min1
            separated = np.any(sep1 | sep2)
            
            if separated:
                continue
            
            # Calculate overlap depth for each axis
            # overlap = min(max1, max2) - max(min1, min2)
            overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
            
            # The minimum overlap is the penetration depth
            min_overlap = np.min(overlaps)
            
            if min_overlap > 1e-9:
                has_overlap = True
                penalty += min_overlap ** 2
                
    return penalty, has_overlap

def outer_hex_side_batch(vertices, outer_params):
    """
    Calculate the minimum side length of a regular hexagon enclosing all vertices.
    outer_params: [cx, cy, rot_deg]
    """
    cx, cy, rot_deg = outer_params
    rot_rad = np.radians(rot_deg)
    
    # Normals for outer hexagon
    # Vertices at 0, 60... => Normals at 30, 90...
    base_normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    normal_angles = base_normal_angles + rot_rad
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # (6, 2)
    
    # Flatten vertices to (N*6, 2)
    pts = vertices.reshape(-1, 2)
    
    # Translate points by -center
    pts_rel = pts - np.array([cx, cy])
    
    # Project onto normals
    # proj shape: (N*6, 6)
    proj = pts_rel @ normals.T
    
    # Max absolute projection
    max_proj = np.max(np.abs(proj))
    
    # Side length S = 2 / sqrt(3) * h, where h is apothem (max_proj)
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    def objective(params):
        if time.time() - start_time > DEADLINE - 2:
            return 1e6
            
        # Split params
        inner_params = params[:3*N].reshape(N, 3)
        outer_params = params[3*N:3*N+3]
        
        # Get vertices
        vertices = get_hex_vertices_batch(inner_params)
        
        # Check overlap
        penalty, has_overlap = sat_overlap_batch(vertices)
        
        # Calculate outer side
        L = outer_hex_side_batch(vertices, outer_params)
        
        return L + 100.0 * penalty

    def make_lattice_config(rows, cols, rot_offset=0.0, gap=0.0):
        """
        Create a hex grid of unit hexagons.
        Returns (N, 3) array of [x, y, rot]
        """
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        # Center the configuration
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_ring_config(gap=0.0):
        """
        1 center + 6 ring + 5 outer
        """
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_lattice_config(4, 3, gap=gap))
    configs.append(make_lattice_config(3, 4, gap=gap))
    configs.append(make_lattice_config(2, 6, gap=gap))
    configs.append(make_lattice_config(6, 2, gap=gap))
    configs.append(make_ring_config(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
        
    # Outer initial guesses: [cx, cy, rot]
    # We can try a few outer rotations and centers
    outer_inits = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 30.0],
        [0.0, 0.0, 15.0],
        [0.5, 0.5, 0.0],
        [-0.5, -0.5, 30.0]
    ]
    
    for cfg in configs:
        if time_left() < 10:
            break
            
        for o_init in outer_inits:
            if time_left() < 10:
                break
                
            init_params = np.concatenate((cfg.flatten(), o_init))
            
            # Evaluate initial
            val = objective(init_params)
            if val < best_L:
                # Need to extract L for comparison? 
                # objective returns L + penalty. If penalty is 0, it's L.
                # Check validity
                inner_params = init_params[:3*N].reshape(N, 3)
                outer_params = init_params[3*N:3*N+3]
                vertices = get_hex_vertices_batch(inner_params)
                penalty, has_overlap = sat_overlap_batch(vertices)
                if not has_overlap:
                    L = outer_hex_side_batch(vertices, outer_params)
                    if L < best_L:
                        best_L = L
                        best_params = init_params.copy()
                        
            try:
                # Optimize
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                # Evaluate result
                val_res = res.fun
                # Check validity
                inner_params = res.x[:3*N].reshape(N, 3)
                outer_params = res.x[3*N:3*N+3]
                vertices = get_hex_vertices_batch(inner_params)
                penalty, has_overlap = sat_overlap_batch(vertices)
                
                if not has_overlap:
                    L = outer_hex_side_batch(vertices, outer_params)
                    if L < best_L:
                        best_L = L
                        best_params = res.x.copy()
                        
            except Exception:
                pass

    # Final refinement if time permits
    if best_params is not None and time_left() > 20:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            inner_params = res.x[:3*N].reshape(N, 3)
            outer_params = res.x[3*N:3*N+3]
            vertices = get_hex_vertices_batch(inner_params)
            penalty, has_overlap = sat_overlap_batch(vertices)
            
            if not has_overlap:
                L = outer_hex_side_batch(vertices, outer_params)
                if L < best_L:
                    best_L = L
                    best_params = res.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        # Simple valid grid
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        # Calculate side for this fallback
        # Use outer_hex_side_batch with dummy outer params
        vertices = get_hex_vertices_batch(np.array(inner))
        L_val = outer_hex_side_batch(vertices, [0.0, 0.0, 0.0])
        return inner, [0.0, 0.0], float(L_val), 0.0

    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_params = best_params[3*N:3*N+3]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [float(outer_params[0]), float(outer_params[1])]
    side = float(best_L)
    angle = float(outer_params[2])
    
    return inner, center, side, angle
