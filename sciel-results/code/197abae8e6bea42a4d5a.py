import numpy as np
from scipy.optimize import differential_evolution

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (pointy-top)."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    # Precompute unit hexagon vertices for projection
    # Vertices of a unit hexagon centered at origin, rotated by angle a
    # We need the max dot product with a direction u.
    # max_v (u . v) where v are vertices.
    # This is equivalent to the support function h(u).
    # For a regular hexagon, h(u) = max_k |u . n_k| * (1/sqrt(3))? 
    # No, let's just compute it numerically for the 6 vertices.
    # Vertices angles: 0, 60, 120, 180, 240, 300 (flat top) or 30, 90... (pointy top)
    # Let's use pointy top to match normals.
    # Vertex angles: 90, 150, 210, 270, 330, 30? No.
    # Normals are at 30, 90, ...
    # Vertices are at 0, 60, 120... (flat top) relative to normal 0?
    # If normal is at 90 (up), vertex is at 90 (up).
    # Let's define vertices for a hexagon with normal at 0 (right).
    # Vertices at 0, 60, 120, 180, 240, 300?
    # If normal is at 0, the face is vertical? No, normal is perpendicular to face.
    # If normal is (1,0), the face is the right-most vertical edge? No, hexagon has flat faces.
    # Regular hexagon with flat top: normals at 0, 60, 120...
    # Vertices at 30, 90, 150...
    # Let's stick to the normals defined in _hex_normals: 30, 90, 150, 210, 270, 330.
    # This corresponds to a "pointy top" hexagon?
    # If normal is at 90 (up), the top edge is horizontal. So it's flat top.
    # Vertices for flat top hexagon (normal at 0 is right face):
    # Angles: 30, 90, 150, 210, 270, 330? No, those are normals.
    # Vertices are at 0, 60, 120, 180, 240, 300?
    # Let's check: Normal at 30 deg. Face is perpendicular to 30 deg.
    # Midpoint of face is at distance apothem = sqrt(3)/2.
    # Vertices are at angles 0, 60, 120, 180, 240, 300.
    
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def get_support(c, angle_rad, u):
        """Calculate max dot product of vertices of hexagon at c with rotation angle_rad with direction u."""
        # Rotate base vertices
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = (base_vertices @ rot_mat.T) + c
        dots = verts @ u
        return np.max(dots)

    def build_centers(params):
        r_in, r_out, rot_in, rot_out = params
        rot_in_rad = np.deg2rad(rot_in)
        rot_out_rad = np.deg2rad(rot_out)
        centers = np.zeros((N, 2))
        angles_in_rad = np.deg2rad([0, 72, 144, 216, 288])
        angles_out_rad = np.deg2rad([0, 60, 120, 180, 240, 300])
        
        centers[0] = [0.0, 0.0]
        
        for k in range(5):
            a = angles_in_rad[k] + rot_in_rad
            centers[1 + k] = [r_in * np.cos(a), r_in * np.sin(a)]
            
        for k in range(6):
            a = angles_out_rad[k] + rot_out_rad
            centers[6 + k] = [r_out * np.cos(a), r_out * np.sin(a)]
            
        return centers

    def calc_L(centers, angles):
        """Calculate minimum container side length for given centers and angles."""
        max_extent = 0.0
        for k in range(6):
            u = normals[k]
            max_proj = 0.0
            for i in range(N):
                proj = get_support(centers[i], angles[i], u)
                if proj > max_proj:
                    max_proj = proj
            # L * sqrt(3) / 2 >= max_proj
            L_k = 2.0 * max_proj / sqrt3
            if L_k > max_extent:
                max_extent = L_k
        return max_extent

    def calc_violation(centers, angles):
        """Calculate max separation violation."""
        max_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                # Check if distance is too small
                # We need max_k |u_k . diff| >= sqrt(3)
                # If not, violation is sqrt(3) - max_k |u_k . diff|
                d_dots = np.abs(diff @ normals.T)
                max_d_dot = np.max(d_dots)
                viol = sqrt3 - max_d_dot
                if viol > max_viol:
                    max_viol = viol
        return max_viol

    def objective(params):
        centers = build_centers(params)
        # Angles: 
        # Center: 0
        # Inner ring: rot_in
        # Outer ring: rot_out
        angles = np.zeros(N)
        angles[1:6] = np.deg2rad(params[2])
        angles[6:12] = np.deg2rad(params[3])
        
        L = calc_L(centers, angles)
        viol = calc_violation(centers, angles)
        
        # Penalty for violation
        # Use a large penalty to enforce separation
        penalty = 1000.0 * viol
        return L + penalty

    # Bounds for parameters
    # r_in: 1.7 to 2.5
    # r_out: 3.0 to 4.5
    # rot_in: 0 to 72
    # rot_out: 0 to 60
    bounds = [
        (1.5, 3.0),
        (2.5, 5.0),
        (0.0, 72.0),
        (0.0, 60.0)
    ]
    
    # Use differential evolution for global search
    # This is slower but more robust for non-convex problems
    # To keep within 120s, we limit maxiter and popsize
    try:
        result = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=10, tol=1e-5, polish=True)
        best_params = result.x
    except Exception:
        best_params = np.array([2.0, 3.5, 0.0, 0.0])
        
    # Final verification and refinement
    centers = build_centers(best_params)
    angles = np.zeros(N)
    angles[1:6] = np.deg2rad(best_params[2])
    angles[6:12] = np.deg2rad(best_params[3])
    
    viol = calc_violation(centers, angles)
    if viol > 1e-5:
        # If invalid, try a known valid configuration or fallback
        # Fallback: spread them out
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    L_final = calc_L(centers, angles)
    
    # Center the cluster
    # The container is centered at (0,0) in our calculation.
    # The inner hexagons are at `centers`.
    # We return inner rows (x, y, angle_degrees).
    inner = [[centers[i, 0], centers[i, 1], np.rad2deg(angles[i])] for i in range(N)]
    
    return inner, (0.0, 0.0), L_final, 0.0
