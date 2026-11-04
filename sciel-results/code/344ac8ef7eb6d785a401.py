import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 145.0

def _hex_vertices_local():
    """Vertices of a unit hexagon centered at origin, rot 0."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

LOCAL_VERTS = _hex_vertices_local()

def _calc_side_length(centers, rot_deg):
    """
    Calculate the side length of the minimal regular hexagon (fixed orientation)
    that contains the cluster of hexagons defined by centers, rotated by rot_deg.
    """
    theta = np.radians(rot_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    # Rotate centers
    rotated_centers = centers @ rot_matrix.T
    
    # Rotate local vertices
    rotated_local_verts = LOCAL_VERTS @ rot_matrix.T
    
    # Construct all vertices
    # We can do this efficiently:
    # Verts = Rotated_Centers[:, np.newaxis, :] + Rotated_Local_Verts[np.newaxis, :, :]
    # Shape: (12, 6, 2)
    all_verts = rotated_centers[:, np.newaxis, :] + rotated_local_verts[np.newaxis, :, :]
    all_verts = all_verts.reshape(-1, 2)
    
    # Normals of the outer hexagon (fixed at 0 orientation)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Projections
    # proj[i, k] = v_i . n_k
    proj = all_verts @ normals.T
    max_proj = np.max(np.abs(proj))
    
    # Side length S such that apothem a = S * sqrt(3) / 2 >= max_proj
    # S = 2 * max_proj / sqrt(3)
    return (2.0 / np.sqrt(3.0)) * max_proj

def _make_grid(rows, cols):
    """Create a grid of hexagons in axial coordinates converted to Cartesian."""
    pts = []
    sqrt3 = np.sqrt(3)
    for r in range(rows):
        for c in range(cols):
            x = c * 2.0 + (r % 2) * 1.0
            y = r * sqrt3
            pts.append([x, y])
    
    pts = np.array(pts)
    # Center the grid
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _optimize_cluster(centers):
    """Find the best rotation for a given cluster topology."""
    def objective(theta_deg):
        return _calc_side_length(centers, theta_deg)
    
    # Grid search first to avoid local minima
    # Step 0.1 degrees
    best_theta = 0.0
    best_val = float('inf')
    for t in np.arange(0, 60, 0.1):
        val = objective(t)
        if val < best_val:
            best_val = val
            best_theta = t
            
    # Local refinement
    try:
        res = minimize_scalar(objective, bounds=(best_theta - 1.0, best_theta + 1.0), method='bounded', options={'maxiter': 500, 'xatol': 1e-6})
        if res.success:
            best_theta = res.x
            best_val = res.fun
    except Exception:
        pass
        
    return best_theta, best_val

def construct():
    t0 = time.time()
    
    # Define candidate topologies
    topologies = [
        "3x4",
        "4x3",
        "2x6",
        "6x2",
        "1x12",
        "12x1",
    ]
    
    best_side = float('inf')
    best_centers = None
    best_theta = 0.0
    
    for topo_name in topologies:
        if time.time() - t0 > DEADLINE:
            break
            
        if topo_name == "3x4":
            centers = _make_grid(3, 4)
        elif topo_name == "4x3":
            centers = _make_grid(4, 3)
        elif topo_name == "2x6":
            centers = _make_grid(2, 6)
        elif topo_name == "6x2":
            centers = _make_grid(6, 2)
        elif topo_name == "1x12":
            centers = _make_grid(1, 12)
        elif topo_name == "12x1":
            centers = _make_grid(12, 1)
        else:
            continue
            
        # Ensure we have exactly 12 points
        if len(centers) != 12:
            continue
            
        theta_opt, side_opt = _optimize_cluster(centers)
        
        if side_opt < best_side:
            best_side = side_opt
            best_centers = centers
            best_theta = theta_opt
    
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = _make_grid(3, 4)
        best_theta = 0.0
        best_side = _calc_side_length(best_centers, best_theta)
        
    # Prepare output
    # Inner hexagons: (x, y, angle)
    # We apply the optimal rotation to the centers
    theta_rad = np.radians(best_theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    rotated_centers = best_centers @ rot_matrix.T
    
    inner = []
    for i in range(12):
        x = float(rotated_centers[i, 0])
        y = float(rotated_centers[i, 1])
        angle = float(best_theta)
        inner.append([x, y, angle])
        
    outer_center = [0.0, 0.0]
    outer_side = float(best_side) * 1.0001 # Small safety margin
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
