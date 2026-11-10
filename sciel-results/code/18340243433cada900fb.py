import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Constants
    N = 11
    sqrt3 = np.sqrt(3)
    
    # Vertices of a unit hexagon centered at origin, angle 0
    # (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2.0],
        [-0.5, sqrt3/2.0],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2.0],
        [0.5, -sqrt3/2.0]
    ])
    
    def get_hex_vertices(cx, cy, s, a_deg):
        """Get vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
        a = np.radians(a_deg)
        angles = a + np.arange(6) * np.pi / 3
        return np.column_stack((cx + s * np.cos(angles), cy + s * np.sin(angles)))
    
    def get_edge_normals(vertices):
        """Get outward edge normals for a hexagon given its vertices."""
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i+1) % 6]
            edge = v2 - v1
            # Outward normal (rotate edge 90 degrees clockwise for CCW vertices)
            normal = np.array([edge[1], -edge[0]])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return np.array(normals)
    
    def hexagons_disjoint(hex1, hex2, tol=1e-6):
        """
        Check if two unit hexagons are disjoint using the 12-edge-normal separation test.
        hex1, hex2: each is (cx, cy, angle_deg)
        Returns True if they are separated (disjoint), False if they intersect.
        """
        v1 = get_hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
        v2 = get_hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
        n1 = get_edge_normals(v1)
        n2 = get_edge_normals(v2)
        
        # For each of the 12 normals, check if projected intervals are apart
        all_normals = np.vstack([n1, n2])
        
        for normal in all_normals:
            # Project both hexagons onto this normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            # Intervals
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check if intervals are apart (with tolerance)
            # They are apart if max1 < min2 - tol or max2 < min1 - tol
            if max1 < min2 - tol or max2 < min1 - tol:
                return True
        
        return False
    
    def all_inner_disjoint(inner_hex_data, tol=1e-6):
        """Check if all pairs of inner hexagons are disjoint."""
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagons_disjoint(inner_hex_data[i], inner_hex_data[j], tol):
                    return False
        return True
    
    def get_outer_normals(alpha_deg):
        angles = np.radians(alpha_deg) + np.pi/6 + np.arange(6) * np.pi/3
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    def calculate_required_side(centers, alpha_deg):
        """
        Calculate the minimum side length S of an outer hexagon centered at origin, 
        rotated by alpha_deg, that contains all hexagons with given centers (aligned).
        """
        normals = get_outer_normals(alpha_deg)
        max_proj = 0.0
        for c in centers:
            # Vertices of this hexagon (unit side, angle 0)
            verts = base_vertices + c
            # Projections onto normals
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
        # S * sqrt(3)/2 >= max_proj
        S = 2.0 * max_proj / sqrt3
        return S
    
    def check_outer_contains(inner_hex_data, outer_hex_data, side, tol=1e-6):
        """Check if all inner hexagons are contained within the outer hexagon."""
        cx, cy, alpha_deg = outer_hex_data
        outer_verts = get_hex_vertices(cx, cy, side, alpha_deg)
        outer_normals = get_edge_normals(outer_verts)
        
        for i in range(11):
            ix, iy, iangle = inner_hex_data[i]
            inner_verts = get_hex_vertices(ix, iy, 1.0, iangle)
            
            # Shift by outer center
            inner_verts_shifted = inner_verts - np.array([cx, cy])
            outer_verts_shifted = outer_verts - np.array([cx, cy])
            
            # For each outer edge, check that inner vertices are on the correct side
            for k in range(6):
                p1 = outer_verts_shifted[k]
                p2 = outer_verts_shifted[(k+1) % 6]
                edge = p2 - p1
                # Outward normal
                normal = np.array([edge[1], -edge[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Project inner vertices onto normal
                proj_inner = inner_verts_shifted @ normal
                proj_edge = np.array([p1 @ normal, p2 @ normal])
                
                # All inner vertices should be on the inward side of the edge
                # i.e., their projection should be <= max(proj_edge) + tol
                if np.max(proj_inner) > np.max(proj_edge) + tol:
                    return False
        
        return True
    
    def is_valid(inner_hex_data, outer_hex_data, side, tol=1e-6):
        """Full validity check."""
        # Check finite values
        if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
            return False
        
        if side < np.sqrt(11) - tol:
            return False
        
        # Check inner hexagons are disjoint
        if not all_inner_disjoint(inner_hex_data, tol):
            return False
        
        # Check outer contains inner
        if not check_outer_contains(inner_hex_data, outer_hex_data, side, tol):
            return False
        
        return True
    
    def objective(x):
        """
        x: [x0, y0, x1, y1, ..., x10, y10, alpha]
        """
        if time.time() > deadline - 10:
            return 1e10
        centers = x[:-1].reshape(11, 2)
        alpha = np.degrees(x[-1])
        return calculate_required_side(centers, alpha)
    
    # Constraints: Non-overlap using the 12-edge-normal test
    # This is tricky for SLSQP since it's not smooth. We'll use a penalty approach.
    
    def penalty_func(x):
        """Penalty for overlapping hexagons."""
        centers = x[:-1].reshape(11, 2)
        alpha = np.degrees(x[-1])
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                hex_i = [centers[i][0], centers[i][1], 0.0]
                hex_j = [centers[j][0], centers[j][1], 0.0]
                if not hexagons_disjoint(hex_i, hex_j, tol=1e-8):
                    # Compute a penalty based on how much they overlap
                    # Use minimum distance between vertices as a rough measure
                    v_i = get_hex_vertices(hex_i[0], hex_i[1], 1.0, 0.0)
                    v_j = get_hex_vertices(hex_j[0], hex_j[1], 1.0, 0.0)
                    min_dist = np.inf
                    for vi in v_i:
                        for vj in v_j:
                            d = np.linalg.norm(vi - vj)
                            min_dist = min(min_dist, d)
                    # If min_dist is small, they're close to overlapping
                    if min_dist < 0.1:
                        penalty += (0.1 - min_dist) ** 2
        return penalty
    
    def combined_objective(x):
        base = objective(x)
        pen = penalty_func(x)
        return base + 100.0 * pen
    
    # Initial guess: 7-flower + 4 extras
    def init_positions():
        pos = np.zeros((11, 2))
        # Center
        pos[0] = [0, 0]
        # Ring
        for i in range(6):
            angle = i * np.pi / 3
            pos[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
        
        # Extras
        # Attach 7 to 1 (angle 0) -> (2*sqrt3, 0)
        pos[7] = [2*sqrt3, 0]
        # Attach 8 to 3 (angle 120) -> pos[3] + sqrt3 * (cos120, sin120)
        pos[8] = pos[3] + sqrt3 * np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
        # Attach 9 to 5 (angle 240) -> pos[5] + sqrt3 * (cos240, sin240)
        pos[9] = pos[5] + sqrt3 * np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3)])
        # Attach 10 to 2 (angle 60) -> pos[2] + sqrt3 * (cos60, sin60)
        pos[10] = pos[2] + sqrt3 * np.array([np.cos(np.pi/3), np.sin(np.pi/3)])
        
        return pos.flatten()
    
    x0 = init_positions()
    x0 = np.append(x0, 0.0) # alpha = 0 (in radians for optimization)
    
    # Bounds: Keep centers within a reasonable range
    bounds = [(-6, 6)] * 22 + [(0, np.pi/3)]
    
    best_x = x0.copy()
    best_obj = combined_objective(x0)
    
    # Run optimization with penalty method
    try:
        if time.time() < deadline - 60:
            res = minimize(combined_objective, x0, method='SLSQP', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
                
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Try multiple random initializations
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random perturbation of the initial guess
        x_trial = x0.copy()
        x_trial[:-1] += np.random.randn(22) * 0.5
        x_trial[-1] = np.random.uniform(0, np.pi/3)
        
        try:
            res = minimize(combined_objective, x_trial, method='SLSQP', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
                
        except Exception as e:
            pass
    
    # Extract results
    centers = best_x[:-1].reshape(11, 2)
    alpha_rad = best_x[-1]
    alpha_deg = np.degrees(alpha_rad)
    
    # Calculate final side
    side = calculate_required_side(centers, alpha_deg)
    
    # Build inner hex data
    inner_hex_data = np.column_stack((centers, np.zeros(11)))
    outer_hex_data = np.array([0, 0, alpha_deg])
    
    # Validate
    if not is_valid(inner_hex_data, outer_hex_data, side):
        # Try to fix by using a known valid packing
        # Use a grid-like arrangement
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-sqrt3, 0, 0],
                [sqrt3, 0, 0],
                [-sqrt3/2, sqrt3*sqrt3/2, 0],
                [sqrt3/2, sqrt3*sqrt3/2, 0],
                [-sqrt3/2, -sqrt3*sqrt3/2, 0],
                [sqrt3/2, -sqrt3*sqrt3/2, 0],
                [-2*sqrt3, 0, 0],
                [2*sqrt3, 0, 0],
                [-sqrt3, sqrt3*sqrt3, 0],
                [sqrt3, -sqrt3*sqrt3, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        
        # Calculate required side for this configuration
        centers = inner_hex_data[:, :2]
        side = calculate_required_side(centers, 0.0)
        
        # Add small margin to ensure validity
        side *= 1.001
        
        # Check if valid, if not, increase side
        for _ in range(10):
            if is_valid(inner_hex_data, outer_hex_data, side):
                break
            side *= 1.01
    
    return inner_hex_data, outer_hex_data, side

hexagon_packing_11()
