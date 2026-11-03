import math
import time
import numpy as np
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def get_topology_3x4():
    """Generate centers for a 3x4 parallelogram of hexagons."""
    # Lattice vectors for unit hexagons (distance sqrt(3))
    # u = (1.5, sqrt(3)/2), v = (1.5, -sqrt(3)/2)
    u = np.array([1.5, math.sqrt(3)/2])
    v = np.array([1.5, -math.sqrt(3)/2])
    
    centers = []
    for i in range(4):
        for j in range(3):
            c = i * u + j * v
            centers.append(c)
    return np.array(centers)

def get_topology_flower_plus():
    """Generate centers for 7-hex flower plus 5 more."""
    u = np.array([1.5, math.sqrt(3)/2])
    v = np.array([1.5, -math.sqrt(3)/2])
    
    # Flower: (0,0) and 6 neighbors
    # Axial coords (q,r) -> Cartesian: q*u + r*v? 
    # Let's use the vectors directly.
    # Neighbors in axial (q,r): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Cartesian for (q,r) is q*u + r*v?
    # Let's verify distance.
    # (1,0) -> u. Dist 1.5, sqrt(3)/2. Len sqrt(3). OK.
    # (0,1) -> v. Len sqrt(3). OK.
    # (1,-1) -> u - v = (0, sqrt(3)). Len sqrt(3). OK.
    
    centers = [np.array([0.0, 0.0])]
    neighbors_axial = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for q, r in neighbors_axial:
        c = q * u + r * v
        centers.append(c)
        
    # Add 5 more. 
    # Try adding them in a "cap" on one side to make it more compact?
    # Or symmetric?
    # Let's try adding (2,0), (2,-1), (1,-2), (0,-2), (-1,-1)
    extra_axial = [(2,0), (2,-1), (1,-2), (0,-2), (-1,-1)]
    for q, r in extra_axial:
        c = q * u + r * v
        centers.append(c)
        
    return np.array(centers)

def solve_min_side(centers):
    """
    Find minimum outer hexagon side length for a given set of hexagon centers.
    Assumes all inner hexagons are aligned with the cluster rotation.
    Uses LP to find optimal translation and L for a given rotation theta.
    """
    # Pre-calculate vertices for each hexagon at origin, rotation 0
    # We will rotate the whole cluster by theta.
    base_verts = []
    for c in centers:
        # Vertices relative to center
        verts = hex_vertices(0, 0, 0, 1.0)
        # Shift to center
        verts = verts + c
        base_verts.append(verts)
    
    all_verts = np.vstack(base_verts) # Shape (72, 2)
    
    # Normals for the outer hexagon (aligned with axes)
    # Hexagon sides are at angles 0, 60, 120.
    # Normals are at 90, 150, 210 (or -90, -30, 30).
    # Let's use 3 pairs of normals.
    # Normal 1: (0, 1) -> y-axis. Constraint |y| <= apothem.
    # Normal 2: (sqrt(3)/2, 1/2) -> 30 deg.
    # Normal 3: (sqrt(3)/2, -1/2) -> -30 deg.
    
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5]
    ])
    
    def calc_L(theta_rad):
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        # Rotation matrix
        # R = [[cos, -sin], [sin, cos]]
        # Rotated vertices:
        # x' = x cos - y sin
        # y' = x sin + y cos
        
        rotated_x = all_verts[:, 0] * cos_t - all_verts[:, 1] * sin_t
        rotated_y = all_verts[:, 0] * sin_t + all_verts[:, 1] * cos_t
        
        # We want to find x_trans, y_trans, L such that:
        # For each vertex (vx, vy) and each normal n_k:
        # n_kx * (vx + x_trans) + n_ky * (vy + y_trans) <= (sqrt(3)/2) * L
        # n_kx * (vx + x_trans) + n_ky * (vy + y_trans) >= -(sqrt(3)/2) * L
        
        # Variables: x_trans, y_trans, L
        # c = [0, 0, 1]
        c = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        apothem_coeff = math.sqrt(3) / 2
        
        for k in range(3):
            nx = normals[k, 0]
            ny = normals[k, 1]
            
            # Constraints for all vertices
            # (nx*vx + ny*vy) + nx*x_trans + ny*y_trans - apothem_coeff*L <= 0
            # -(nx*vx + ny*vy) - nx*x_trans - ny*y_trans - apothem_coeff*L <= 0
            
            # Calculate constant terms
            dot_pos = rotated_x * nx + rotated_y * ny
            dot_neg = -rotated_x * nx - rotated_y * ny
            
            # Rows for positive constraint
            # [nx, ny, -apothem_coeff] . [xt, yt, L] <= -dot_pos
            row_pos = np.tile(np.array([nx, ny, -apothem_coeff]), (len(all_verts), 1))
            b_pos = -dot_pos
            
            # Rows for negative constraint
            # [-nx, -ny, -apothem_coeff] . [xt, yt, L] <= -dot_neg
            row_neg = np.tile(np.array([-nx, -ny, -apothem_coeff]), (len(all_verts), 1))
            b_neg = -dot_neg
            
            A_ub.append(row_pos)
            b_ub.append(b_pos)
            A_ub.append(row_neg)
            b_ub.append(b_neg)
            
        A_ub = np.vstack(A_ub)
        b_ub = np.concatenate(b_ub)
        
        # Bounds
        # L >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.fun
            else:
                return float('inf')
        except Exception:
            return float('inf')
            
    # Scan theta
    best_L = float('inf')
    best_theta = 0.0
    
    # Scan 0 to 30 degrees (pi/6)
    thetas = np.linspace(0, math.pi/6, 61)
    for t in thetas:
        L = calc_L(t)
        if L < best_L:
            best_L = L
            best_theta = t
            
    # Refine around best_theta
    if best_L < float('inf'):
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(calc_L, bounds=(best_theta - 0.05, best_theta + 0.05), method='bounded')
        if res.success and res.fun < best_L:
            best_L = res.fun
            best_theta = res.x
            
    return best_L, best_theta

def construct():
    start_time = time.time()
    
    topologies = [
        get_topology_3x4(),
        get_topology_flower_plus()
    ]
    
    best_L = float('inf')
    best_result = None
    
    for centers in topologies:
        if time.time() - start_time > 110:
            break
            
        L, theta = solve_min_side(centers)
        
        if L < best_L:
            best_L = L
            # Reconstruct inner coordinates
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            # Find optimal translation from LP? 
            # We didn't store it. We need to re-solve or store.
            # Let's modify solve_min_side to return the translation too, or just re-solve.
            # Re-solving is safer.
            
            # Actually, let's just return the L and theta, and in construct, we can assume centering is optimal?
            # No, we need the exact x_trans, y_trans.
            
            # Let's re-run the LP for the best theta to get the variables.
            # This is cheap.
            
            # For now, let's assume the LP solution is symmetric enough that x_trans, y_trans are small?
            # No, we must get it right.
            
            # I will create a helper to get the full solution.
            
            pass # Placeholder, will implement below
            
    # Implementing the full solution retrieval
    # I'll refactor slightly to return the translation.
    
    return None # Will be replaced

# Refactored solve to return translation
def solve_min_side_full(centers):
    base_verts = []
    for c in centers:
        verts = hex_vertices(0, 0, 0, 1.0)
        verts = verts + c
        base_verts.append(verts)
    all_verts = np.vstack(base_verts)
    
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5]
    ])
    
    def get_solution(theta_rad):
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        rotated_x = all_verts[:, 0] * cos_t - all_verts[:, 1] * sin_t
        rotated_y = all_verts[:, 0] * sin_t + all_verts[:, 1] * cos_t
        
        c = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        apothem_coeff = math.sqrt(3) / 2
        
        for k in range(3):
            nx = normals[k, 0]
            ny = normals[k, 1]
            
            dot_pos = rotated_x * nx + rotated_y * ny
            dot_neg = -rotated_x * nx - rotated_y * ny
            
            row_pos = np.tile(np.array([nx, ny, -apothem_coeff]), (len(all_verts), 1))
            b_pos = -dot_pos
            
            row_neg = np.tile(np.array([-nx, -ny, -apothem_coeff]), (len(all_verts), 1))
            b_neg = -dot_neg
            
            A_ub.append(row_pos)
            b_ub.append(b_pos)
            A_ub.append(row_neg)
            b_ub.append(b_neg)
            
        A_ub = np.vstack(A_ub)
        b_ub = np.concatenate(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.fun, res.x
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    best_L = float('inf')
    best_theta = 0.0
    best_x = 0.0
    best_y = 0.0
    
    thetas = np.linspace(0, math.pi/6, 61)
    for t in thetas:
        L, x_sol = get_solution(t)
        if L < best_L:
            best_L = L
            best_theta = t
            if x_sol is not None:
                best_x, best_y = x_sol[0], x_sol[1]
                
    # Refine
    if best_L < float('inf'):
        from scipy.optimize import minimize_scalar
        def obj(t):
            L, _ = get_solution(t)
            return L
        res = minimize_scalar(obj, bounds=(best_theta - 0.05, best_theta + 0.05), method='bounded')
        if res.success and res.fun < best_L:
            L, x_sol = get_solution(res.x)
            if x_sol is not None:
                best_L = L
                best_theta = res.x
                best_x, best_y = x_sol[0], x_sol[1]
                
    return best_L, best_theta, best_x, best_y

def construct():
    start_time = time.time()
    
    topologies = [
        get_topology_3x4(),
        get_topology_flower_plus()
    ]
    
    best_L = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for centers in topologies:
        if time.time() - start_time > 110:
            break
            
        L, theta, x_trans, y_trans = solve_min_side_full(centers)
        
        if L < best_L:
            best_L = L
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            inner = []
            for c in centers:
                cx = c[0]
                cy = c[1]
                # Rotate center
                rx = cx * cos_t - cy * sin_t
                ry = cx * sin_t + cy * cos_t
                # Translate
                fx = rx + x_trans
                fy = ry + y_trans
                inner.append([fx, fy, math.degrees(theta)])
                
            best_inner = inner
            best_center = (0.0, 0.0) # Container is at origin
            best_angle = 0.0
            
    if best_inner is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
        
    return best_inner, best_center, best_L, best_angle

# EVOLVE-BLOCK-END
