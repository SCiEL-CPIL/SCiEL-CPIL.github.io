import numpy as np
import math
import time
import random
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons into a regular hexagon of minimum side length s.
    Uses a hybrid strategy:
    1. Discrete angular search over 30-degree increments (exploiting hexagon symmetry).
    2. For each angular configuration, use gradient-based optimization (L-BFGS-B) 
       to minimize the outer hexagon side length 's' subject to collision constraints 
       handled via a smooth penalty function.
    
    Returns (inner, outer, s) where:
    inner: (12, 3) array of (x, y, angle_deg)
    outer: (3,) array of (x, y, angle_deg)
    s: side length of outer hexagon
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    N = 12
    sqrt3 = math.sqrt(3)
    
    # Unit hexagon vertices (angle 0, side 1)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([
        [math.cos(math.radians(i * 60)), math.sin(math.radians(i * 60))] for i in range(6)
    ])
    
    # Outward normals for unit hexagon (angle 0)
    # Normals at 30, 90, 150, 210, 270, 330 degrees.
    base_normals = np.array([
        [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] for i in range(6)
    ])
    
    def rot_matrix(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]])

    def get_hex_vertices(cx, cy, side, angle_rad):
        R = rot_matrix(angle_rad)
        return cx + side * (unit_verts @ R.T)

    def get_hex_normals(angle_rad):
        R = rot_matrix(angle_rad)
        return base_normals @ R.T

    def compute_s(centers, angles_rad):
        """
        Compute the minimum side length s of an outer hexagon (centered at origin, angle 0)
        that contains all inner hexagons.
        """
        max_proj = 0.0
        # Outer hex normals are fixed at 30, 90, ... (angle 0)
        outer_normals = base_normals
        
        for i in range(N):
            verts = centers[i] + (unit_verts @ rot_matrix(angles_rad[i]).T)
            # Project vertices onto outer hex normals
            projs = verts @ outer_normals.T
            max_proj_hex = projs.max(axis=0)
            current_max = np.max(max_proj_hex)
            if current_max > max_proj:
                max_proj = current_max
        
        # Distance from center to edge is s * sqrt3 / 2
        # So max_proj <= s * sqrt3 / 2  => s >= max_proj * 2 / sqrt3
        return max_proj * 2.0 / sqrt3

    def collision_penalty(centers, angles_rad):
        """
        Compute a smooth penalty for collisions between inner hexagons.
        Uses SAT to find minimum separation distance (negative if overlapping).
        Returns sum of squared penetration depths for all pairs.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                c1, a1 = centers[i], angles_rad[i]
                c2, a2 = centers[j], angles_rad[j]
                
                verts1 = c1 + (unit_verts @ rot_matrix(a1).T)
                verts2 = c2 + (unit_verts @ rot_matrix(a2).T)
                
                normals1 = get_hex_normals(a1)
                normals2 = get_hex_normals(a2)
                
                axes = np.vstack([normals1, normals2])
                
                proj1 = verts1 @ axes.T
                proj2 = verts2 @ axes.T
                
                min1 = proj1.min(axis=0)
                max1 = proj1.max(axis=0)
                min2 = proj2.min(axis=0)
                max2 = proj2.max(axis=0)
                
                # Overlap on each axis
                overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
                
                # If all overlaps > 0, they are intersecting.
                # The "penetration depth" is not simply the min overlap, but for a smooth penalty,
                # we can sum the squares of the positive overlaps. This encourages separation on all axes.
                pos_overlaps = np.maximum(0, overlaps)
                penalty += np.sum(pos_overlaps ** 2)
        
        return penalty

    def total_cost(x, angles_rad):
        """
        Total cost to minimize. x is flattened (centers, s).
        We minimize s + lambda * collision_penalty.
        """
        centers = x[:2*N].reshape(N, 2)
        s = x[2*N]
        
        # s must be at least the geometric minimum to contain the hexagons
        s_min = compute_s(centers, angles_rad)
        if s < s_min:
            s = s_min
            
        col_pen = collision_penalty(centers, angles_rad)
        
        return s + 50.0 * col_pen

    def gradient(x, angles_rad):
        """
        Numerical gradient for L-BFGS-B.
        """
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = total_cost(x_plus, angles_rad)
            
            x_minus = x.copy()
            x_minus[i] -= eps
            f_minus = total_cost(x_minus, angles_rad)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def validate_config(centers, angles_rad, s, eps=1e-6):
        """
        Strict validation.
        """
        # 1. Collisions
        for i in range(N):
            for j in range(i + 1, N):
                c1, a1 = centers[i], angles_rad[i]
                c2, a2 = centers[j], angles_rad[j]
                
                verts1 = c1 + (unit_verts @ rot_matrix(a1).T)
                verts2 = c2 + (unit_verts @ rot_matrix(a2).T)
                
                normals1 = get_hex_normals(a1)
                normals2 = get_hex_normals(a2)
                
                axes = np.vstack([normals1, normals2])
                
                proj1 = verts1 @ axes.T
                proj2 = verts2 @ axes.T
                
                min1 = proj1.min(axis=0)
                max1 = proj1.max(axis=0)
                min2 = proj2.min(axis=0)
                max2 = proj2.max(axis=0)
                
                # Separated if there is an axis where intervals don't overlap
                # max1 < min2 - eps OR max2 < min1 - eps
                sep1 = max1 < min2 - eps
                sep2 = max2 < min1 - eps
                
                if not (np.any(sep1) or np.any(sep2)):
                    return False
        
        # 2. Containment
        outer_verts = get_hex_vertices(0, 0, s, 0.0)
        
        for i in range(N):
            verts = centers[i] + (unit_verts @ rot_matrix(angles_rad[i]).T)
            for v in verts:
                inside = True
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0] * vec[1] - edge[1] * vec[0]
                    if cross < -eps:
                        inside = False
                        break
                if not inside:
                    return False
        
        return True

    # Initial configuration: Hexagonal lattice (Pointy top for better lattice packing)
    def generate_initial(angle_deg=30.0):
        # For pointy top (30 deg), the "flat" sides are vertical? No.
        # Angle 0: Flat top? Vertices at 0, 60... Edge 0 is (1,0) to (0.5, 0.866). 
        # Normal is 30 deg.
        # If we rotate by 30 deg, vertices are at 30, 90... Edge 0 is (0.866, 0.5) to (0, 1).
        # This is "Pointy Top" (vertex at top).
        # Lattice spacing for pointy top:
        # Horizontal distance between centers in same row: sqrt(3)
        # Vertical distance between rows: 1.5
        # Offset for odd rows: sqrt(3)/2
        
        dx = sqrt3
        dy = 1.5
        init_c = []
        init_a = []
        
        # 12 hexagons. 
        # Rows: 4, 4, 4? Or 3, 4, 5? 
        # Let's try a 3-4-5 or 4-4-4 arrangement.
        # 4-4-4 is 12.
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2.0
                init_c.append([x, y])
                init_a.append(math.radians(angle_deg))
        
        centers = np.array(init_c)
        angles = np.array(init_a)
        
        # Center the configuration
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        
        return centers, angles

    best_centers = None
    best_angles = None
    best_s = float('inf')
    best_cost = float('inf')
    
    # Discrete angular search: 0, 30, 60, 90, 120, 150 degrees
    # Due to 6-fold symmetry, 0, 30, 60 are unique orientations relative to the outer hexagon (which is fixed at 0).
    # Actually, rotating inner hex by 60 deg is identical. So 0, 30, 60, 90, 120, 150.
    # But 30 and 150 are related by symmetry? 
    # Let's just try all 6: 0, 30, 60, 90, 120, 150.
    
    angle_options = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    
    for ang_deg in angle_options:
        if time.time() > deadline:
            break
            
        centers, angles = generate_initial(ang_deg)
        
        # Initial s estimate
        s_init = compute_s(centers, angles) * 1.1
        
        # Flatten variables: x = [cx1, cy1, ..., cx12, cy12, s]
        x0 = np.concatenate([centers.flatten(), [s_init]])
        
        # Bounds: Centers in [-10, 10], s in [sqrt(12), 20]
        bounds = [(-10, 10)] * (2*N) + [(math.sqrt(12), 20.0)]
        
        # Run L-BFGS-B
        try:
            res = minimize(
                total_cost, 
                x0, 
                args=(angles,), 
                method='L-BFGS-B', 
                jac=lambda x, a: gradient(x, a), 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
        except Exception:
            continue
        
        if res.success or res.fun < best_cost:
            x_opt = res.x
            centers_opt = x_opt[:2*N].reshape(N, 2)
            s_opt = x_opt[2*N]
            
            # Ensure s is at least the geometric minimum
            s_geo = compute_s(centers_opt, angles)
            if s_opt < s_geo:
                s_opt = s_geo
            
            # Validate
            if validate_config(centers_opt, angles, s_opt):
                if s_opt < best_s:
                    best_centers = centers_opt
                    best_angles = angles
                    best_s = s_opt
                    best_cost = res.fun

    # If no valid solution found via optimization, fallback to initial lattice
    if best_centers is None:
        centers, angles = generate_initial(30.0)
        s_curr = compute_s(centers, angles)
        if validate_config(centers, angles, s_curr):
            best_centers = centers
            best_angles = angles
            best_s = s_curr
        else:
            # Last resort: very large s
            best_centers = centers
            best_angles = angles
            best_s = compute_s(centers, angles) * 2

    inner_ret = np.column_stack((best_centers, best_angles * 180.0 / math.pi))
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s

if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"Best s: {s}")
