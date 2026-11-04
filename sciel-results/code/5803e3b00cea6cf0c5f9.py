import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    sqrt3 = np.sqrt(3.0)
    
    # Initial layout: 3 rows x 4 columns hexagonal packing
    # Rows at y = -sqrt3, 0, sqrt3
    # Cols at x = -3, -1, 1, 3
    inner = []
    for row in range(3):
        y = (row - 1) * sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0
            inner.append([x, y, 0.0])
    
    inner = np.array(inner, dtype=float)
    
    def get_inner_vertices(inner_positions, angles_rad):
        """Get all vertices of all inner hexagons."""
        all_verts = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            theta = angles_rad[i]
            # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300
            for k in range(6):
                phi = theta + k * np.pi / 3.0
                vx = cx + np.cos(phi)
                vy = cy + np.sin(phi)
                all_verts.append([vx, vy])
        return np.array(all_verts)
    
    def get_container_vertices(center, side, angle_deg):
        """Get vertices of the outer container hexagon."""
        cx, cy = center
        alpha = np.radians(angle_deg)
        verts = []
        for k in range(6):
            phi = alpha + k * np.pi / 3.0
            vx = cx + side * np.cos(phi)
            vy = cy + side * np.sin(phi)
            verts.append([vx, vy])
        return np.array(verts)
    
    def is_inside_hexagon(point, center, side, angle_deg):
        """Check if a point is inside the container hexagon."""
        cx, cy = center
        alpha = np.radians(angle_deg)
        px, py = point
        # Transform to container's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -alpha
        rx = dx * np.cos(-alpha) - dy * np.sin(-alpha)
        ry = dx * np.sin(-alpha) + dy * np.cos(-alpha)
        
        # Container is regular hexagon centered at origin with side 'side'
        # In local frame, vertices at (side*cos(k*pi/3), side*sin(k*pi/3))
        # The hexagon can be described as: for each of 6 edges, the point must be on the correct side
        
        # Edge normals in local frame
        # Vertices: (s,0), (s/2, s*sqrt3/2), (-s/2, s*sqrt3/2), (-s,0), (-s/2,-s*sqrt3/2), (s/2,-s*sqrt3/2)
        # Edges connect consecutive vertices. The inward normal for each edge:
        # Edge from (s,0) to (s/2, s*sqrt3/2): direction (-s/2, s*sqrt3/2)
        #   Normal (inward): (s*sqrt3/2, s/2) normalized -> (sqrt3/2, 1/2) -> angle 30 deg
        # Actually, let's use the half-plane method.
        # For a regular hexagon centered at origin with circumradius = side,
        # the distance from center to each edge (apothem) = side * sqrt3 / 2
        # The 6 edge normals are at angles 0, 60, 120, 180, 240, 300 degrees? No.
        # Vertices at 0,60,120,180,240,300.
        # Edge between vertex at 0 and vertex at 60: midpoint at 30 deg, distance = apothem.
        # The outward normal is at 30 degrees.
        # So the 6 outward normals are at 30, 90, 150, 210, 270, 330 degrees.
        
        apothem = side * sqrt3 / 2.0
        for k in range(6):
            normal_angle = np.radians(30 + k * 60)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            # The constraint: rx*nx + ry*ny <= apothem
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True
    
    def objective(params):
        """Objective: minimize the container side length needed to contain all inner hexagons."""
        # params: [cx, cy, angle_deg, scale_x, scale_y, shear, rot_grid]
        # We'll optimize the grid transformation and container orientation
        cx, cy, angle_deg, scale_x, scale_y, shear, rot_grid = params
        
        # Transform inner positions
        base = inner.copy()
        # Apply grid rotation
        theta_g = np.radians(rot_grid)
        R = np.array([[np.cos(theta_g), -np.sin(theta_g)],
                      [np.sin(theta_g), np.cos(theta_g)]])
        base[:, :2] = base[:, :2] @ R.T
        # Apply scaling and shear
        base[:, 0] *= scale_x
        base[:, 1] *= scale_y
        base[:, 0] += shear * base[:, 1]
        
        # For each inner hexagon, we need to find the minimal container that fits
        # But we also allow inner hexagons to rotate. Let's fix inner rotations to 0 for simplicity
        # and just find the minimal container.
        
        all_verts = get_inner_vertices(base, np.zeros(12))
        
        # Find minimal container side that contains all vertices
        # Binary search or direct computation
        # For a given container orientation, the minimal side is:
        # For each vertex, compute the required side length, take the max.
        
        # Required side for a point (px,py) relative to container center (cx,cy) with orientation angle_deg:
        # Transform point to container frame
        # The constraint is: for each of 6 half-planes, |projection| <= apothem = s*sqrt3/2
        # So s >= max_over_normals (|projection| / (sqrt3/2))
        
        alpha = np.radians(angle_deg)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        
        max_ratio = 0.0
        for vx, vy in all_verts:
            dx = vx - cx
            dy = vy - cy
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            for k in range(6):
                normal_angle = np.radians(30 + k * 60)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                proj = rx * nx + ry * ny
                if proj > max_ratio:
                    max_ratio = proj
        
        side = max_ratio * 2.0 / sqrt3
        return side
    
    # Optimize: center, angle, grid rotation, scale, shear
    # Start with reasonable initial values
    x0 = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    
    best_side = 5.0
    best_params = x0.copy()
    
    # Try multiple starting angles for the grid rotation
    for init_rot in [0, 15, 30, 45, 60, -15, -30, -45]:
        x0_try = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, float(init_rot)]
        try:
            res = minimize(objective, x0_try, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except:
            pass
    
    # Extract best parameters
    cx, cy, angle_deg, scale_x, scale_y, shear, rot_grid = best_params
    
    # Apply transformation to inner positions
    base = inner.copy()
    theta_g = np.radians(rot_grid)
    R = np.array([[np.cos(theta_g), -np.sin(theta_g)],
                  [np.sin(theta_g), np.cos(theta_g)]])
    base[:, :2] = base[:, :2] @ R.T
    base[:, 0] *= scale_x
    base[:, 1] *= scale_y
    base[:, 0] += shear * base[:, 1]
    
    # Convert to list
    inner_list = []
    for i in range(12):
        inner_list.append([base[i, 0], base[i, 1], 0.0])
    
    return inner_list, (float(cx), float(cy)), float(best_side), float(angle_deg)
