"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices (side length 1, centered at origin, angle 0)
    # Vertices are at angles 0, 60, 120, 180, 240, 300
    unit_hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in [0, 60, 120, 180, 240, 300]
    ])

    def get_hex_vertices(center_x, center_y, scale, angle_deg):
        """
        Returns vertices of a regular hexagon with given center, side length (scale), and rotation.
        Side length 'scale' means distance from center to vertex is 'scale'.
        """
        angle_rad = math.radians(angle_deg)
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        
        # Scale the unit vertices
        verts_scaled = unit_hex_verts * scale
        # Rotate
        verts_rot = verts_scaled @ R.T
        # Translate
        verts_final = verts_rot + np.array([center_x, center_y])
        return verts_final

    def point_in_convex_polygon(point, polygon):
        """
        Check if a point is inside or on the boundary of a convex polygon.
        polygon: list of (x, y) tuples in counter-clockwise order.
        """
        n = len(polygon)
        if n < 3:
            return False
        
        sign = 0
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
            if cross > 1e-9:
                if sign == 0:
                    sign = 1
                elif sign == -1:
                    return False
            elif cross < -1e-9:
                if sign == 0:
                    sign = -1
                elif sign == 1:
                    return False
        return True

    def polygons_overlap_sat(poly1, poly2, eps=1e-7):
        """
        Separating Axis Theorem to check if two convex polygons overlap.
        Returns True if they overlap (including touching), False if separated.
        """
        def get_axes(polygon):
            axes = []
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                edge = (p2[0] - p1[0], p2[1] - p1[1])
                # Normal is perpendicular to edge
                length = math.sqrt(edge[0]**2 + edge[1]**2)
                if length < 1e-9:
                    continue
                nx = -edge[1] / length
                ny = edge[0] / length
                axes.append((nx, ny))
            return axes

        axes = get_axes(poly1) + get_axes(poly2)
        
        for ax, ay in axes:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            
            for x, y in poly1:
                proj = x * ax + y * ay
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
                
            for x, y in poly2:
                proj = x * ax + y * ay
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
                
            if max1 + eps < min2 or max2 + eps < min1:
                return False
                
        return True

    def check_validity(inner_positions, inner_angles, inner_scale, outer_center, outer_side, outer_angle):
        """
        Check if the packing is valid:
        1. Inner hexagons do not overlap each other.
        2. All inner hexagons are inside the outer hexagon.
        """
        # 1. Check overlaps between inner hexagons
        hexes = []
        for i in range(len(inner_positions)):
            x, y = inner_positions[i]
            ang = inner_angles[i]
            verts = get_hex_vertices(x, y, inner_scale, ang)
            hexes.append(verts)
            
        for i in range(len(hexes)):
            for j in range(i + 1, len(hexes)):
                if polygons_overlap_sat(hexes[i], hexes[j]):
                    return False
                    
        # 2. Check if inner hexagons are inside outer hexagon
        outer_verts = get_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        
        for i in range(len(hexes)):
            # Check if all vertices of inner hex i are inside outer hex
            for v in hexes[i]:
                if not point_in_convex_polygon(v, outer_verts):
                    return False
        return True

    def compute_min_outer_side(positions, angles, scale, outer_angle=0.0, outer_center=(0.0, 0.0)):
        """
        Calculate the minimum side length S of a regular hexagon with fixed center and angle
        that contains all given unit hexagons (scaled by 'scale').
        """
        max_proj = 0.0
        theta_rad = math.radians(outer_angle)
        
        for k in range(6):
            # Normals to the sides of the outer hexagon
            # The sides are at angles 30, 90, 150, 210, 270, 330 relative to the vertex directions?
            # Let's use the support function approach.
            # A regular hexagon centered at origin with 'flat' top (vertex at 90) has normals at 0, 60, 120...
            # Or if vertex at 0, normals are at 30, 90, 150...
            # The problem statement says "regular hexagon". The orientation is defined by 'angle'.
            # Let's assume the outer hexagon has vertices at angle + k*60.
            # Then the sides are perpendicular to directions angle + 30 + k*60.
            
            normal_angle = math.radians(30 + k * 60) + theta_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            h_max = 0.0
            
            for i in range(len(positions)):
                cx, cy = positions[i]
                ang_i = angles[i]
                
                # Support function of the inner hexagon in direction (nx, ny)
                # Vertices of inner hex i
                verts_i = get_hex_vertices(cx, cy, scale, ang_i)
                
                # Project all vertices onto the normal
                for vx, vy in verts_i:
                    # Projection relative to outer center
                    # We want the max extent in direction (nx, ny)
                    # The outer hexagon boundary in direction (nx, ny) is at distance S * (1/sqrt(3))?
                    # For a hexagon with side S, the apothem (distance to side) is S * sqrt(3) / 2.
                    # The projection of the hexagon onto the normal of a side is exactly the apothem + S/2?
                    # No. The support function h(n) of a regular hexagon with side S centered at origin:
                    # If n is normal to a side, h(n) = S * sqrt(3) / 2.
                    # If n is towards a vertex, h(n) = S.
                    # Here, nx, ny is the normal to a side of the OUTER hexagon.
                    # So we need the max projection of the inner hexagons onto this normal.
                    # And this max projection must be <= Apothem_outer.
                    # Apothem_outer = S_outer * sqrt(3) / 2.
                    # So S_outer = 2 * max_proj / sqrt(3).
                    
                    proj = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                    if proj > h_max:
                        h_max = proj
                        
            if h_max > max_proj:
                max_proj = h_max
                
        return max_proj * 2.0 / sqrt3

    def build_ring12():
        """
        Central hexagon + 6 neighbors + 5 more to make 12?
        Standard honeycomb packing.
        Center (0,0).
        Ring 1: 6 hexagons at distance 2.
        Ring 2: Pick 5 more from the next shell?
        Let's just use the first 12 positions from a honeycomb grid.
        """
        positions = [(0, 0)]
        angles = [0.0]
        
        # Ring 1
        for k in range(6):
            ang = math.radians(k * 60)
            x = 2.0 * math.cos(ang)
            y = 2.0 * math.sin(ang)
            positions.append((x, y))
            angles.append(0.0)
            
        # Ring 2: Positions at distance ~ 3.46 or 4.0
        # Let's add 5 more from the next layer.
        # The next layer has 12 hexagons. We pick 5 that are compact.
        # Directions: 30, 90, 150, 210, 270 (offsets)
        for k in range(5):
            ang = math.radians(30 + k * 60)
            # Distance to center for ring 2 hexagons in honeycomb
            # They are at distance 2*sqrt(3) from center? No.
            # The hexagons in ring 2 are adjacent to ring 1.
            # Center of ring 1 hex: (2,0).
            # Adjacent in "up" direction: (2, 0) + (1, sqrt(3)) = (3, sqrt(3)). Dist = sqrt(9+3)=sqrt(12).
            # Let's use specific coordinates for a compact 12-hex cluster.
            # 1 in center, 6 around, 5 around those.
            # This is getting complicated. Let's use a simpler "barrel" shape.
            pass
            
        # Fallback: Just use a 3x4 grid rotated?
        # Let's stick to the honeycomb 12:
        # 1 center, 6 ring, 5 from next ring.
        # The 5 from next ring can be placed at angles 0, 60, 120, 180, 240 in the second shell.
        # Distance to center for second shell hexagons (center to center) is 4? No.
        # In hex grid, if neighbor is dist 2, next neighbor is dist 4?
        # (0,0) -> (2,0) -> (4,0) is dist 4.
        # But (2,0) has neighbor at (3, sqrt(3)). Dist from origin: sqrt(9+3) = sqrt(12) = 3.46.
        
        # Let's construct explicitly:
        hex_centers = [
            (0, 0),
            (2, 0), (1, math.sqrt(3)), (-1, math.sqrt(3)),
            (-2, 0), (-1, -math.sqrt(3)), (1, -math.sqrt(3)),
            # Add 5 more to make 12.
            (4, 0), (3, math.sqrt(3)), (0, 2*math.sqrt(3)), (-3, math.sqrt(3)), (-4, 0)
        ]
        
        # Center the cluster
        cx = sum(p[0] for p in hex_centers) / len(hex_centers)
        cy = sum(p[1] for p in hex_centers) / len(hex_centers)
        
        centered = [(x - cx, y - cy) for x, y in hex_centers]
        return centered, [0.0] * len(centered)

    def build_hex6_plus6():
        """
        Two rings of 6 hexagons, separated by a gap or close together.
        Ring A centered at (-1.5, 0), Ring B centered at (1.5, 0)?
        If they touch, distance between centers is 2*sqrt(3) approx?
        Let's just place two hexagonal rings side by side.
        """
        positions = []
        angles = []
        
        # Ring 1 centered at (-2, 0)
        cx1, cy1 = -2.0, 0.0
        for k in range(6):
            ang = math.radians(k * 60)
            x = cx1 + 2.0 * math.cos(ang)
            y = cy1 + 2.0 * math.sin(ang)
            positions.append((x, y))
            angles.append(0.0)
            
        # Ring 2 centered at (2, 0)
        cx2, cy2 = 2.0, 0.0
        for k in range(6):
            ang = math.radians(k * 60)
            x = cx2 + 2.0 * math.cos(ang)
            y = cy2 + 2.0 * math.sin(ang)
            positions.append((x, y))
            angles.append(0.0)
            
        # Center the whole thing
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        
        centered = [(x - cx, y - cy) for x, y in positions]
        return centered, [0.0] * len(centered)

    def build_rect3x4():
        """
        3 rows, 4 columns grid.
        """
        positions = []
        angles = []
        # Hex grid coordinates for a 3x4 block
        # Row 0: 4 hexes
        # Row 1: 3 hexes (offset)
        # Row 2: 4 hexes
        # Total 11? Need 12.
        # Let's do 4x3 grid.
        # Row 0: 3
        # Row 1: 4
        # Row 2: 3
        # Row 3: 4? No, that's 14.
        
        # Simple 2x6 grid?
        # Row 0: 6
        # Row 1: 6
        for row in range(2):
            for col in range(6):
                x = col * 2.0
                y = row * math.sqrt(3)
                if row % 2 == 1:
                    x += 1.0
                positions.append((x, y))
                angles.append(0.0)
                
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        
        centered = [(x - cx, y - cy) for x, y in positions]
        return centered, [0.0] * len(centered)

    def optimize_topology(base_positions, base_angles, time_limit):
        """
        Optimize the rigid body parameters (cx, cy, theta, scale) for a given topology.
        """
        best_score = float('inf')
        best_params = None
        base_positions_arr = np.array(base_positions)
        
        def objective(params):
            cx, cy, theta_deg, scale = params
            
            # Transform positions
            theta_rad = math.radians(theta_deg)
            c = math.cos(theta_rad)
            s = math.sin(theta_rad)
            
            x_rot = base_positions_arr[:, 0] * scale * c - base_positions_arr[:, 1] * scale * s
            y_rot = base_positions_arr[:, 0] * scale * s + base_positions_arr[:, 1] * scale * c
            
            x_final = x_rot + cx
            y_final = y_rot + cy
            
            pos_final = list(zip(x_final, y_final))
            angles_final = base_angles # Inner hexagons keep their relative rotation to the cluster?
            # Wait, if we rotate the cluster, the inner hexagons rotate with it.
            # The angle of the inner hexagon relative to the global frame is theta_deg + base_angle.
            # But for the overlap check and containment, we just need the absolute orientation.
            # Let's pass absolute angles.
            angles_abs = [a + theta_deg for a in base_angles]
            
            # Calculate min side length assuming outer hex is at (0,0) angle 0
            side = compute_min_outer_side(pos_final, angles_abs, scale, 0.0, (0.0, 0.0))
            return side

        # Grid search
        rot_steps = np.linspace(0, 60, 12)
        scale_steps = np.linspace(0.8, 1.2, 5)
        
        for theta in rot_steps:
            for scale in scale_steps:
                def obj_center(params):
                    return objective([params[0], params[1], theta, scale])
                
                # Optimize translation
                res = minimize(obj_center, [0.0, 0.0], method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_params = [res.x[0], res.x[1], theta, scale]
                    
                if time.time() > time_limit:
                    return best_score, best_params

        # Fine tune
        if best_params is not None and time.time() < time_limit - 5:
            def obj_full(params):
                return objective(params)
            
            res = minimize(obj_full, best_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        return best_score, best_params

    # Generate topologies
    topologies = [
        build_ring12(),
        build_hex6_plus6(),
        build_rect3x4()
    ]
    
    best_global_score = float('inf')
    best_global_params = None
    best_global_topology = None
    
    time_per_topo = 40.0
    
    for pos, ang in topologies:
        if time.time() > deadline - 10:
            break
            
        current_time_limit = min(deadline - 5, time.time() + time_per_topo)
        score, params = optimize_topology(pos, ang, current_time_limit)
        
        if score < best_global_score:
            best_global_score = score
            best_global_params = params
            best_global_topology = (pos, ang)
            
    if best_global_params is None:
        # Fallback to a simple valid configuration
        # Just return a large box
        return [], (0.0, 0.0), 100.0, 0.0

    cx, cy, theta_deg, scale = best_global_params
    base_pos, base_ang = best_global_topology
    
    # Final construction
    theta_rad = math.radians(theta_deg)
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    
    final_positions = []
    final_angles = []
    
    for i in range(len(base_pos)):
        bx, by = base_pos[i]
        b_ang = base_ang[i]
        
        x_rot = bx * scale * c - by * scale * s
        y_rot = bx * scale * s + by * scale * c
        
        x_final = x_rot + cx
        y_final = y_rot + cy
        ang_final = b_ang + theta_deg
        
        final_positions.append((x_final, y_final))
        final_angles.append(ang_final)
        
    # Calculate final outer side length
    final_side = compute_min_outer_side(final_positions, final_angles, scale, 0.0, (0.0, 0.0))
    
    # Validate
    if not check_validity(final_positions, final_angles, scale, (0.0, 0.0), final_side, 0.0):
        # If invalid, increase side length until valid (should be rare with SAT)
        while not check_validity(final_positions, final_angles, scale, (0.0, 0.0), final_side, 0.0):
            final_side *= 1.01
            if final_side > 100:
                break
                
    inner = []
    for i in range(len(final_positions)):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
        
    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
