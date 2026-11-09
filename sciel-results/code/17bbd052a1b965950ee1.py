import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    SQRT3 = math.sqrt(3)

    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / SQRT3

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Pre-define topologies. 
    # We use a tight "flower" core and various extensions.
    # Topology 1: Center + 6 neighbors + 4 more
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # Topology 2: 3-5-3 block
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # Topology 3: Another variation
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    
    topologies = [coords1, coords2, coords3]
    
    best_global_score = float('inf')
    best_global_params = None
    best_global_outer_angle = 0.0

    for topo in topologies:
        if time.time() > deadline - 5.0:
            break
            
        # Initialize rigid body parameters: x, y, rotation (radians)
        # Start at 0,0,0
        params0 = np.array([0.0, 0.0, 0.0])
        
        # Objective function for Rigid Body
        # p = [x, y, rot_rad]
        def objective(p):
            x, y, rot = p
            rot_deg = math.degrees(rot)
            
            hexagons = []
            hex_axes = []
            
            # Transform each hexagon in the topology
            for q, r in topo:
                bx, by = axial_to_cartesian(q, r, spacing=1.0)
                # Rotate base position by 'rot'
                cos_r = math.cos(rot)
                sin_r = math.sin(rot)
                cx = bx * cos_r - by * sin_r + x
                cy = bx * sin_r + by * cos_r + y
                
                # Note: The hexagon itself is also rotated by 'rot'
                a = rot_deg
                
                hex_verts = get_hex_vertices(cx, cy, a)
                hexagons.append(hex_verts)
                hex_axes.append(get_hex_axes(a))
            
            # Check overlaps
            # For rigid body, we can early exit if we detect overlap
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                        # Penalty for overlap
                        # We want to minimize this, so return large value
                        return 1000.0
            
            # Calculate outer side length
            # We optimize outer angle separately later, so fix at 0 for now?
            # No, the cluster's orientation relative to outer hexagon matters.
            # But since we are rotating the whole cluster, we can fix the outer hexagon angle to 0
            # and let the cluster rotation handle the relative orientation.
            side = calc_outer_side(hexagons, 0.0)
            
            return side

        # Run Nelder-Mead
        try:
            res = minimize(objective, params0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_global_score:
                # Verify validity
                x, y, rot = res.x
                rot_deg = math.degrees(rot)
                hexagons = []
                hex_axes = []
                for q, r in topo:
                    bx, by = axial_to_cartesian(q, r, spacing=1.0)
                    cos_r = math.cos(rot)
                    sin_r = math.sin(rot)
                    cx = bx * cos_r - by * sin_r + x
                    cy = bx * sin_r + by * cos_r + y
                    a = rot_deg
                    hex_verts = get_hex_vertices(cx, cy, a)
                    hexagons.append(hex_verts)
                    hex_axes.append(get_hex_axes(a))
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_global_score = res.fun
                    best_global_params = res.x
                    # Store the topology used
                    best_global_params = np.append(res.x, topo) # Hacky, better to store separately
                    # Let's store properly
                    best_global_params = res.x
                    best_global_topo = topo
                else:
                    # If invalid, try to repair or skip
                    pass
        except Exception as e:
            pass

    # If we found a valid solution, refine outer angle
    if best_global_params is not None:
        x, y, rot = best_global_params
        rot_deg = math.degrees(rot)
        
        hexagons = []
        hex_axes = []
        for q, r in best_global_topo:
            bx, by = axial_to_cartesian(q, r, spacing=1.0)
            cos_r = math.cos(rot)
            sin_r = math.sin(rot)
            cx = bx * cos_r - by * sin_r + x
            cy = bx * sin_r + by * cos_r + y
            a = rot_deg
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        # Center the configuration to (0,0)
        # The outer hexagon is centered at (0,0) in our calc_outer_side
        # But the cluster center might not be (0,0).
        # calc_outer_side assumes outer hex center is (0,0).
        # So we need to shift the cluster so its "center" is at (0,0)?
        # No, calc_outer_side calculates the minimum S such that all points are within S*sqrt(3)/2 from origin.
        # This effectively centers the outer hexagon at the origin.
        # The inner hexagons are at their absolute coordinates.
        # So the returned side length is correct for an outer hexagon centered at (0,0).
        
        # Optimize outer angle
        best_side = float('inf')
        best_oa = 0.0
        for oa_deg in range(0, 600):
            oa = oa_deg * 0.1
            side = calc_outer_side(hexagons, oa)
            if side < best_side:
                best_side = side
                best_oa = oa
        
        # Construct final inner list
        inner_list = []
        for q, r in best_global_topo:
            bx, by = axial_to_cartesian(q, r, spacing=1.0)
            cos_r = math.cos(rot)
            sin_r = math.sin(rot)
            cx = bx * cos_r - by * sin_r + x
            cy = bx * sin_r + by * cos_r + y
            a = rot_deg
            inner_list.append([cx, cy, a])
            
        return (inner_list, (0.0, 0.0), best_side, best_oa)
    else:
        # Fallback: Use topology 1 with no rotation/translation
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

# EVOLVE-BLOCK-END
