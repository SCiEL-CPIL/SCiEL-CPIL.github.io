import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
SQRT3 = math.sqrt(3)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute unit hexagon vertices (flat-top orientation, angle=0)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in [0, 60, 120, 180, 240, 300]
    ])

    def get_hex_verts(cx, cy, angle_deg):
        """Get vertices of a unit hexagon centered at (cx, cy) with rotation."""
        theta = math.radians(angle_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        return hex_verts @ R.T + np.array([cx, cy])

    def hexagon_in_hexagon_check(pts, center, side, angle_deg):
        """Check if all inner hexagons are inside the outer hexagon."""
        outer_verts = get_hex_verts(center[0], center[1], angle_deg) * side
        for p in pts:
            # Check if point is inside outer hexagon using cross products
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % 6]
                edge = v2 - v1
                to_point = p - v1
                cross = edge[0] * to_point[1] - edge[1] * to_point[0]
                if cross < -1e-9:
                    inside = False
                    break
            if not inside:
                return False
        return True

    def overlap_check(pts, angles):
        """Check if any two hexagons overlap using separating axis theorem."""
        n = len(pts)
        # For each pair, check separation along 12 axes (6 from each hex)
        for i in range(n):
            for j in range(i + 1, n):
                # Get vertices
                vi = get_hex_verts(pts[i][0], pts[i][1], angles[i])
                vj = get_hex_verts(pts[j][0], pts[j][1], angles[j])
                
                # Check all edges of both hexagons as potential separating axes
                separated = False
                for v in [vi, vj]:
                    for k in range(6):
                        edge = v[(k + 1) % 6] - v[k]
                        # Normal to edge
                        nx, ny = -edge[1], edge[0]
                        norm = math.sqrt(nx**2 + ny**2)
                        if norm < 1e-12:
                            continue
                        nx, ny = nx / norm, ny / norm
                        
                        # Project both hexagons
                        proj_i = vi @ np.array([nx, ny])
                        proj_j = vj @ np.array([nx, ny])
                        
                        min_i, max_i = np.min(proj_i), np.max(proj_i)
                        min_j, max_j = np.min(proj_j), np.max(proj_j)
                        
                        if max_i < min_j + 1e-9 or max_j < min_i + 1e-9:
                            separated = True
                            break
                    if separated:
                        break
                if not separated:
                    return False
        return True

    def compute_outer_side(pts, angles, center, angle_deg):
        """Compute minimum outer hexagon side that contains all inner hexagons."""
        # For each of the 6 normals of the outer hexagon, find max projection
        theta = math.radians(angle_deg)
        normals = np.array([
            [math.cos(theta + math.radians(30 + 60*k)), 
             math.sin(theta + math.radians(30 + 60*k))]
            for k in range(6)
        ])
        
        max_proj = 0.0
        for p in pts:
            for a in angles:
                verts = get_hex_verts(p[0], p[1], a)
                for n in normals:
                    proj = (verts - np.array(center)) @ n
                    max_proj = max(max_proj, np.max(np.abs(proj)))
        
        # D = max distance from center to any vertex along normal
        # Side = (D + apothem) / apothem
        return (max_proj + APOTHEM) / APOTHEM

    def force_directed_optimize(pts_init, angles_init, center_init, angle_deg_init, max_iters=5000):
        """Optimize using force-directed method with soft constraints."""
        pts = [list(p) for p in pts_init]
        angles = list(angles_init)
        center = list(center_init)
        angle_deg = angle_deg_init
        
        best_S = float('inf')
        best_state = None
        
        # Parameters
        repulsion_strength = 5.0
        containment_strength = 10.0
        lr = 0.05
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            forces = [[0.0, 0.0] for _ in range(N)]
            angle_forces = [0.0] * N
            center_force = [0.0, 0.0]
            angle_deg_force = 0.0
            
            # Compute current S
            current_S = compute_outer_side(pts, angles, center, angle_deg)
            
            if current_S < best_S and overlap_check(pts, angles):
                best_S = current_S
                best_state = ([list(p) for p in pts], list(angles), list(center), angle_deg)
            
            # Repulsion forces between hexagons
            for i in range(N):
                for j in range(i + 1, N):
                    vi = get_hex_verts(pts[i][0], pts[i][1], angles[i])
                    vj = get_hex_verts(pts[j][0], pts[j][1], angles[j])
                    
                    # Find minimum distance between the two hexagons
                    min_dist = float('inf')
                    for a in vi:
                        for b in vj:
                            d = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
                            min_dist = min(min_dist, d)
                    
                    # If overlapping or close, apply repulsion
                    if min_dist < SQRT3:
                        strength = repulsion_strength * (SQRT3 - min_dist)
                        dx = pts[j][0] - pts[i][0]
                        dy = pts[j][1] - pts[i][1]
                        dist = math.sqrt(dx**2 + dy**2)
                        if dist > 1e-9:
                            fx = strength * dx / dist
                            fy = strength * dy / dist
                        else:
                            fx = strength * 0.01
                            fy = strength * 0.01
                        
                        forces[i][0] -= fx
                        forces[i][1] -= fy
                        forces[j][0] += fx
                        forces[j][1] += fy
            
            # Containment forces
            theta = math.radians(angle_deg)
            normals = np.array([
                [math.cos(theta + math.radians(30 + 60*k)), 
                 math.sin(theta + math.radians(30 + 60*k))]
                for k in range(6)
            ])
            
            for i in range(N):
                verts = get_hex_verts(pts[i][0], pts[i][1], angles[i])
                for k in range(6):
                    n = normals[k]
                    proj = (verts - np.array(center)) @ n
                    max_p = np.max(proj)
                    min_p = np.min(proj)
                    
                    # Target: |proj| <= D where D = (S-1)*apothem
                    # For now, use current S as target
                    D_target = (current_S - 1.0) * APOTHEM
                    
                    if max_p > D_target:
                        # Push back
                        f = containment_strength * (max_p - D_target)
                        forces[i][0] -= f * n[0]
                        forces[i][1] -= f * n[1]
                        center_force[0] += f * n[0] * 0.1
                        center_force[1] += f * n[1] * 0.1
                    elif min_p < -D_target:
                        f = containment_strength * (-D_target - min_p)
                        forces[i][0] += f * n[0]
                        forces[i][1] += f * n[1]
                        center_force[0] -= f * n[0] * 0.1
                        center_force[1] -= f * n[1] * 0.1
            
            # Update positions
            for i in range(N):
                pts[i][0] += lr * forces[i][0]
                pts[i][1] += lr * forces[i][1]
            
            center[0] += lr * 0.5 * center_force[0]
            center[1] += lr * 0.5 * center_force[1]
            
            # Decay learning rate
            lr *= cooling
            
            # Every 100 iterations, try to optimize angle
            if it % 100 == 99:
                # Try small perturbations of angle_deg
                best_local_S = current_S
                best_local_angle = angle_deg
                for da in [-10, -5, -2, 0, 2, 5, 10]:
                    test_angle = angle_deg + da
                    test_S = compute_outer_side(pts, angles, center, test_angle)
                    if test_S < best_local_S:
                        best_local_S = test_S
                        best_local_angle = test_angle
                angle_deg = best_local_angle
        
        return best_S, best_state

    # Generate diverse initial configurations
    def gen_random_config():
        pts = []
        angles = []
        # Start with a rough hexagonal arrangement
        base_positions = [
            (0, 0),
            (SQRT3, 0), (0, 1.5), (-SQRT3/2, 1.5), (-SQRT3, 0), (0, -1.5), (SQRT3/2, -1.5),
            (2*SQRT3, 0), (0, 3), (-SQRT3, 3), (SQRT3, -3)
        ]
        for x, y in base_positions:
            pts.append([x + np.random.uniform(-0.5, 0.5), y + np.random.uniform(-0.5, 0.5)])
            angles.append(float(np.random.uniform(0, 60)))
        
        # Center
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        center = [float(cx), float(cy)]
        angle_deg = float(np.random.uniform(0, 60))
        
        return pts, angles, center, angle_deg

    def gen_grid_config(row_pattern):
        """Generate a grid-based configuration."""
        pts = []
        angles = []
        y = 0.0
        for row in row_pattern:
            x_offset = (max(row_pattern) - row) * SQRT3 / 2
            for i in range(row):
                x = x_offset + i * SQRT3
                pts.append([float(x), float(y)])
                angles.append(0.0)
            y += 1.5
        
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        center = [float(cx), float(cy)]
        angle_deg = 0.0
        
        return pts, angles, center, angle_deg

    # Multiple initial configurations
    configs = []
    
    # Grid patterns
    for pattern in [[1, 5, 5], [3, 5, 3], [4, 4, 3], [5, 4, 2], [2, 5, 4], [3, 4, 4], [4, 3, 4], [5, 3, 3]]:
        configs.append(gen_grid_config(pattern))
    
    # Random perturbations of base config
    for _ in range(5):
        configs.append(gen_random_config())
    
    # Run force-directed optimization on each config
    best_global_S = float('inf')
    best_global_state = None
    
    for config in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts, angles, center, angle_deg = config
        S, state = force_directed_optimize(pts, angles, center, angle_deg, max_iters=3000)
        
        if S < best_global_S and state is not None:
            best_global_S = S
            best_global_state = state
    
    # Final polish: try to reduce S by optimizing center and angle
    if best_global_state is not None:
        pts, angles, center, angle_deg = best_global_state
        
        # Optimize center
        for _ in range(20):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            best_cx, best_cy = center[0], center[1]
            best_S = compute_outer_side(pts, angles, center, angle_deg)
            for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                    test_center = [center[0] + dx, center[1] + dy]
                    test_S = compute_outer_side(pts, angles, test_center, angle_deg)
                    if test_S < best_S:
                        best_S = test_S
                        best_cx, best_cy = test_center
            center = [best_cx, best_cy]
        
        # Optimize angle
        best_angle = angle_deg
        best_S = compute_outer_side(pts, angles, center, angle_deg)
        for da in np.linspace(-30, 30, 61):
            test_angle = angle_deg + da
            test_S = compute_outer_side(pts, angles, center, test_angle)
            if test_S < best_S:
                best_S = test_S
                best_angle = test_angle
        angle_deg = best_angle
        
        # Verify no overlap
        if not overlap_check(pts, angles):
            # Fallback to a known valid config
            pts, angles, center, angle_deg = gen_grid_config([3, 5, 3])
            best_global_S = compute_outer_side(pts, angles, center, angle_deg)
    
    if best_global_state is None:
        # Fallback
        pts, angles, center, angle_deg = gen_grid_config([3, 5, 3])
        best_global_S = compute_outer_side(pts, angles, center, angle_deg)
    
    # Construct output
    inner = []
    for i, p in enumerate(pts):
        inner.append([float(p[0]), float(p[1]), float(angles[i])])
    
    cx, cy = center
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, [float(cx), float(cy)], side, rotation
