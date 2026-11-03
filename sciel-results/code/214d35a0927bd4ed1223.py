"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

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

def polygons_overlap_sat(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            # Edge normal
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If there's a gap, no overlap
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (list of [cx, cy, angle_deg]) and outer hexagon params,
    find the minimum outer side length such that all inner hexagons fit inside.
    """
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    
    max_scale = 0.0
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = hex_vertices(cx, cy, ang)
        for vx, vy in verts:
            # Vector from outer center to this vertex
            dx, dy = vx - ox, vy - oy
            required_s = 0.0
            for k in range(6):
                # Direction from center to edge k midpoint
                edge_mid_angle = oa + k * math.pi / 3 + math.pi / 6
                # Inward normal: from edge toward center, opposite to edge_mid_angle
                inward_nx = -math.cos(edge_mid_angle)
                inward_ny = -math.sin(edge_mid_angle)
                dot_val = dx * inward_nx + dy * inward_ny
                # Constraint: dot_val <= s * sqrt(3)/2
                # So s >= dot_val / (sqrt(3)/2) if dot_val > 0
                if dot_val > 0:
                    s_req = dot_val / (math.sqrt(3) / 2)
                    required_s = max(required_s, s_req)
            max_scale = max(max_scale, required_s)
    
    return max_scale

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    sqrt3 = math.sqrt(3)
    
    # Initial setup: Random or Grid
    # Let's start with a grid to give the physics a head start
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    velocities = np.zeros((N, 2))
    
    # 4x3 grid initialization
    idx = 0
    for row in range(4):
        for col in range(3):
            cx = col * 1.5
            cy = row * sqrt3
            if row % 2 == 1:
                cx += 0.75
            positions[idx, 0] = cx - 1.5
            positions[idx, 1] = cy - 1.5 * sqrt3
            angles[idx] = 0.0
            idx += 1

    outer_center = np.array([0.0, 0.0])
    outer_angle_deg = 0.0
    oa_rad = math.radians(outer_angle_deg)
    
    # Physics parameters
    dt = 0.01
    repulsion_strength = 50.0
    damping = 0.95
    container_shrink_rate = 0.005
    min_side = 3.0 # Theoretical min is around 4-5, start larger
    current_side = 6.0
    
    # Pre-calculate outer hexagon normals (inward)
    # Edge k midpoint angle: oa + k*pi/3 + pi/6
    # Inward normal: -(cos(mid_angle), sin(mid_angle))
    outer_normals = []
    for k in range(6):
        mid_angle = oa_rad + k * math.pi / 3 + math.pi / 6
        outer_normals.append(np.array([-math.cos(mid_angle), -math.sin(mid_angle)]))
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = float('inf')
    
    steps = 0
    max_steps = 20000
    
    while steps < max_steps:
        if time.time() > deadline:
            break
            
        # 1. Calculate Forces
        forces = np.zeros((N, 2))
        torques = np.zeros(N)
        
        # Update container size
        # Shrink slowly
        current_side = max(min_side, current_side - container_shrink_rate)
        
        # 2. Inter-particle repulsion
        for i in range(N):
            v1 = hex_vertices(positions[i,0], positions[i,1], angles[i])
            for j in range(i+1, N):
                v2 = hex_vertices(positions[j,0], positions[j,1], angles[j])
                
                if polygons_overlap_sat(v1, v2, tol=1e-8):
                    # Simple repulsion: push apart along center vector
                    # More accurate would be SAT axis, but center vector is faster and usually sufficient for hexagons
                    center_vec = positions[j] - positions[i]
                    dist = np.linalg.norm(center_vec)
                    if dist < 1e-6:
                        # Random push if exactly same position
                        center_vec = np.array([np.random.randn(), np.random.randn()])
                        dist = np.linalg.norm(center_vec)
                    
                    normal = center_vec / dist
                    # Force magnitude proportional to overlap depth (approximated by 1 if overlapping)
                    # We use a constant strong force to separate
                    f = repulsion_strength * normal
                    
                    forces[i] -= f
                    forces[j] += f
                    
                    # Simple torque: try to rotate to reduce overlap
                    # This is heuristic. If they overlap, rotating one might help.
                    # We can estimate torque by cross product of contact point and force, but we don't have exact contact point.
                    # Let's apply a small random torque or none. 
                    # Actually, for hexagons, rotation is important.
                    # Let's add a small torque to align them if they are "stuck"
                    # A better heuristic: if overlap persists, add torque to both to separate.
                    # Here we just rely on translation first. If stuck, rotation helps.
                    # We can add a small torque based on the angle difference to prefer parallel alignment?
                    # Or just random small torque to break symmetry.
                    pass 

        # 3. Container Constraint
        for i in range(N):
            v_i = hex_vertices(positions[i,0], positions[i,1], angles[i])
            for k in range(6):
                n_k = outer_normals[k]
                # Distance of each vertex from the boundary
                # Boundary is at distance (side * sqrt(3)/2) from center along -n_k (outward)
                # Inward normal n_k points to center.
                # Point p is inside if dot(p - center, n_k) <= side * sqrt(3)/2
                # Wait, if n_k is inward, then dot(p, n_k) is positive for points on the "inside" side of the edge?
                # Let's check: Center is (0,0). Edge is at distance d from center.
                # Inward normal n points from edge to center.
                # Vector from center to edge is -n * d.
                # For a point p inside, dot(p, n) should be less than d?
                # If p is at center, dot(0, n) = 0 < d. Correct.
                # If p is at edge, p = -n * d. dot(-n*d, n) = -d.
                # So the condition is dot(p, n) >= -d? No.
                # Let's redefine: Outward normal o_k.
                # Condition: dot(p, o_k) <= d.
                # My outer_normals are inward. So o_k = -outer_normals[k].
                # Condition: dot(p, -n_k) <= d  =>  -dot(p, n_k) <= d  =>  dot(p, n_k) >= -d.
                
                d = current_side * sqrt3 / 2.0
                n_k = outer_normals[k]
                o_k = -n_k
                
                # Check max projection of vertices on outward normal
                proj = v_i @ o_k
                max_proj = np.max(proj)
                
                if max_proj > d:
                    # Violation
                    penetration = max_proj - d
                    # Force is along inward normal (n_k)
                    f = repulsion_strength * 2.0 * penetration * n_k
                    forces[i] += f
                    
                    # Torque: push the specific vertex that is violating back in
                    # Find which vertex is violating
                    idx_v = np.argmax(proj)
                    p_v = v_i[idx_v]
                    r_vec = p_v - positions[i]
                    # Torque = r x F (in 2D, scalar z-component)
                    # F is applied at vertex p_v in direction n_k
                    # torque = r_vec[0]*F[1] - r_vec[1]*F[0]
                    torque_val = r_vec[0]*f[1] - r_vec[1]*f[0]
                    torques[i] += torque_val

        # 4. Integration (Velocity Verlet simplified / Euler with damping)
        # Update velocities
        velocities += forces * dt
        velocities *= damping
        angles += torques * dt * 0.1 # Scale down torque effect
        
        # Update positions
        positions += velocities * dt
        
        # Center of mass correction to keep cluster centered
        com = np.mean(positions, axis=0)
        positions -= com
        
        # Record best valid state
        # Check if all inside current container
        valid = True
        for i in range(N):
            v_i = hex_vertices(positions[i,0], positions[i,1], angles[i])
            for k in range(6):
                o_k = -outer_normals[k]
                d = current_side * sqrt3 / 2.0
                if np.max(v_i @ o_k) > d + 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check overlaps for validity
            no_overlap = True
            for i in range(N):
                v1 = hex_vertices(positions[i,0], positions[i,1], angles[i])
                for j in range(i+1, N):
                    v2 = hex_vertices(positions[j,0], positions[j,1], angles[j])
                    if polygons_overlap_sat(v1, v2, tol=1e-8):
                        no_overlap = False
                        break
                if not no_overlap:
                    break
            
            if no_overlap:
                # This is a valid configuration for current_side
                # We want to minimize the final side length.
                # The "score" is the side length required to fit this configuration.
                # Since we are simulating inside a shrinking box, the current_side is an upper bound.
                # We should calculate the exact minimum side for this configuration to be precise.
                inner_hexes = [[positions[i,0], positions[i,1], math.degrees(angles[i])] for i in range(N)]
                exact_side = compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg)
                
                if exact_side < best_side:
                    best_side = exact_side
                    best_positions = positions.copy()
                    best_angles = angles.copy()
        
        steps += 1
        
        # Optional: If stuck, add small noise
        if steps % 1000 == 0:
            velocities += np.random.randn(N, 2) * 0.01
            angles += np.random.randn(N) * 0.01

    # Final Result
    if best_side == float('inf'):
        # Fallback to initial grid if simulation failed
        best_positions = positions
        best_angles = angles
        inner_hexes = [[positions[i,0], positions[i,1], math.degrees(angles[i])] for i in range(N)]
        best_side = compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg)
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i,0], best_positions[i,1], math.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_side, outer_angle_deg

# EVOLVE-BLOCK-END
