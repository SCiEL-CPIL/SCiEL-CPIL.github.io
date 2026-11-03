import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * math.sqrt(3) / 2.0

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            # Edge 1
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            # Edge 2
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True # Separating axis found, disjoint
                
        return False # No separating axis, overlapping

    def simulate(start_side, max_steps=2000, dt=0.005):
        """
        Run a physics simulation with a shrinking container.
        Returns the best valid configuration found.
        """
        # Initialize positions in a loose honeycomb pattern
        sqrt3 = math.sqrt(3)
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # We need to ensure the initial positions fit inside the start_side hexagon.
        # If start_side is too small, we might start with overlaps or outside.
        # Let's scale the initial positions to fit comfortably inside start_side * 0.9
        init_pos = np.array(lattice_points[:N])
        init_pos -= np.mean(init_pos, axis=0)
        
        # Check if initial config fits in start_side
        # If not, we might need to start larger and shrink faster, or just accept initial penalty
        # For robustness, let's ensure start_side is large enough for the initial pattern.
        # The honeycomb of 12 hexes roughly fits in a hex of side ~3.5
        if start_side < 4.0:
             # Scale down initial positions to fit
             current_side = calc_min_outer_side(init_pos, np.zeros(N))
             if current_side > start_side * 0.95:
                 scale = (start_side * 0.95) / current_side
                 init_pos *= scale

        positions = init_pos.copy()
        angles = np.zeros(N)
        velocities = np.zeros((N, 2))
        angular_velocities = np.zeros(N)
        
        # Physics parameters
        k_repel = 500.0      # Repulsion strength between hexes
        k_wall = 100.0       # Repulsion strength from walls
        damping = 0.98       # Velocity damping per step
        angular_damping = 0.95
        
        # Target side length to shrink to
        target_side = 3.0    # We know the answer is around 3.2-3.5, let's aim low
        
        # Current container side
        current_side = start_side
        
        # Shrink rate
        shrink_rate = (start_side - target_side) / max_steps
        
        best_valid_side = float('inf')
        best_positions = None
        best_angles = None
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Shrink container
            current_side = max(target_side, start_side - shrink_rate * step)
            
            # Compute forces
            forces = np.zeros((N, 2))
            torques = np.zeros(N)
            
            # Wall forces
            # Hexagon container defined by 6 half-planes: n_k . p <= d_k
            # For a regular hexagon centered at 0 with side s and rotation 0:
            # The distance from center to edge (apothem) is s * sqrt(3) / 2
            apothem = current_side * math.sqrt(3) / 2.0
            normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 390, 60)])
            
            for i in range(N):
                # Check all vertices of hex i against all walls
                for k in range(6):
                    n = normals[k]
                    # Distance of vertex v from wall line n.p = apothem
                    # If n.p > apothem, it's penetrating
                    verts = polys[i]
                    projs = verts @ n
                    max_proj = np.max(projs)
                    pen = max_proj - apothem
                    if pen > 0:
                        # Force is along normal n, pointing inwards (-n)
                        # Strength proportional to penetration
                        force = -k_wall * pen * n
                        forces[i] += force
                        # Add a small torque to align with wall? 
                        # For now, just linear force. Rotation is handled by hex-hex interactions mostly.
                        
            # Hex-hex repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    # SAT check for overlap
                    # We use the same axes as in check_disjoint_fast
                    # But we need the penetration depth and direction for force
                    
                    # Axes from i
                    axes = []
                    for edge_idx in range(6):
                        e1 = polys[i][(edge_idx+1)%6] - polys[i][edge_idx]
                        axes.append(np.array([-e1[1], e1[0]]))
                        e2 = polys[j][(edge_idx+1)%6] - polys[j][edge_idx]
                        axes.append(np.array([-e2[1], e2[0]]))
                        
                    min_pen = float('inf')
                    min_axis = None
                    
                    for axis in axes:
                        norm = np.linalg.norm(axis)
                        if norm < 1e-9: continue
                        axis = axis / norm
                        
                        proj_i = polys[i] @ axis
                        proj_j = polys[j] @ axis
                        
                        min_i = np.min(proj_i)
                        max_i = np.max(proj_i)
                        min_j = np.min(proj_j)
                        max_j = np.max(proj_j)
                        
                        # Check for overlap
                        if max_i < min_j or max_j < min_i:
                            # Separating axis found, no overlap on this axis
                            # But we need to check if they are overlapping at all.
                            # If any axis separates, they are disjoint.
                            # If NO axis separates, they are overlapping.
                            # This logic is inverted for finding penetration depth.
                            # To find penetration depth, we look for the MINIMUM overlap among all axes.
                            # If any axis has separation, pen is negative (or 0).
                            # We want the minimum positive overlap.
                            # If separation exists, overlap is negative.
                            overlap = min(max_i - min_j, max_j - min_i)
                            if overlap < 0:
                                # Separated on this axis
                                # We can break early if we find separation? 
                                # No, we need to know if they are overlapping at all.
                                # If we find separation, they are disjoint.
                                # But we are in the force loop. We should only apply force if they are overlapping.
                                # So if we find a separating axis, we break and apply no force.
                                min_pen = -1.0
                                break
                            else:
                                if overlap < min_pen:
                                    min_pen = overlap
                                    min_axis = axis
                                    
                    if min_pen > 0 and min_axis is not None:
                        # They are overlapping
                        # Force direction: from j to i? 
                        # If min_axis points from i to j, and overlap is positive, 
                        # we need to push i along -min_axis and j along +min_axis?
                        # Let's check orientation.
                        # proj_i is on axis. min_i < max_i.
                        # If max_i > min_j, they overlap.
                        # The axis is normal to an edge.
                        # Standard SAT: if overlap, push apart along the axis with minimum overlap.
                        # Direction: if center_i projected is less than center_j projected, push i in -axis, j in +axis?
                        # Actually, just push i in -axis and j in +axis if axis points from i to j?
                        # Let's use the center difference to determine sign.
                        center_i = positions[i]
                        center_j = positions[j]
                        diff = center_i - center_j
                        dot = np.dot(diff, min_axis)
                        
                        if dot > 0:
                            # i is in the direction of the axis relative to j
                            # So push i in +axis? No, if they overlap, we want to increase distance.
                            # If i is at higher proj, push i higher (+axis) and j lower (-axis)?
                            # Wait, if axis is normal to edge of i, pointing OUT of i?
                            # My axis generation: edge = v[i+1]-v[i]. normal = [-ey, ex].
                            # For a CCW polygon, this points OUT.
                            # So if axis points out of i, and i is "above" j in proj, 
                            # then i is penetrating j's space?
                            # Let's just use a robust heuristic:
                            # Push i in direction of (center_i - center_j) scaled by axis alignment.
                            # Simpler: Force on i is k * pen * axis_dir
                            # If dot > 0, i is "forward" along axis. 
                            # We want to move i further forward? No, that increases overlap if j is behind?
                            # If they overlap, moving i in +axis and j in -axis increases separation IF axis points from j to i.
                            # Let's assume axis points from j to i (since dot > 0).
                            # Then push i in +axis, j in -axis.
                            f = k_repel * min_pen * min_axis
                            forces[i] += f
                            forces[j] -= f
                        else:
                            # i is "behind" j along axis.
                            # Axis points from i to j?
                            # Push i in -axis, j in +axis.
                            f = k_repel * min_pen * min_axis
                            forces[i] -= f
                            forces[j] += f
                        
                        # Torque: try to rotate to reduce overlap
                        # Simple heuristic: add small random torque or align?
                        # Let's add a small damping torque to stabilize
                        torques[i] += -angular_damping * angular_velocities[i]
                        torques[j] += -angular_damping * angular_velocities[j]
                        
                        # Optional: Torque based on center of overlap
                        # For simplicity, rely on linear forces and let angles stay 0 or drift slightly
                        # Actually, allowing rotation is key for packing.
                        # Let's add a torque that tries to align the hexes to minimize bounding box?
                        # Or just random small torque to explore.
                        # Let's try a torque that pushes vertices out of the other polygon.
                        # This is complex. Let's stick to linear forces for now and see if angles need to change.
                        # If angles are fixed at 0, it's a restricted problem.
                        # The task allows rotation.
                        # Let's add a small random torque to escape local minima in orientation.
                        # torques[i] += np.random.normal(0, 0.1)
                        # torques[j] += np.random.normal(0, 0.1)
                        
            # Update velocities and positions (Symplectic Euler)
            velocities += forces * dt
            velocities *= damping
            angular_velocities += torques * dt
            angular_velocities *= angular_damping
            
            positions += velocities * dt
            angles += angular_velocities * dt
            
            # Update polygons
            for i in range(N):
                polys[i] = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                
            # Check validity and score
            # Check if all hexes are inside the current container
            # And if they are disjoint
            valid = True
            for i in range(N):
                for k in range(6):
                    n = normals[k]
                    projs = polys[i] @ n
                    if np.max(projs) > apothem + 1e-6:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                # Check disjointness
                for i in range(N):
                    for j in range(i + 1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current_side_val = calc_min_outer_side(positions, angles)
                    if current_side_val < best_valid_side:
                        best_valid_side = current_side_val
                        best_positions = positions.copy()
                        best_angles = angles.copy()
                        
        return best_positions, best_angles, best_valid_side

    # Start simulation
    # Initial guess side length
    init_side = 5.0
    best_pos, best_ang, best_side = simulate(init_side)
    
    if best_pos is None:
        # Fallback to honeycomb if simulation failed
        sqrt3 = math.sqrt(3)
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        best_pos = init_positions
        best_ang = np.zeros(N)
        best_side = calc_min_outer_side(best_pos, best_ang)
        
    # Local refinement with Nelder-Mead on the best found
    if time.time() < deadline - 20:
        def objective(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            
            # Check validity
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint_fast(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if not valid:
                return 1e6 + calc_min_outer_side(positions, angles)
                
            return calc_min_outer_side(positions, angles)
            
        x0 = np.concatenate([best_pos.flatten(), best_ang])
        
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_side:
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_side = res.fun
                    best_pos = new_positions
                    best_ang = new_angles
        except Exception:
            pass

    final_positions = best_pos
    final_angles = best_ang
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
