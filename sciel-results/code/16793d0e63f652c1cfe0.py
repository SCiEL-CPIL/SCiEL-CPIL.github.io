"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Helper: Support function of a regular hexagon (side 1) rotated by angle_deg in direction (ux, uy)
    def hex_support(ux, uy, angle_deg):
        # Rotate direction vector by -angle to align hexagon
        a = -math.radians(angle_deg)
        rx = ux * math.cos(a) - uy * math.sin(a)
        ry = ux * math.sin(a) + uy * math.cos(a)
        
        # Support function of regular hexagon (side 1) in direction (rx, ry)
        # Vertices are at angles 0, 60, ...
        # h(u) = max( cos(theta_i - phi) ) where phi is angle of u?
        # Easier: Project vertices onto u.
        # Vertices of unit hex: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Actually, vertices of side 1 hex:
        # v0 = (1, 0)
        # v1 = (0.5, sqrt(3)/2)
        # ...
        # Let's compute max dot product with these 6 vertices.
        verts_x = [1.0, 0.5, -0.5, -1.0, -0.5, 0.5]
        verts_y = [0.0, math.sqrt(3)/2, math.sqrt(3)/2, 0.0, -math.sqrt(3)/2, -math.sqrt(3)/2]
        
        max_proj = -1e9
        for vx, vy in zip(verts_x, verts_y):
            # The vector (rx, ry) is the direction in the hexagon's local frame
            # The support is max(v . u_local)
            proj = vx * rx + vy * ry
            if proj > max_proj:
                max_proj = proj
        return max_proj

    # Helper: Get vertices of a regular hexagon (side 1) centered at (0,0) rotated by angle_deg
    def hex_vertices_local(angle_deg):
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3
            x = math.cos(theta)
            y = math.sin(theta)
            verts.append((x, y))
        return verts

    # Main Optimization Loop
    # State: positions (Nx2), angles (N,), outer_angle (float)
    # We will optimize positions and outer_side.
    # Angles will be fixed to 0 initially, or we can add a coarse angle search.
    # For speed and stability, we'll start with angles = 0.
    
    # Initialize Honeycomb
    def init_honeycomb():
        positions = []
        angles = [0.0] * N
        # 4 rows of 3
        # Spacing for non-overlapping hexagons: distance between centers must be >= 2 * (sqrt(3)/2) = sqrt(3)
        # In a honeycomb, vertical dist = 1.5, horizontal = sqrt(3)
        # Let's use a standard honeycomb lattice
        for row in range(4):
            for col in range(3):
                x = col * math.sqrt(3) + (row % 2) * math.sqrt(3) / 2
                y = row * 1.5
                positions.append([x, y])
        
        # Center around origin
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [[x - cx, y - cy] for x, y in positions]
        return positions, angles

    positions, angles = init_honeycomb()
    outer_angle = 0.0
    
    best_side = 1e10
    best_positions = positions
    best_angles = angles
    best_outer_angle = outer_angle

    # LP Variables: x_0...x_11, y_0...y_11, S
    # Indices: 0..11 are x, 12..23 are y, 24 is S
    n_vars = 2 * N + 1
    idx_S = 2 * N

    # Objective: Minimize S
    c_obj = np.zeros(n_vars)
    c_obj[idx_S] = 1.0

    # Precompute outer normals for angle 0 (we can rotate outer hex if needed, but for now fix to 0)
    # If we allow outer rotation, it adds a variable or requires re-solving. 
    # Let's fix outer angle to 0 for the LP, and maybe try a few fixed outer angles in a loop.
    
    for outer_angle_trial in [0.0, 30.0]:
        if time.time() > deadline - 10:
            break
            
        # Current best for this outer angle
        cur_pos = [p[:] for p in best_positions]
        cur_angles = best_angles[:]
        cur_side = best_side
        
        # We will run the LP relaxation loop for this outer angle
        # We need to handle the non-overlap constraints carefully.
        # We maintain a set of "active" separating axes for each pair.
        # Init axes to be the vector connecting centers.
        
        pair_axes = {}
        for i in range(N):
            for j in range(i+1, N):
                dx = cur_pos[i][0] - cur_pos[j][0]
                dy = cur_pos[i][1] - cur_pos[j][1]
                norm = math.sqrt(dx*dx + dy*dy)
                if norm < 1e-6:
                    pair_axes[(i,j)] = (1.0, 0.0)
                else:
                    pair_axes[(i,j)] = (dx/norm, dy/norm)

        max_lp_iters = 50
        for it in range(max_lp_iters):
            if time.time() > deadline - 5:
                break

            # Build A_ub and b_ub
            # Constraints: A_ub @ x <= b_ub
            
            A_rows = []
            b_rows = []

            # 1. Containment Constraints
            # Outer hex normals
            theta_out = math.radians(outer_angle_trial)
            outer_normals = []
            for k in range(6):
                a = theta_out + k * math.pi / 3
                outer_normals.append((math.cos(a), math.sin(a)))
            
            # For each inner hex i, and each of its 6 vertices v_m
            for i in range(N):
                verts = hex_vertices_local(angles[i])
                for vx, vy in verts:
                    # Vertex pos = (x_i + vx, y_i + vy)
                    # For each outer normal (nx, ny):
                    # (x_i + vx) * nx + (y_i + vy) * ny <= S / sqrt(3)
                    # x_i * nx + y_i * ny - S / sqrt(3) <= - (vx * nx + vy * ny)
                    # AND
                    # -x_i * nx - y_i * ny + S / sqrt(3) <= (vx * nx + vy * ny)
                    
                    for nx, ny in outer_normals:
                        # Constraint 1: x_i*nx + y_i*ny - S/sqrt(3) <= -dot(v, n)
                        row1 = np.zeros(n_vars)
                        row1[i] = nx
                        row1[N+i] = ny
                        row1[idx_S] = -1.0 / math.sqrt(3)
                        A_rows.append(row1)
                        b_rows.append(-(vx * nx + vy * ny))
                        
                        # Constraint 2: -x_i*nx - y_i*ny + S/sqrt(3) <= dot(v, n)
                        row2 = np.zeros(n_vars)
                        row2[i] = -nx
                        row2[N+i] = -ny
                        row2[idx_S] = 1.0 / math.sqrt(3)
                        A_rows.append(row2)
                        b_rows.append(vx * nx + vy * ny)

            # 2. Non-Overlap Constraints
            # For each pair (i, j):
            # (x_i - x_j) * ux + (y_i - y_j) * uy >= h_i(u) + h_j(u)
            # -x_i*ux + x_j*ux - y_i*uy + y_j*uy <= -(h_i(u) + h_j(u))
            
            for i in range(N):
                for j in range(i+1, N):
                    ux, uy = pair_axes[(i,j)]
                    
                    # Calculate support functions
                    h_i = hex_support(ux, uy, angles[i])
                    h_j = hex_support(ux, uy, angles[j])
                    required_sep = h_i + h_j
                    
                    row = np.zeros(n_vars)
                    row[i] = -ux
                    row[N+i] = -uy
                    row[j] = ux
                    row[N+j] = uy
                    # S coefficient is 0
                    
                    A_rows.append(row)
                    b_rows.append(-required_sep)

            A_ub = np.array(A_rows)
            b_ub = np.array(b_rows)

            # Bounds: None (variables can be any real number, S >= 0 is implicit from containment)
            # Actually, S must be positive. But containment constraints will force it.
            # We should add S >= 0.
            # linprog handles bounds separately.
            bounds = [(None, None)] * n_vars
            bounds[idx_S] = (0, None)

            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    new_pos = []
                    for i in range(N):
                        new_pos.append([res.x[i], res.x[N+i]])
                    new_side = res.x[idx_S]
                    
                    if new_side < cur_side:
                        cur_side = new_side
                        cur_pos = new_pos
                        
                        # Update axes: 
                        # For each pair, check if the current axis is still the "tightest" or if we need to switch.
                        # We can check the gap for a set of candidate axes.
                        # Candidate axes: normals of the Minkowski sum edges.
                        # Simpler: Check the current axis gap. If it's large, try others?
                        # Actually, the LP solution satisfies the constraint for the *chosen* axis.
                        # To improve, we should choose the axis that is *most violated* in the current configuration 
                        # or the axis that gives the *smallest* required separation (tightest constraint).
                        
                        # Let's re-evaluate the "best" separating axis for each pair based on new positions.
                        # We want the axis that maximizes the gap? No, we want the axis that is *required* to be positive.
                        # The constraint is $(c_i - c_j) \cdot u \ge h_i(u) + h_j(u)$.
                        # The "slack" is $(c_i - c_j) \cdot u - (h_i(u) + h_j(u))$.
                        # We want to pick the $u$ that minimizes this slack (makes the constraint tightest).
                        # If we pick a "loose" axis, the LP might allow the hexagons to overlap in reality.
                        # So we must pick the axis that is *most likely to be violated* or is the *actual* separating axis.
                        
                        # Heuristic: Check the normal to the line connecting centers, and the normals of the edges of the Minkowski sum.
                        # For simplicity, we will check a grid of 12 angles.
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = cur_pos[i][0] - cur_pos[j][0]
                                dy = cur_pos[i][1] - cur_pos[j][1]
                                norm = math.sqrt(dx*dx + dy*dy)
                                if norm < 1e-6:
                                    best_axis = (1.0, 0.0)
                                    best_slack = 1e9
                                else:
                                    best_axis = (dx/norm, dy/norm)
                                    best_slack = 1e9
                                    
                                    # Check 12 candidate axes
                                    for k in range(12):
                                        theta = k * math.pi / 6
                                        ux = math.cos(theta)
                                        uy = math.sin(theta)
                                        
                                        h_i = hex_support(ux, uy, angles[i])
                                        h_j = hex_support(ux, uy, angles[j])
                                        req = h_i + h_j
                                        
                                        # Actual projection
                                        proj = dx * ux + dy * uy
                                        
                                        slack = proj - req
                                        if slack < best_slack:
                                            best_slack = slack
                                            best_axis = (ux, uy)
                                    
                                    pair_axes[(i,j)] = best_axis

                    # If no improvement or failure, break to avoid infinite loop on bad LP
                    if not res.success:
                        break
                        
            except Exception as e:
                break

        if cur_side < best_side:
            best_side = cur_side
            best_positions = cur_pos
            best_angles = cur_angles
            best_outer_angle = outer_angle_trial

    # Final Verification
    # We must ensure the solution is actually valid (no overlaps) because the LP relaxation might be loose
    # if the axis selection didn't catch the true overlap.
    # We will run a final check and if invalid, we might need to reject or fallback.
    
    def is_valid_solution(positions, angles, outer_angle, side):
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                # Quick distance check
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Min distance between centers of two unit hexagons is sqrt(3) if aligned, but can be less if rotated?
                # The minimum distance between centers of two disjoint unit hexagons is sqrt(3).
                # If dist < sqrt(3) - epsilon, they MUST overlap.
                if dist < math.sqrt(3) - 0.01:
                    return False
                
                # Full SAT check
                v1 = np.array(hex_vertices_local(angles[i])) + np.array(positions[i])
                v2 = np.array(hex_vertices_local(angles[j])) + np.array(positions[j])
                
                # SAT
                axes = []
                for k in range(6):
                    e1 = v1[k] - v1[(k+1)%6]
                    axes.append((-e1[1], e1[0]))
                    e2 = v2[k] - v2[(k+1)%6]
                    axes.append((-e2[1], e2[0]))
                
                for ax in axes:
                    p1 = np.dot(v1, ax)
                    p2 = np.dot(v2, ax)
                    if p1.max() < p2.min() or p2.max() < p1.min():
                        break
                else:
                    return False # No separating axis found -> Overlap
        
        # Check Containment
        theta_out = math.radians(outer_angle)
        outer_normals = []
        for k in range(6):
            a = theta_out + k * math.pi / 3
            outer_normals.append((math.cos(a), math.sin(a)))
        
        for i in range(N):
            verts = np.array(hex_vertices_local(angles[i])) + np.array(positions[i])
            for nx, ny in outer_normals:
                proj = np.dot(verts, (nx, ny))
                if proj.max() > side / math.sqrt(3) + 1e-6:
                    return False
                if proj.min() < -side / math.sqrt(3) - 1e-6:
                    return False
        
        return True

    if is_valid_solution(best_positions, best_angles, best_outer_angle, best_side):
        inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
        center = (0.0, 0.0)
        return inner, center, best_side, best_outer_angle
    else:
        # Fallback to initial honeycomb if LP result is invalid
        # This ensures we always return a valid solution
        positions, angles = init_honeycomb()
        # Compute side for this
        # Max projection of honeycomb
        max_proj = 0
        theta_out = math.radians(0.0)
        outer_normals = []
        for k in range(6):
            a = theta_out + k * math.pi / 3
            outer_normals.append((math.cos(a), math.sin(a)))
        
        for i in range(N):
            verts = np.array(hex_vertices_local(0.0)) + np.array(positions[i])
            for nx, ny in outer_normals:
                proj = np.dot(verts, (nx, ny))
                max_proj = max(max_proj, proj.max())
        
        side = 2 * max_proj / math.sqrt(3)
        inner = [[positions[i][0], positions[i][1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
