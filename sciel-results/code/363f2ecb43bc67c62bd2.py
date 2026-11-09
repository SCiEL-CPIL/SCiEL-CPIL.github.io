"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s, keep 5s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    # The outer hexagon is centered at the origin.
    def calc_outer_side(hexagons, outer_angle_deg):
        # Outer hexagon normals (flat-top, rotated by outer_angle_deg)
        # Sides are perpendicular to normals at 30, 90, 150, 210, 270, 330 + outer_angle
        base_angles = [30, 90, 150, 210, 270, 330]
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # Max projection must be <= Apothem
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Helper: Calculate penetration depth between two convex polygons using SAT
    # Returns 0 if disjoint, > 0 if overlapping
    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm == 0: continue
            nx = -dy / norm
            ny = dx / norm
            axes.append((nx, ny))
        return axes

    def project(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def penetration_depth(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        min_pen = float('inf')
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                return 0.0
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_pen:
                min_pen = overlap
        return min_pen

    # PBD Simulation State
    # State: list of [x, y, angle_deg] for each hexagon
    # We use a "soft" potential energy approach with PBD-like corrections.
    
    def axial_to_cartesian(q, r, spacing=2.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial setup: Start with a loose lattice configuration to ensure valid start
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    extra = [(2,0), (2,-1), (2,1), (1,1)]
    coords = base_7 + extra
    
    # Initialize positions
    positions = []
    angles = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, spacing=2.2) # Slightly loose to start
        positions.append([x, y])
        angles.append(0.0)
        
    # Center the cluster
    cx_avg = sum(p[0] for p in positions) / N
    cy_avg = sum(p[1] for p in positions) / N
    for i in range(N):
        positions[i][0] -= cx_avg
        positions[i][1] -= cy_avg

    best_side = float('inf')
    best_state = None
    best_angles = None
    
    # PBD Parameters
    # We iterate over "frames". In each frame, we resolve overlaps and apply confinement.
    num_frames = 5000
    dt = 1.0
    stiffness = 0.5 # Correction factor for overlaps
    pressure_strength = 0.1 # Force pushing towards center (confinement)
    damping = 0.9 # Velocity damping
    
    velocities = [[0.0, 0.0] for _ in range(N)]
    angular_velocities = [0.0 for _ in range(N)]
    
    # Pre-compute vertices for initial state? No, they change.
    
    for frame in range(num_frames):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Calculate Forces / Constraints
        # We use a simple iterative solver approach (Gauss-Seidel style for PBD)
        # Instead of full forces, we apply positional corrections.
        
        # Update velocities with "pressure" (confinement)
        # Pressure pushes all hexagons towards the origin to minimize bounding box
        for i in range(N):
            # Force towards origin
            fx = -positions[i][0] * pressure_strength
            fy = -positions[i][1] * pressure_strength
            
            velocities[i][0] += fx * dt
            velocities[i][1] += fy * dt
            
            # Angular damping (we don't have a clear torque source yet, but let's allow rotation freedom)
            # Actually, for regular hexagons, rotation doesn't change the shape relative to itself, 
            # but it changes the orientation relative to neighbors. 
            # In a tight packing, orientations matter. 
            # Let's add a small random angular perturbation to escape local minima in orientation.
            angular_velocities[i] += random.gauss(0, 0.1)
            angular_velocities[i] *= 0.9 # Damping

        # 2. Integrate positions
        for i in range(N):
            positions[i][0] += velocities[i][0] * dt
            positions[i][1] += velocities[i][1] * dt
            angles[i] += angular_velocities[i] * dt
            
            # Damping
            velocities[i][0] *= damping
            velocities[i][1] *= damping

        # 3. Collision Resolution (PBD Step)
        # Iterate a few times to resolve overlaps
        iterations = 5
        for _ in range(iterations):
            for i in range(N):
                for j in range(i+1, N):
                    verts_i = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
                    verts_j = get_hex_vertices(positions[j][0], positions[j][1], angles[j])
                    
                    pen = penetration_depth(verts_i, verts_j)
                    if pen > 1e-6:
                        # Calculate correction direction
                        # Simple approach: push centers apart
                        dx = positions[j][0] - positions[i][0]
                        dy = positions[j][1] - positions[i][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            dx = random.uniform(-1, 1)
                            dy = random.uniform(-1, 1)
                            dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Correction amount
                        corr = pen * stiffness
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Apply correction
                        positions[i][0] -= nx * corr * 0.5
                        positions[i][1] -= ny * corr * 0.5
                        positions[j][0] += nx * corr * 0.5
                        positions[j][1] += ny * corr * 0.5
                        
                        # Adjust velocities to reflect the impulse (optional, but helps stability)
                        # We just zero out the relative velocity component along the normal
                        v_rel_x = velocities[j][0] - velocities[i][0]
                        v_rel_y = velocities[j][1] - velocities[i][1]
                        v_n = v_rel_x * nx + v_rel_y * ny
                        if v_n < 0: # Moving towards each other
                            impulse = -v_n * 0.5
                            velocities[i][0] -= nx * impulse
                            velocities[i][1] -= ny * impulse
                            velocities[j][0] += nx * impulse
                            velocities[j][1] += ny * impulse

        # 4. Check validity and score
        # Only evaluate if no overlaps remain
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                verts_i = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
                verts_j = get_hex_vertices(positions[j][0], positions[j][1], angles[j])
                if penetration_depth(verts_i, verts_j) > 1e-5:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Calculate outer side
            # Try a few outer angles to find the best fit
            hexagons = [get_hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
            current_side = float('inf')
            for oa_deg in range(0, 60, 5):
                side = calc_outer_side(hexagons, oa_deg)
                if side < current_side:
                    current_side = side
            
            if current_side < best_side:
                best_side = current_side
                # Store deep copy
                best_state = [p[:] for p in positions]
                best_angles = angles[:]
                # Save the outer angle that gave this best side? 
                # We'll re-calculate the best outer angle at the end for the final state.
                
    # Final Polish: If we have a valid state, try to polish it with a simple local search
    if best_state is not None:
        # Use the best state found by PBD
        final_positions = [p[:] for p in best_state]
        final_angles = best_angles[:]
        
        # Quick local optimization of outer angle
        hexagons = [get_hex_vertices(final_positions[i][0], final_positions[i][1], final_angles[i]) for i in range(N)]
        best_outer_angle = 0.0
        final_side = float('inf')
        for oa_deg in range(0, 60, 1):
            side = calc_outer_side(hexagons, oa_deg)
            if side < final_side:
                final_side = side
                best_outer_angle = oa_deg
    else:
        # Fallback to loose lattice if PBD failed to find a valid packing
        final_positions = []
        final_angles = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=2.2)
            final_positions.append([x, y])
            final_angles.append(0.0)
            
        cx_avg = sum(p[0] for p in final_positions) / N
        cy_avg = sum(p[1] for p in final_positions) / N
        for i in range(N):
            final_positions[i][0] -= cx_avg
            final_positions[i][1] -= cy_avg
            
        hexagons = [get_hex_vertices(final_positions[i][0], final_positions[i][1], final_angles[i]) for i in range(N)]
        best_outer_angle = 0.0
        final_side = float('inf')
        for oa_deg in range(0, 60, 1):
            side = calc_outer_side(hexagons, oa_deg)
            if side < final_side:
                final_side = side
                best_outer_angle = oa_deg

    # Construct return list
    inner_list = []
    for i in range(N):
        inner_list.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
        
    return (inner_list, (0.0, 0.0), final_side, best_outer_angle)

# EVOLVE-BLOCK-END
