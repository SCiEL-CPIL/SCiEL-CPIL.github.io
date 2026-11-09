"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator limit 150s)
    
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

    # Helper: Check if two hexagons overlap using Separating Axis Theorem (SAT)
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

    def is_disjoint(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
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

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Base 7 hexagons + 4 more
    coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    
    def run_dynamics(initial_positions, initial_angles, dt, n_steps, attract_strength, damping):
        """
        Rigid-body dynamics simulation with velocity-Verlet integration.
        Returns best (positions, angles, side_length) found during simulation.
        """
        # State: positions [N,2], velocities [N,2], angles [N], angular_vels [N]
        pos = np.array(initial_positions, dtype=float)
        vel = np.zeros((N, 2))
        ang = np.array(initial_angles, dtype=float)
        ang_vel = np.zeros(N)
        
        # Center of mass
        com = pos.mean(axis=0)
        
        best_side = float('inf')
        best_pos = pos.copy()
        best_ang = ang.copy()
        
        # Precompute hexagon vertices for each body (updated each step)
        # Force constants
        k_contact = 50.0  # Contact stiffness
        k_attract = attract_strength  # Attraction to center
        k_boundary = 100.0  # Boundary repulsion
        
        for step in range(n_steps):
            # Center positions
            pos_centered = pos - com
            
            # Compute forces on each hexagon
            forces = np.zeros((N, 2))
            torques = np.zeros(N)
            
            # 1. Attraction toward center (spring force)
            for i in range(N):
                dx = -pos_centered[i, 0]
                dy = -pos_centered[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0.01:
                    f = k_attract * pos_centered[i]  # Pull toward center
                    forces[i, 0] += f[0]
                    forces[i, 1] += f[1]
            
            # 2. Contact forces between hexagons (penalty method)
            for i in range(N):
                for j in range(i+1, N):
                    # Get vertices
                    verts_i = get_hex_vertices(pos_centered[i,0], pos_centered[i,1], math.degrees(ang[i]))
                    verts_j = get_hex_vertices(pos_centered[j,0], pos_centered[j,1], math.degrees(ang[j]))
                    
                    # Check for overlap using SAT
                    axes_i = get_axes(verts_i)
                    axes_j = get_axes(verts_j)
                    all_axes = axes_i + axes_j
                    
                    min_overlap = float('inf')
                    min_axis = None
                    
                    for axis in all_axes:
                        min1, max1 = project(verts_i, axis)
                        min2, max2 = project(verts_j, axis)
                        
                        # If separated
                        if max1 < min2 or max2 < min1:
                            min_overlap = 0.0
                            min_axis = None
                            break
                        
                        # Overlap depth
                        overlap = min(max1, max2) - max(min1, min2)
                        if overlap < min_overlap:
                            min_overlap = overlap
                            min_axis = axis
                    
                    if min_axis is not None and min_overlap > 0:
                        # Apply repulsive force
                        # Determine direction: from i to j
                        dx = pos_centered[j,0] - pos_centered[i,0]
                        dy = pos_centered[j,1] - pos_centered[i,1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 0.01:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        f_mag = k_contact * min_overlap
                        forces[i, 0] -= f_mag * dx
                        forces[i, 1] -= f_mag * dy
                        forces[j, 0] += f_mag * dx
                        forces[j, 1] += f_mag * dy
                        
                        # Torque from off-center force application
                        # Simplified: use perpendicular component
                        cross = dx * (pos_centered[i,1] - com[1]) - dy * (pos_centered[i,0] - com[0])
                        # This is a simplification; proper torque would need contact point
                        torques[i] += 0.1 * f_mag * cross
                        torques[j] -= 0.1 * f_mag * cross
            
            # 3. Boundary constraint: keep hexagons within a virtual outer hexagon
            # We'll use a circular boundary approximation for simplicity
            # The outer hexagon has apothem = S * SQRT3 / 2
            # We want to minimize S, so we use a tight boundary
            # For now, we'll let the attraction force handle compaction
            
            # Velocity-Verlet integration
            # Update velocities
            for i in range(N):
                vel[i, 0] += forces[i, 0] * dt
                vel[i, 1] += forces[i, 1] * dt
                ang_vel[i] += torques[i] * dt * 0.1  # Reduced inertia effect
                
                # Damping
                vel[i, 0] *= (1.0 - damping)
                vel[i, 1] *= (1.0 - damping)
                ang_vel[i] *= (1.0 - damping)
            
            # Update positions
            for i in range(N):
                pos[i, 0] += vel[i, 0] * dt
                pos[i, 1] += vel[i, 1] * dt
                ang[i] += ang_vel[i] * dt
            
            # Keep angles in [0, 2*pi)
            ang = ang % (2 * math.pi)
            
            # Evaluate current configuration
            if step % 10 == 0:  # Check every 10 steps
                # Get vertices
                hexagons = []
                for i in range(N):
                    hex_verts = get_hex_vertices(pos_centered[i,0], pos_centered[i,1], math.degrees(ang[i]))
                    hexagons.append(hex_verts)
                
                # Check validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not is_disjoint(hexagons[i], hexagons[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Find best outer angle
                    best_side_curr = float('inf')
                    for oa_deg in range(0, 60, 5):
                        side = calc_outer_side(hexagons, oa_deg)
                        if side < best_side_curr:
                            best_side_curr = side
                    
                    if best_side_curr < best_side:
                        best_side = best_side_curr
                        best_pos = pos.copy()
                        best_ang = ang.copy()
            
            # Time check
            if time.time() > deadline - 10.0:
                break
        
        return best_pos, best_ang, best_side
    
    # Initial configuration: tight hexagonal lattice
    initial_positions = []
    initial_angles = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, spacing=1.0)
        initial_positions.append([x, y])
        initial_angles.append(0.0)
    
    # Run multiple dynamics simulations with different parameters
    best_result = None
    best_side_global = float('inf')
    
    # Try different attraction strengths and damping
    param_sets = [
        (0.5, 0.05, 1000),   # dt, damping, n_steps
        (0.3, 0.03, 1500),
        (0.2, 0.02, 2000),
        (0.4, 0.08, 800),
    ]
    
    for dt, damping, n_steps in param_sets:
        if time.time() > deadline - 20.0:
            break
        
        # Add small random perturbation to break symmetry
        perturbed_pos = []
        perturbed_ang = []
        for i in range(N):
            px = initial_positions[i][0] + random.gauss(0, 0.01)
            py = initial_positions[i][1] + random.gauss(0, 0.01)
            pa = initial_angles[i] + random.gauss(0, 0.01)
            perturbed_pos.append([px, py])
            perturbed_ang.append(pa)
        
        pos, ang, side = run_dynamics(perturbed_pos, perturbed_ang, dt, n_steps, 0.5, damping)
        
        if side < best_side_global:
            best_side_global = side
            best_result = (pos, ang, side)
    
    # If dynamics didn't find a valid solution, fall back to initial lattice
    if best_result is None or best_result[2] >= float('inf'):
        # Use initial lattice
        pos = np.array(initial_positions)
        ang = np.array(initial_angles)
        
        # Center
        com = pos.mean(axis=0)
        pos_centered = pos - com
        
        hexagons = []
        for i in range(N):
            hex_verts = get_hex_vertices(pos_centered[i,0], pos_centered[i,1], math.degrees(ang[i]))
            hexagons.append(hex_verts)
        
        best_side = float('inf')
        best_oa = 0.0
        for oa_deg in range(0, 60, 1):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                best_oa = oa_deg
        
        inner_list = []
        for i in range(N):
            inner_list.append([pos_centered[i][0], pos_centered[i][1], math.degrees(ang[i])])
        
        return (inner_list, (0.0, 0.0), best_side, best_oa)
    
    # Extract best result
    pos, ang, side = best_result
    
    # Center
    com = pos.mean(axis=0)
    pos_centered = pos - com
    
    # Get vertices for final validation
    hexagons = []
    for i in range(N):
        hex_verts = get_hex_vertices(pos_centered[i,0], pos_centered[i,1], math.degrees(ang[i]))
        hexagons.append(hex_verts)
    
    # Check validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to initial lattice
        pos = np.array(initial_positions)
        ang = np.array(initial_angles)
        com = pos.mean(axis=0)
        pos_centered = pos - com
        
        hexagons = []
        for i in range(N):
            hex_verts = get_hex_vertices(pos_centered[i,0], pos_centered[i,1], math.degrees(ang[i]))
            hexagons.append(hex_verts)
        
        best_side = float('inf')
        best_oa = 0.0
        for oa_deg in range(0, 60, 1):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                best_oa = oa_deg
        
        inner_list = []
        for i in range(N):
            inner_list.append([pos_centered[i][0], pos_centered[i][1], math.degrees(ang[i])])
        
        return (inner_list, (0.0, 0.0), best_side, best_oa)
    
    # Find best outer angle (finer search)
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 60, 1):
        side = calc_outer_side(hexagons, oa_deg)
        if side < best_side:
            best_side = side
            best_oa = oa_deg
    
    inner_list = []
    for i in range(N):
        inner_list.append([pos_centered[i][0], pos_centered[i][1], math.degrees(ang[i])])
    
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
