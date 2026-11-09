"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """Compute minimum outer side to contain all inner hexagons."""
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        # Pre-compute normals for the outer hexagon in local coordinates
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to align with outer hexagon
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                # Find maximum projection on each of 6 normals
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap_sat(pos1, pos2):
        """
        Returns True if hexagons overlap, False if disjoint.
        Uses Separating Axis Theorem (SAT).
        """
        # Bounding box check for quick rejection
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        if dx*dx + dy*dy > 4.0:
            return False
        
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        # Collect all normals
        normals = []
        for i in range(6):
            # Edge 1
            ex = v1[(i+1)%6][0] - v1[i][0]
            ey = v1[(i+1)%6][1] - v1[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
            
            # Edge 2
            ex = v2[(i+1)%6][0] - v2[i][0]
            ey = v2[(i+1)%6][1] - v2[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
        
        # Check for separation along each normal
        for nx, ny in normals:
            # Project both hexagons onto this normal
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            # If separated, no overlap
            if max1 <= min2 or max2 <= min1:
                return False
        
        # If we get here, they overlap
        return True

    def has_overlap(inner_positions):
        """Check if any pair of hexagons overlap."""
        n = len(inner_positions)
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_sat(inner_positions[i], inner_positions[j]):
                    return True
        return False

    # Generate hexagonal lattice coordinates for 11 hexagons
    # Use a compact arrangement: 1 center + 6 ring + 4 from second ring
    lattice_coords = [(0, 0)]  # center
    # First ring (6 hexagons)
    for k in range(6):
        angle = math.radians(60 * k)
        q = round(math.cos(angle))
        r = round(math.sin(angle))
        lattice_coords.append((q, r))
    # Second ring - pick 4 more to get total of 11
    second_ring = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                second_ring.append((q, r))
    
    # Sort second ring by distance and pick closest 4
    def hex_dist_to_center(q, r):
        return abs(q) + abs(r) + abs(q+r)
    
    second_ring.sort(key=lambda x: hex_dist_to_center(x[0], x[1]))
    lattice_coords.extend(second_ring[:4])
    
    # Now optimize: lattice spacing, global rotation, individual small offsets
    # Parameters: [spacing, global_rot, offset_0x, offset_0y, offset_0_angle, ...]
    # Total params: 2 + 11*3 = 35
    
    def generate_positions(params):
        """Generate inner positions from parameter vector."""
        spacing = params[0]
        global_rot = math.radians(params[1])
        positions = []
        for i, (q, r) in enumerate(lattice_coords):
            # Base position in hex lattice
            base_x = sqrt3 * spacing * (q + r/2.0)
            base_y = 1.5 * spacing * r
            
            # Add individual offset
            ox = params[2 + i*3]
            oy = params[2 + i*3 + 1]
            oang = math.radians(params[2 + i*3 + 2])
            
            # Rotate base position by global rotation
            cos_g = math.cos(global_rot)
            sin_g = math.sin(global_rot)
            rx = base_x * cos_g - base_y * sin_g
            ry = base_x * sin_g + base_y * cos_g
            
            # Add offset (in rotated frame)
            fx = rx + ox
            fy = ry + oy
            
            # Total angle = global rotation + individual offset angle
            total_angle = params[1] + params[2 + i*3 + 2]
            
            positions.append([fx, fy, total_angle])
        return positions

    def objective(params):
        """Objective: outer side length + penalty for overlaps."""
        if time.time() > deadline - 2:
            return float('inf')
        
        positions = generate_positions(params)
        
        # Check for overlaps
        if has_overlap(positions):
            # Compute penalty as sum of minimum distances that need to be cleared
            # For simplicity, use a large penalty
            return 100.0 + compute_min_side(positions, (0.0, 0.0), 0.0)
        
        # Find best outer angle
        best_side = float('inf')
        for outer_angle in [0.0, 30.0, 60.0]:
            side = compute_min_side(positions, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
        return best_side

    # Differential Evolution / Multi-start optimization
    best_params = None
    best_val = float('inf')
    
    num_starts = 8
    pop_size = 24
    max_iter = 150
    cr = 0.8  # crossover rate
    f = 0.5   # differential weight
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Initialize population
        # Start near the ideal lattice (spacing = 1.0 for touching hexagons)
        base_params = [1.0 + 0.05 * (start_idx - num_starts/2), 0.0]
        for i in range(N):
            base_params.extend([0.0, 0.0, 0.0])
        
        population = []
        for p_idx in range(pop_size):
            params = base_params[:]
            # Add random perturbation
            for j in range(len(params)):
                if j == 0:  # spacing
                    params[j] += random.gauss(0, 0.1)
                elif j == 1:  # global rotation
                    params[j] += random.gauss(0, 10.0)
                else:  # offsets
                    params[j] += random.gauss(0, 0.1)
            population.append(params)
        
        # Evaluate initial population
        pop_vals = [objective(p) for p in population]
        
        # Track best
        for i in range(pop_size):
            if pop_vals[i] < best_val:
                best_val = pop_vals[i]
                best_params = population[i][:]
        
        # Differential evolution iterations
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Find best, second best, third best
            sorted_idx = sorted(range(pop_size), key=lambda x: pop_vals[x])
            best_i = sorted_idx[0]
            second_i = sorted_idx[1]
            third_i = sorted_idx[2]
            
            for i in range(pop_size):
                if time.time() > deadline - 3:
                    break
                
                # Skip if this is the best
                if i == best_i:
                    continue
                
                # Select three distinct random individuals (excluding i)
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = random.sample(candidates, 3)
                
                # Create trial vector
                trial = population[i][:]
                for j in range(len(trial)):
                    if random.random() < cr:
                        trial[j] = population[a][j] + f * (population[b][j] - population[c][j])
                    else:
                        trial[j] = population[i][j]
                
                # Evaluate trial
                trial_val = objective(trial)
                
                # Selection
                if trial_val <= pop_vals[i]:
                    population[i] = trial
                    pop_vals[i] = trial_val
                    if trial_val < best_val:
                        best_val = trial_val
                        best_params = trial[:]
                
                # Update best index
                sorted_idx = sorted(range(pop_size), key=lambda x: pop_vals[x])
                best_i = sorted_idx[0]
                second_i = sorted_idx[1]
                third_i = sorted_idx[2]
    
    # Generate final positions from best parameters
    if best_params is None:
        # Fallback: use ideal lattice
        inner_positions = []
        for q, r in lattice_coords:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            inner_positions.append([x, y, 0.0])
    else:
        inner_positions = generate_positions(best_params)
    
    # Fine-tune with local search: adjust outer angle
    best_side = float('inf')
    best_angle = 0.0
    for outer_angle in np.arange(0, 60, 2.0):
        if time.time() > deadline - 1:
            break
        side = compute_min_side(inner_positions, (0.0, 0.0), outer_angle)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
    
    # Verify no overlap
    if has_overlap(inner_positions):
        # Fallback to safe packing
        safe_inner = []
        for i in range(N):
            if i == 0:
                safe_inner.append([0.0, 0.0, 0.0])
            else:
                ring = (i - 1) // 6 + 1
                pos_in_ring = (i - 1) % 6
                r = 2.1 * ring
                angle = math.radians(60 * pos_in_ring + 30)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                safe_inner.append([x, y, 0.0])
        
        best_side = float('inf')
        for outer_angle in [0.0, 30.0, 60.0]:
            side = compute_min_side(safe_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle
                inner_positions = safe_inner
    
    return inner_positions, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
