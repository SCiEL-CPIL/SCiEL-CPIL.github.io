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
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
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
        # The normals to the edges of a regular hexagon (centered at origin)
        # are at angles 30, 90, 150, 210, 270, 330 degrees.
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
                    # The distance from center to edge (apothem) is side * sqrt(3) / 2
                    # So side = proj / (sqrt(3) / 2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap_sat(pos1, pos2):
        """Check if two hexagons overlap using Separating Axis Theorem."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dist_sq = dx*dx + dy*dy
        # Max distance between centers for non-overlap is roughly 2*side*sqrt(3)/2 * 2? 
        # Actually, if they are far apart, no overlap. 
        # If dist > 4.0, they definitely don't overlap (side 1, max width 2).
        if dist_sq > 4.0:
            return False
        
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        normals = []
        for i in range(6):
            ex = v1[(i+1)%6][0] - v1[i][0]
            ey = v1[(i+1)%6][1] - v1[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
            
            ex = v2[(i+1)%6][0] - v2[i][0]
            ey = v2[(i+1)%6][1] - v2[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
        
        for nx, ny in normals:
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            if max1 <= min2 or max2 <= min1:
                return False
        
        return True

    def relax_positions(positions, max_iters=50):
        """Iteratively push overlapping hexagons apart."""
        n = len(positions)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            any_overlap = False
            for i in range(n):
                for j in range(i+1, n):
                    if check_overlap_sat(positions[i], positions[j]):
                        any_overlap = True
                        # Push apart
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            # Random nudge
                            angle = random.uniform(0, 2*math.pi)
                            positions[i][0] += 0.1 * math.cos(angle)
                            positions[i][1] += 0.1 * math.sin(angle)
                            positions[j][0] -= 0.1 * math.cos(angle)
                            positions[j][1] -= 0.1 * math.sin(angle)
                        else:
                            # Move each 0.5 * overlap_depth? 
                            # Approximate overlap depth as 0.1 for stability
                            # Or just move them apart slightly
                            move = 0.05
                            positions[i][0] += move * (dx / dist)
                            positions[i][1] += move * (dy / dist)
                            positions[j][0] -= move * (dx / dist)
                            positions[j][1] -= move * (dy / dist)
            if not any_overlap:
                break
        return positions

    # Initial guess: hexagonal lattice
    # 11 hexagons: center + 6 ring + 4 more
    # Use axial coordinates
    lattice_coords = [(0, 0)]
    for k in range(6):
        angle = math.radians(60 * k)
        q = round(math.cos(angle))
        r = round(math.sin(angle))
        lattice_coords.append((q, r))
    second_ring = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                second_ring.append((q, r))
    def hex_dist(q, r):
        return abs(q) + abs(r) + abs(q+r)
    second_ring.sort(key=lambda x: hex_dist(x[0], x[1]))
    lattice_coords.extend(second_ring[:4])
    
    # Convert to Cartesian
    init_positions = []
    for q, r in lattice_coords:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        init_positions.append([x, y, 0.0])
    
    # Relax to ensure no overlaps
    relaxed_positions = relax_positions(init_positions)
    
    best_side = float('inf')
    best_angle = 0.0
    best_positions = relaxed_positions
    
    # Try a few random perturbations and relaxations to find a tighter packing
    num_starts = 5
    for start in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Perturb initial guess
        positions = [p[:] for p in init_positions]
        for p in positions:
            p[0] += random.gauss(0, 0.5)
            p[1] += random.gauss(0, 0.5)
            p[2] = random.uniform(0, 60)
        
        # Relax
        positions = relax_positions(positions, max_iters=30)
        
        # Evaluate
        for outer_angle in [0.0, 30.0]:
            side = compute_min_side(positions, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle
                best_positions = positions
    
    # Final check and fine-tune outer angle
    # Ensure best_positions are valid
    # (Relaxation should have ensured this, but just in case)
    best_positions = relax_positions(best_positions, max_iters=10)
    
    final_side = float('inf')
    final_angle = 0.0
    for outer_angle in np.arange(0, 60, 2.0):
        if time.time() > deadline - 1:
            break
        side = compute_min_side(best_positions, (0.0, 0.0), outer_angle)
        if side < final_side:
            final_side = side
            final_angle = outer_angle
    
    return best_positions, (0.0, 0.0), final_side, final_angle
# EVOLVE-BLOCK-END
