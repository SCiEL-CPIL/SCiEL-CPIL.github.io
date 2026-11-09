"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap_sat(pos1, pos2):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dist_sq = dx*dx + dy*dy
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

    def overlap_depth(pos1, pos2):
        """Compute approximate overlap depth between two hexagons."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1e-9:
            return 0.5, (1.0, 0.0)
        
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
        
        min_overlap = float('inf')
        best_normal = (dx/dist, dy/dist)
        
        for nx, ny in normals:
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            if max1 > min2 and max2 > min1:
                overlap = min(max1 - min2, max2 - min1)
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_normal = (nx, ny)
        
        if min_overlap < float('inf'):
            return min_overlap, best_normal
        return 0.0, (dx/dist, dy/dist)

    def relax_positions(positions, max_iters=100, dt=0.02):
        """Position-Based Dynamics relaxation."""
        n = len(positions)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            any_overlap = False
            forces = [[0.0, 0.0] for _ in range(n)]
            
            for i in range(n):
                for j in range(i+1, n):
                    if check_overlap_sat(positions[i], positions[j]):
                        any_overlap = True
                        depth, normal = overlap_depth(positions[i], positions[j])
                        if depth > 0:
                            # Apply corrective force
                            forces[i][0] += depth * normal[0]
                            forces[i][1] += depth * normal[1]
                            forces[j][0] -= depth * normal[0]
                            forces[j][1] -= depth * normal[1]
            
            # Update positions
            for i in range(n):
                positions[i][0] += dt * forces[i][0]
                positions[i][1] += dt * forces[i][1]
            
            if not any_overlap:
                break
        
        return positions

    def anneal(positions, max_iters=200):
        """Simulated annealing to find tighter packing."""
        n = len(positions)
        best_positions = [p[:] for p in positions]
        best_side = compute_min_side(positions, (0.0, 0.0), 0.0)
        
        temp = 1.0
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            new_positions = [p[:] for p in positions]
            idx = random.randint(0, n-1)
            new_positions[idx][0] += random.gauss(0, 0.3 * temp)
            new_positions[idx][1] += random.gauss(0, 0.3 * temp)
            new_positions[idx][2] += random.gauss(0, 10 * temp)
            new_positions[idx][2] %= 60
            
            # Relax
            new_positions = relax_positions(new_positions, max_iters=20)
            
            # Evaluate
            new_side = compute_min_side(new_positions, (0.0, 0.0), 0.0)
            
            # Accept if better or with probability
            if new_side < best_side or random.random() < math.exp(-(new_side - best_side) / (temp + 1e-10)):
                positions = new_positions
                if new_side < best_side:
                    best_side = new_side
                    best_positions = [p[:] for p in positions]
            
            temp *= cooling
        
        return best_positions, best_side

    # Initial guess: hexagonal lattice arrangement
    # 11 hexagons: center + 6 in first ring + 4 in second ring
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
    
    init_positions = []
    for q, r in lattice_coords:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        init_positions.append([x, y, 0.0])
    
    # Relax initial positions
    relaxed_positions = relax_positions(init_positions, max_iters=50)
    
    best_side = float('inf')
    best_positions = relaxed_positions
    best_angle = 0.0
    
    # Multi-start with annealing
    num_starts = 8
    for start in range(num_starts):
        if time.time() > deadline - 20:
            break
        
        # Perturb initial guess
        positions = [p[:] for p in init_positions]
        for p in positions:
            p[0] += random.gauss(0, 0.8)
            p[1] += random.gauss(0, 0.8)
            p[2] = random.uniform(0, 60)
        
        # Anneal
        annealed, side = anneal(positions, max_iters=150)
        
        if side < best_side:
            best_side = side
            best_positions = annealed
    
    # Fine-tune outer angle
    final_side = float('inf')
    final_angle = 0.0
    for outer_angle in np.arange(0, 60, 1.0):
        if time.time() > deadline - 2:
            break
        side = compute_min_side(best_positions, (0.0, 0.0), outer_angle)
        if side < final_side:
            final_side = side
            final_angle = outer_angle
    
    # Final relaxation with best angle
    best_positions = relax_positions(best_positions, max_iters=20)
    
    return best_positions, (0.0, 0.0), final_side, final_angle
# EVOLVE-BLOCK-END
