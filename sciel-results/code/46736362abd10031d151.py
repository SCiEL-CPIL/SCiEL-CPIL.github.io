"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    # Helper to get vertices of a unit hexagon centered at (x, y) with rotation angle
    def get_hex_vertices(x, y, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            verts.append((x + math.cos(theta), y + math.sin(theta)))
        return np.array(verts)

    # Helper to check if two convex polygons are disjoint (no overlap)
    # Using Separating Axis Theorem (SAT) is efficient for convex polygons
    def polygons_disjoint(verts1, verts2):
        # Check all edges of both polygons
        for i in range(len(verts1)):
            # Normal to edge i of poly1
            p1 = verts1[i]
            p2 = verts1[(i + 1) % len(verts1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            
            # Project both polygons onto this normal
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they are disjoint
            if max1 < min2 or max2 < min1:
                return True
        
        for i in range(len(verts2)):
            p1 = verts2[i]
            p2 = verts2[(i + 1) % len(verts2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    # Calculate the minimum side length of a regular hexagon (rotated by outer_angle)
    # centered at (cx, cy) that contains all points P
    def min_containing_hexagon_side(points, cx, cy, outer_angle_deg):
        # The width of a regular hexagon with side s in direction theta is:
        # s * (cos(theta) + 0.5 * cos(2*theta)) ... wait, simpler:
        # The support function of a regular hexagon with side s and rotation phi in direction u is:
        # h(u) = s * max_{v in vertices} dot(u, v)
        # Vertices of unit hexagon (side 1, rot 0) are at angles 0, 60, ...
        # Actually, let's just compute the required radius R for the circumcircle? No, we need side length.
        # Let's use the property: Distance from center to edge (apothem) a = s * sqrt(3)/2.
        # The hexagon contains a point p if for all 6 normals n_k of the hexagon, dot(p - center, n_k) <= a.
        # So a >= max_k dot(p - center, n_k).
        # Thus s >= 2/sqrt(3) * max_k dot(p - center, n_k).
        
        outer_angle_rad = math.radians(outer_angle_deg)
        normals = []
        for i in range(6):
            # Normals to the edges of a regular hexagon
            # Edges are at angles 30, 90, ... relative to the vertex angles 0, 60...
            # If vertices are at angle phi + i*pi/3, the outward normal to the edge between i and i+1 is phi + i*pi/3 + pi/6
            angle = outer_angle_rad + i * math.pi / 3 + math.pi / 6
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_proj = 0.0
        for p in points:
            px, py = p
            dx = px - cx
            dy = py - cy
            for nx, ny in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
        
        # max_proj is the required apothem
        # s = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3.0)

    # Objective function: Minimize outer side length
    # State: [x1, y1, a1, ..., x12, y12, a12, cx, cy, outer_angle]
    # We fix outer center to (0,0) initially? No, let it move? 
    # Due to symmetry, we can fix the center of mass or just fix cx, cy = 0 to reduce dimensions?
    # Better to fix cx, cy = 0.0 and just optimize outer_angle and inner positions.
    # Why? Shifting all inner hexagons and the outer hexagon together doesn't change the relative geometry.
    # So we can WLOG set outer center to (0,0).
    
    # Initial Layout: Symmetric "Flower" or Grid
    # 12 hexagons. 
    # Let's try a 3x4 grid or a hexagonal packing.
    # Hexagonal packing of 12:
    # Center 1, Ring 1 (6), Ring 2 (5 more to make 12? No, 1+6+12=19. 1+6=7. Need 5 more.)
    # Let's just use a simple grid initialization and let SA fix it.
    # Grid 3x4
    positions = []
    angles = []
    for i in range(3):
        for j in range(4):
            # Staggered grid for hexagons
            x = j * 1.8 # Approx distance
            y = i * 1.6
            if j % 2 == 1:
                y += 0.8
            positions.append([x, y])
            angles.append(0.0)
            
    # Center the grid
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_avg = sum(xs) / len(xs)
    cy_avg = sum(ys) / len(ys)
    for p in positions:
        p[0] -= cx_avg
        p[1] -= cy_avg

    state = []
    for i in range(N):
        state.extend([positions[i][0], positions[i][1], angles[i]])
    # Add outer angle
    state.append(0.0)
    
    # We will optimize: inner x,y,angle (36 vars) + outer angle (1 var) = 37 vars
    # Outer center is fixed at (0,0)
    
    def evaluate(state_vec):
        # Extract params
        inner_params = state_vec[:3*N]
        outer_angle = state_vec[3*N]
        
        # Get vertices
        all_verts = []
        for i in range(N):
            x = inner_params[i*3]
            y = inner_params[i*3+1]
            a = inner_params[i*3+2]
            verts = get_hex_vertices(x, y, a)
            all_verts.append(verts)
            
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not polygons_disjoint(all_verts[i], all_verts[j]):
                    # Penalty for overlap
                    # Calculate minimum translation vector or just a large penalty?
                    # SAT gives the axis of separation. If not disjoint, find min overlap depth.
                    # For simplicity, return a large penalty if overlap.
                    # To make SA work, we need a smooth penalty.
                    # Let's calculate the minimum distance to separate them? 
                    # That's expensive. 
                    # Alternative: If overlap, return a penalty based on how much they overlap.
                    # Since SAT says they overlap, let's just return a large constant penalty.
                    # This makes the landscape very rugged. 
                    # Better: Use a penalty that grows with overlap.
                    # Let's estimate overlap depth.
                    # For each axis, if projections overlap, depth = min(max1, max2) - max(min1, min2).
                    # Sum of depths? Or max depth?
                    min_depth = float('inf')
                    # Re-check axes to find depth
                    for i_v in range(len(all_verts[i])):
                        p1 = all_verts[i][i_v]
                        p2 = all_verts[i][(i_v + 1) % len(all_verts[i])]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        proj1 = np.dot(all_verts[i], normal)
                        proj2 = np.dot(all_verts[j], normal)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        if max1 >= min2 and max2 >= min1:
                            depth = min(max1, max2) - max(min1, min2)
                            if depth < min_depth:
                                min_depth = depth
                    # Penalty is proportional to depth squared
                    return min_depth ** 2 + 1000.0 # Large offset
                    
        # If disjoint, calculate containing hexagon side
        # Points to contain: all vertices of all inner hexagons
        points = []
        for v in all_verts:
            points.extend(v)
        
        side = min_containing_hexagon_side(points, 0.0, 0.0, outer_angle)
        return side

    # Simulated Annealing
    current_state = state
    current_score = evaluate(current_state)
    best_state = current_state[:]
    best_score = current_score
    
    T = 1.0
    cooling = 0.999
    step_size = 1.0
    
    while time.time() < deadline:
        # Perturb state
        new_state = current_state[:]
        # Randomly pick a few variables to perturb
        num_perturb = random.randint(1, 3)
        for _ in range(num_perturb):
            idx = random.randint(0, len(new_state)-1)
            # Gaussian perturbation
            # Scale perturbation based on variable type?
            # Positions: ~1.0
            # Angles: ~10 degrees
            if idx < 3*N:
                # Check if angle index
                if idx % 3 == 2:
                    new_state[idx] += random.gauss(0, step_size * 5.0)
                else:
                    new_state[idx] += random.gauss(0, step_size)
            else:
                # Outer angle
                new_state[idx] += random.gauss(0, step_size * 5.0)
                
        new_score = evaluate(new_state)
        
        # Acceptance criterion
        if new_score < current_score:
            current_state = new_state
            current_score = new_score
            if current_score < best_score:
                best_state = new_state[:]
                best_score = current_score
        else:
            delta = new_score - current_score
            if T > 0:
                p = math.exp(-delta / T)
                if random.random() < p:
                    current_state = new_state
                    current_score = new_score
                    
        T *= cooling
        # If T is very small, increase step size slightly to explore? No, usually decrease.
        # But if stuck, maybe increase?
        # Let's keep it simple.
        
        # Check time
        if time.time() > deadline:
            break

    # Extract best result
    inner = []
    for i in range(N):
        x = best_state[i*3]
        y = best_state[i*3+1]
        a = best_state[i*3+2]
        inner.append([x, y, a])
        
    outer_center = (0.0, 0.0)
    outer_side = best_score
    outer_angle = best_state[3*N]
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
