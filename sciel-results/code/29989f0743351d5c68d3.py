import math
import time
import numpy as np
import itertools
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at angles 0, 60, ..., 300 degrees
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
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    def solve_contact_graph(contact_pairs, angles, outer_angle=0.0):
        """
        Solve for positions given a contact graph and fixed angles.
        Returns positions or None if inconsistent.
        """
        # Fix hex 0 at (0,0)
        # Variables: x_i, y_i for i=1..N-1 (2*(N-1) vars)
        # Constraints:
        # 1. For each contact pair (i, j), dist(p_i, p_j) = 2.0 (since side=1, distance between centers of touching hexes is 2)
        #    Actually, for regular hexagons of side 1, the distance between centers when touching is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)?
        #    Wait. Side length s=1.
        #    Apothem a = s * sqrt(3) / 2.
        #    Distance between centers of two touching hexagons is 2 * a = sqrt(3).
        #    Let's verify.
        #    Hexagon vertices: (1,0), (1/2, sqrt(3)/2), ...
        #    Center to vertex is 1.
        #    Center to edge midpoint is sqrt(3)/2.
        #    If they touch edge-to-edge, the distance between centers is 2 * (sqrt(3)/2) = sqrt(3).
        #    If they touch vertex-to-vertex (unlikely in optimal packing without rotation), it would be different.
        #    Assuming edge-to-edge contact for the graph definition.
        
        # We use least squares to solve for positions.
        # Since it's a rigid body, we can fix x_0=0, y_0=0.
        # We have N-1 points to find.
        # Each contact gives a constraint: (x_i - x_j)^2 + (y_i - y_j)^2 = 3.
        # This is non-linear. We linearize or use iterative refinement.
        # However, for a "rigid" graph, we can often just do a force-directed relaxation or simple algebra if the graph is a tree.
        # For general graphs, we use a simple iterative solver (Gauss-Seidel style).
        
        # Initialize positions randomly or on a lattice
        positions = np.zeros((N, 2))
        # Simple random init for non-zero nodes
        np.random.seed(42)
        for i in range(1, N):
            positions[i] = np.random.uniform(-1, 1, 2)
            
        # Iterative refinement
        for _ in range(50):
            for i in range(1, N):
                # Sum of forces from contacts
                fx = 0
                fy = 0
                count = 0
                for (a, b) in contact_pairs:
                    if a == i:
                        j = b
                    elif b == i:
                        j = a
                    else:
                        continue
                    # Vector from i to j
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9:
                        continue
                    # We want dist to be sqrt(3).
                    # Force direction: towards desired distance
                    # Simple gradient step
                    # f = (dist - sqrt(3)) / dist
                    # Move i towards/away from j
                    target_dist = math.sqrt(3)
                    error = dist - target_dist
                    # Move i by 0.5 * error * unit_vector (sharing responsibility)
                    # Actually, for i, if dist > target, move towards j. If dist < target, move away.
                    # Unit vector from i to j is (dx, dy)/dist.
                    # If dist > target, we want to reduce dist. Move i towards j.
                    # Step = 0.5 * error * (dx, dy)/dist
                    step_x = 0.5 * error * (dx / dist)
                    step_y = 0.5 * error * (dy / dist)
                    positions[i][0] += step_x
                    positions[i][1] += step_y
                    count += 1
                # If no contacts, leave as is (or drift to center? No, keep)
                
        # Check if solution is valid (all contacts satisfied)
        for (a, b) in contact_pairs:
            dx = positions[a][0] - positions[b][0]
            dy = positions[a][1] - positions[b][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if abs(dist - math.sqrt(3)) > 0.1: # Tolerance
                return None
        
        # Center the configuration
        positions -= np.mean(positions, axis=0)
        
        # Check disjointness
        if not is_valid(positions, angles):
            return None
            
        return positions

    # --- Strategy: Topological Search ---
    
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Candidate angles for each hexagon.
    # In optimal packings of hexagons, angles are often multiples of 60 or 30.
    # Let's try a few discrete angle sets.
    angle_sets = [
        [0.0] * N,
        [0.0 if i % 2 == 0 else 60.0 for i in range(N)],
        [0.0] * N, # Try rotating outer
    ]
    
    # Generate candidate contact graphs.
    # We don't enumerate ALL graphs (too many). We use heuristics.
    # 1. Honeycomb lattice subset.
    # 2. Random graphs with degree constraints.
    
    # Heuristic 1: Honeycomb lattice contacts
    # Generate lattice points
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Select 12 points
    selected_indices = list(range(12))
    selected_points = [lattice_points[i] for i in selected_indices]
    
    # Build contact graph based on proximity in lattice
    contacts = []
    for i in range(12):
        for j in range(i+1, 12):
            dx = selected_points[i][0] - selected_points[j][0]
            dy = selected_points[i][1] - selected_points[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if abs(dist - 1.5) < 0.1: # Lattice neighbor distance is 1.5? 
                # Wait, lattice points are centers.
                # Distance between centers of touching hexes is sqrt(3) approx 1.732.
                # In hex lattice, distance between adjacent centers is sqrt(3).
                # Let's check lattice generation:
                # i=1, j=0 -> x=1.5, y=sqrt(3)/2. dist = sqrt(2.25 + 0.75) = sqrt(3). Correct.
                if abs(dist - sqrt3) < 0.1:
                    contacts.append((i, j))
    
    # Try solving this graph with various angles
    for angles in angle_sets:
        if time.time() > deadline - 20: break
        pos = solve_contact_graph(contacts, angles)
        if pos is not None:
            # Optimize outer angle
            best_local_side = float('inf')
            best_local_outer = 0
            for oa in [0, 30]:
                s = calc_min_outer_side(pos, angles, oa)
                if s < best_local_side:
                    best_local_side = s
                    best_local_outer = oa
            if best_local_side < best_score:
                best_score = best_local_side
                best_positions = pos
                best_angles = angles
                best_outer_angle = best_local_outer
    
    # Heuristic 2: Random Graphs
    # Generate random graphs with 12 nodes, ensuring connectivity and degree <= 3 (hex max neighbors)
    np.random.seed(123)
    for _ in range(200):
        if time.time() > deadline - 10: break
        
        # Generate random edges
        edges = []
        # Ensure connectivity: random spanning tree
        nodes = list(range(12))
        random.shuffle(nodes)
        for i in range(1, 12):
            # Connect to a random previous node
            j = random.randint(0, i-1)
            edges.append((nodes[i], nodes[j]))
        
        # Add random extra edges up to degree 3
        for i in range(12):
            # Count current degree
            deg = sum(1 for e in edges if i in e)
            while deg < 3 and random.random() < 0.5:
                # Pick random neighbor
                j = random.randint(0, 11)
                if i == j: continue
                if (i, j) in edges or (j, i) in edges: continue
                edges.append((i, j))
                deg += 1
        
        # Solve
        for angles in angle_sets[:1]: # Use uniform angles for speed in random search
            pos = solve_contact_graph(edges, angles)
            if pos is not None:
                s = calc_min_outer_side(pos, angles, 0)
                if s < best_score:
                    best_score = s
                    best_positions = pos
                    best_angles = angles
                    best_outer_angle = 0

    # Fallback if no valid solution found
    if best_positions is None:
        # Use the honeycomb subset directly without solving (just place them)
        # This might have overlaps, but we force separation by scaling? No, scale is fixed.
        # We must find a valid configuration.
        # Let's try a simple ring configuration.
        r = 2.0
        positions = []
        angles = []
        for k in range(6):
            theta = k * math.pi / 3
            positions.append([r * math.cos(theta), r * math.sin(theta)])
            angles.append(math.degrees(theta))
        r2 = 4.0
        for k in range(6):
            theta = k * math.pi / 3 + math.pi/6
            positions.append([r2 * math.cos(theta), r2 * math.sin(theta)])
            angles.append(math.degrees(theta))
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        if not is_valid(positions, angles):
            # Force separation by moving outward
            for i in range(N):
                positions[i] *= 1.5
            if not is_valid(positions, angles):
                # Last resort: grid
                positions = np.array([
                    [0, 0], [2.1, 0], [-2.1, 0],
                    [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
                    [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
                    [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
                ])
                angles = np.zeros(N)
        
        best_positions = positions
        best_angles = angles
        best_score = calc_min_outer_side(positions, angles, 0)
        best_outer_angle = 0

    # Local refinement using SLSQP on the best topological solution
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            
            def objective(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:-1]
                outer_angle = params[-1]
                return calc_min_outer_side(positions, angles, outer_angle)
            
            def constraint_func(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:-1]
                # SAT constraints
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                cons = []
                for i in range(N):
                    for j in range(i+1, N):
                        p1 = polys[i]
                        p2 = polys[j]
                        axes = []
                        for k in range(6):
                            e1 = p1[(k+1)%6] - p1[k]
                            n1 = np.array([-e1[1], e1[0]])
                            axes.append(n1)
                            e2 = p2[(k+1)%6] - p2[k]
                            n2 = np.array([-e2[1], e2[0]])
                            axes.append(n2)
                        
                        min_separation = float('inf')
                        for axis in axes:
                            norm = np.linalg.norm(axis)
                            if norm < 1e-9: continue
                            axis = axis / norm
                            proj1 = p1 @ axis
                            proj2 = p2 @ axis
                            min1, max1 = np.min(proj1), np.max(proj1)
                            min2, max2 = np.min(proj2), np.max(proj2)
                            sep = min(max1, max2) - max(min1, min2)
                            if sep < min_separation:
                                min_separation = sep
                        cons.append(min_separation)
                return np.array(cons)
            
            x0 = np.concatenate([best_positions.flatten(), best_angles, [best_outer_angle]])
            bounds = [(-10, 10)] * (2*N) + [(0, 360)] * N + [(0, 60)]
            cons = {'type': 'ineq', 'fun': constraint_func}
            
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 50, 'ftol': 1e-6, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                final_positions = res.x[:2*N].reshape(N, 2)
                final_angles = res.x[2*N:-1]
                final_outer_angle = res.x[-1]
                if is_valid(final_positions, final_angles):
                    best_score = res.fun
                    best_positions = final_positions
                    best_angles = final_angles
                    best_outer_angle = final_outer_angle
        except Exception:
            pass

    final_side = calc_min_outer_side(best_positions, best_angles, best_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), final_side, best_outer_angle

if __name__ == "__main__":
    construct()
