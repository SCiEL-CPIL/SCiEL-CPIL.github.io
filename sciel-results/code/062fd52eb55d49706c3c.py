import numpy as np
import time

N = 12

def get_hex_vertices(cx, cy, angle_rad):
    # Regular hexagon side 1
    # Vertices at distance 1 from center
    angles = angle_rad + np.arange(6) * (np.pi / 3)
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def get_normals(vertices):
    # vertices: 6x2
    # edges: v[i+1] - v[i]
    edges = np.roll(vertices, -1, axis=0) - vertices
    # normals: perpendicular to edges, pointing inward or outward?
    # For SAT, direction doesn't matter for gap, but for penetration depth we need consistent orientation.
    # Let's use outward normals.
    # Normal to (dx, dy) is (dy, -dx) or (-dy, dx).
    # Let's calculate cross product with center to ensure outward?
    # Simpler: Just use the vector perpendicular to edge.
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    # Normalize
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]
    # Ensure normals point away from center (0,0) of the polygon?
    # Midpoint of edge
    mids = (vertices + np.roll(vertices, -1, axis=0)) / 2
    # Dot product with normal
    dots = np.sum(mids * normals, axis=1)
    # If dot < 0, flip
    # Wait, vertices are absolute coords. Center is not (0,0) in world space.
    # We need center of the polygon.
    return normals

def project(vertices, normal):
    # vertices: Nx2, normal: 2
    projections = np.dot(vertices, normal)
    return np.min(projections), np.max(projections)

def sat_overlap_penalty(verts1, verts2):
    # Returns 0 if no overlap, else penetration depth
    min_overlap = float('inf')
    
    # Get normals from both polygons
    # For regular hexagons, we can just use the 6 axes of symmetry?
    # No, relative rotation matters.
    # Get edges of poly1 and poly2
    
    edges1 = np.roll(verts1, -1, axis=0) - verts1
    edges2 = np.roll(verts2, -1, axis=0) - verts2
    
    normals1 = np.column_stack((edges1[:, 1], -edges1[:, 0]))
    normals2 = np.column_stack((edges2[:, 1], -edges2[:, 0]))
    
    all_normals = np.vstack((normals1, normals2))
    
    for n in all_normals:
        n_norm = np.linalg.norm(n)
        if n_norm == 0: continue
        n = n / n_norm
        
        min1, max1 = project(verts1, n)
        min2, max2 = project(verts2, n)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return 0.0 # No overlap
        
        # Calculate overlap width
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def boundary_penalty(verts, S, outer_center=(0.0, 0.0), outer_angle=0.0):
    # Outer hexagon side S
    # Vertices of outer hexagon
    # Center (0,0)
    # Distance from center to side (apothem) of outer hex:
    # S * sqrt(3) / 2
    # Wait, side length S.
    # Apothem a = S * sqrt(3) / 2
    # The boundary is defined by 6 lines.
    # Equation: x cos(theta) + y sin(theta) <= a
    # Where theta are the angles of the normals to the sides.
    
    a = S * np.sqrt(3) / 2.0
    # Normals angles
    thetas = outer_angle + np.arange(6) * (np.pi / 3)
    # Wait, the normals to the sides of a regular hexagon are at angles 0, 60, 120...
    # If a vertex is at angle 0, the side opposite is at 180?
    # Let's define the outer hexagon such that one side is horizontal at y = -a?
    # Standard orientation:
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # Sides are between these.
    # The normal to the side between vertex 0 and 1 is at 30 deg?
    # Let's just use the support function.
    # A point (x,y) is inside if for all 6 normals n_k:
    # dot((x,y)-center, n_k) <= a
    # The normals are at angles 30, 90, 150, 210, 270, 330?
    # Or 0, 60, 120...?
    # If vertices are at 0, 60..., the sides are centered at 30, 90...
    # So normals are at 30, 90, 150, 210, 270, 330.
    
    normal_angles = outer_angle + (np.pi / 6) + np.arange(6) * (np.pi / 3)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Check all vertices of inner hex
    # verts: 6x2
    # Translate to outer center
    verts_rel = verts - np.array(outer_center)
    
    # Project verts onto normals
    # projections: 6x6
    projections = np.dot(verts_rel, normals.T)
    
    # For each vertex (row), it must satisfy all 6 inequalities (cols)
    # projection[i, j] <= a
    # Violation = max(0, projection[i, j] - a)
    violations = np.maximum(0, projections - a)
    
    # Sum of squared violations
    return np.sum(violations ** 2)

def total_cost(positions, angles, S):
    # positions: Nx2
    # angles: Nx1 (radians)
    cost = 0.0
    
    # Precompute vertices for all hexes
    verts_list = []
    for i in range(N):
        v = get_hex_vertices(positions[i, 0], positions[i, 1], angles[i])
        verts_list.append(v)
        
    # Overlaps
    for i in range(N):
        for j in range(i + 1, N):
            cost += sat_overlap_penalty(verts_list[i], verts_list[j]) ** 2
            
    # Boundary
    for i in range(N):
        cost += boundary_penalty(verts_list[i], S)
        
    return cost

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Initialize
    # Grid layout
    # 4 cols, 3 rows
    dx = np.sqrt(3)
    dy = 3.0 # Safe distance
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    idx = 0
    for r in range(3):
        for c in range(4):
            positions[idx, 0] = (c - 1.5) * dx
            positions[idx, 1] = (r - 1.0) * dy
            angles[idx] = 0.0
            idx += 1
            
    # Initial S
    # Max distance from origin to any vertex
    max_r = 0
    for i in range(N):
        v = get_hex_vertices(positions[i,0], positions[i,1], angles[i])
        max_r = max(max_r, np.max(np.linalg.norm(v, axis=1)))
    # S is related to max_r
    # For a hexagon, R = S. So S_init = max_r + 1.0
    S = max_r + 1.0
    
    best_cost = float('inf')
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_S = S
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    step_size = 0.5
    
    iterations = 0
    while time.time() - start_time < time_limit:
        # Propose move
        i = np.random.randint(N)
        move_type = np.random.rand()
        
        new_positions = positions.copy()
        new_angles = angles.copy()
        
        if move_type < 0.5:
            # Translate
            new_positions[i, 0] += np.random.normal(0, step_size)
            new_positions[i, 1] += np.random.normal(0, step_size)
        elif move_type < 0.8:
            # Rotate
            new_angles[i] += np.random.normal(0, step_size * 0.5)
        else:
            # Shrink/Expand cluster?
            # Just move a few random ones
            for k in range(3):
                idx_k = np.random.randint(N)
                new_positions[idx_k, 0] += np.random.normal(0, step_size * 0.5)
                new_positions[idx_k, 1] += np.random.normal(0, step_size * 0.5)
                
        # Calculate Cost
        c_new = total_cost(new_positions, new_angles, S)
        
        # Acceptance
        if c_new < best_cost:
            best_cost = c_new
            best_positions = new_positions.copy()
            best_angles = new_angles.copy()
            # If valid, try to reduce S
            if c_new < 1e-6:
                S -= 0.05
                best_S = S
                # Reset cost tracking? No, keep current state
                # But we need to update positions to the best valid ones
                positions = best_positions.copy()
                angles = best_angles.copy()
                # Continue SA with new S
                # Note: total_cost uses S. If we change S, cost changes.
                # We should re-evaluate or just let the next loop handle it.
                # Actually, if we change S, the current positions might be invalid.
                # The cost function will reflect that.
                pass
            else:
                positions = new_positions
                angles = new_angles
        else:
            delta = c_new - best_cost
            if np.random.rand() < np.exp(-delta / T):
                positions = new_positions
                angles = new_angles
                # Don't update best_cost if worse, but we might accept it to explore
                # However, for minimizing S, we need to be careful.
                # If we accept a worse cost, we might drift away from valid packing.
                # Usually, we accept if it's better or with prob.
                # Here "better" means lower cost.
                # If we accept worse cost, we are exploring.
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            # Reheat if stuck?
            if best_cost > 1e-6:
                T = 0.5
                step_size *= 1.1
            
        iterations += 1
        if iterations % 1000 == 0:
            # Check time
            if time.time() - start_time > time_limit - 5:
                break

    return [ [best_positions[i,0], best_positions[i,1], np.degrees(best_angles[i])] for i in range(N) ], (0.0, 0.0), best_S, 0.0

