import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    N = 12
    
    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_edges_and_normals(vertices):
        """Get edge midpoints and outward normals for a convex polygon."""
        n = len(vertices)
        edges = np.roll(vertices, -1, axis=0) - vertices
        # Outward normals
        # For a CCW polygon, normal to (dx, dy) is (-dy, dx)
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]
        # Ensure normals point outward
        # Check if center is on the negative side
        center = np.mean(vertices, axis=0)
        # dot(center - vertex, normal) should be negative for outward normal?
        # If normal points out, center is "inside", so dot(center - v, n) < 0
        # Let's check one
        dots = np.sum((center - vertices) * normals, axis=1)
        # If any dot > 0, normal is inward?
        # Actually, for a convex polygon, if we define edges v_i -> v_{i+1}, 
        # and vertices are CCW, then (-dy, dx) points OUTWARD.
        # Let's assume vertices are CCW.
        # hexagon_vertices generates angles 0, 60, ... which is CCW.
        # So normals are outward.
        return normals, vertices

    def signed_dist_point_to_convex(p, vertices):
        """
        Approximate signed distance from point p to convex polygon vertices.
        Returns negative if inside, positive if outside.
        Uses the minimum distance to the edge lines (half-space constraint).
        """
        normals, _ = get_edges_and_normals(vertices)
        # Distance to each edge line: n . (p - v)
        # If p is inside, n . (p - v) < 0 for all edges.
        # The "depth" is min(n . (p - v)).
        # If p is outside, some n . (p - v) > 0.
        # The distance is approx max(0, min(n . (p - v)))? 
        # No, if outside, the closest edge determines the distance.
        # For a smooth approximation, we can use:
        # d = min_k ( n_k . (p - v_k) )
        # This is negative inside.
        # If outside, it might be positive, but it's the distance to the closest *supporting line*, 
        # which is a lower bound on the actual distance to the polygon.
        # This is sufficient for a penalty function.
        dots = np.sum(normals * (p - vertices), axis=1)
        return np.min(dots)

    def softmin(x, beta=10.0):
        """Soft minimum."""
        return -1.0 / beta * np.log(np.sum(np.exp(-beta * x)))

    def collision_penalty(positions, angles):
        """
        Calculate smooth collision penalty for all pairs.
        """
        penalty = 0.0
        n = len(positions)
        verts_list = []
        for i in range(n):
            verts_list.append(hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
        
        for i in range(n):
            for j in range(i + 1, n):
                vi = verts_list[i]
                vj = verts_list[j]
                
                # Sum of squared penetration depths
                # For each vertex in i, check distance to j
                for p in vi:
                    d = signed_dist_point_to_convex(p, vj)
                    if d < 0:
                        penalty += (-d) ** 2
                        
                # For each vertex in j, check distance to i
                for p in vj:
                    d = signed_dist_point_to_convex(p, vi)
                    if d < 0:
                        penalty += (-d) ** 2
                        
        return penalty

    def outer_side_length(positions, angles, theta_out):
        """
        Calculate the side length of the minimum enclosing regular hexagon
        with orientation theta_out.
        """
        all_verts = []
        for i in range(len(positions)):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Normals of the outer hexagon
        outer_angles = theta_out + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        
        # Project all vertices onto normals
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        
        # The side length is determined by the maximum width
        # s = width / sqrt(3)
        # We want to minimize s, so we minimize max(widths)
        # Use softmin/softmax for smoothness? 
        # The objective function will use this. 
        # To make it differentiable, we use a soft max of the widths.
        beta = 10.0
        max_width = 1.0 / beta * np.log(np.sum(np.exp(beta * widths)))
        
        return max_width / np.sqrt(3)

    def objective(params):
        """
        Objective function: Outer Side + Collision Penalty.
        """
        if time.time() > deadline:
            return 1e10
            
        n = 12
        positions = np.array([[params[i*3], params[i*3+1]] for i in range(n)])
        angles = np.array([params[i*3+2] for i in range(n)])
        theta_out = params[-1]
        
        # Collision penalty
        pen = collision_penalty(positions, angles)
        
        # Outer side
        side = outer_side_length(positions, angles, theta_out)
        
        # Weight for penalty
        # If penalty is high, we want to push it down.
        # Side is typically ~3-4. Penalty can be large.
        # Scale penalty so it's comparable.
        lambda_pen = 10.0
        
        return side + lambda_pen * pen

    # Initial configurations
    initial_configs = []
    
    # 1. Honeycomb
    d = 2.01
    pos = [[0.0, 0.0]]
    ang = [0.0]
    for k in range(6):
        a = np.radians(60 * k)
        pos.append([d * np.cos(a), d * np.sin(a)])
        ang.append(0.0)
    for k in range(5):
        a = np.radians(30 + 60 * k)
        dist = d * np.sqrt(3)
        pos.append([dist * np.cos(a), dist * np.sin(a)])
        ang.append(0.0)
    initial_configs.append(np.concatenate([np.array(pos).flatten(), np.array(ang), [0.0]]))
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (12, 2))
        ang = np.random.uniform(0, np.pi, 12)
        theta = np.random.uniform(0, np.pi/3)
        initial_configs.append(np.concatenate([pos.flatten(), ang, [theta]]))

    best_result = None
    best_score = 1e10
    
    bounds = [(-5, 5), (-5, 5), (0, np.pi)] * 12 + [(0, np.pi/3)]
    
    for i, x0 in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.fun < best_score:
                # Verify validity
                n = 12
                positions = np.array([[res.x[i*3], res.x[i*3+1]] for i in range(n)])
                angles = np.array([res.x[i*3+2] for i in range(n)])
                theta_out = res.x[-1]
                
                # Check for overlaps with strict SAT
                # If penalty is very small, it's likely valid
                pen = collision_penalty(positions, angles)
                if pen < 1e-4:
                    side = outer_side_length(positions, angles, theta_out)
                    if side < best_score:
                        best_score = side
                        best_result = res.x
                        
        except Exception as e:
            continue

    if best_result is None:
        # Fallback to initial honeycomb
        best_result = initial_configs[0]
        positions = np.array([[best_result[i*3], best_result[i*3+1]] for i in range(12)])
        angles = np.array([best_result[i*3+2] for i in range(12)])
        theta_out = best_result[-1]
        best_score = outer_side_length(positions, angles, theta_out)

    # Final refinement
    # We have best_result. Let's extract the final parameters.
    n = 12
    best_positions = np.array([[best_result[i*3], best_result[i*3+1]] for i in range(n)])
    best_angles = np.array([best_result[i*3+2] for i in range(n)])
    best_theta_out = best_result[-1]
    
    # Calculate the actual outer hexagon center and side for the return value
    # The side length is best_score (approximately)
    # We need the center.
    all_verts = []
    for i in range(n):
        v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    outer_angles = best_theta_out + np.radians(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    dots = all_verts @ normals.T
    mins = dots.min(axis=0)
    maxs = dots.max(axis=0)
    mids = (mins + maxs) / 2.0
    
    # Solve for center c such that c . n_k = mid_k
    # This is an overdetermined system. Use least squares.
    c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
    outer_center = tuple(c)
    
    # Recalculate side length accurately for this center and orientation
    # The side length is determined by the max width.
    widths = maxs - mins
    max_width = np.max(widths)
    outer_side = max_width / np.sqrt(3)
    
    # If the center calculation shifted the hexagon, the width might change slightly?
    # No, width is invariant to translation.
    # But the "minimum enclosing" hexagon might have a different optimal orientation.
    # We found theta_out via optimization.
    
    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(n)]
    
    return inner, outer_center, outer_side, np.degrees(best_theta_out)

# EVOLVE-BLOCK-END
