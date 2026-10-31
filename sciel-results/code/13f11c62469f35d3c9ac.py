"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices for angle 0 (pointy-top orientation)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    unit_hex_verts = np.array([
        [math.cos(math.radians(60*k)), math.sin(math.radians(60*k))] for k in range(6)
    ])
    
    def rotate_points(points, angle_rad):
        """Rotate an array of points by angle_rad."""
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return points @ R.T
    
    def get_hex_vertices(center, angle_deg):
        """Returns (6, 2) array of vertices for unit hexagon."""
        cx, cy = center
        angle_rad = math.radians(angle_deg)
        rotated = rotate_points(unit_hex_verts, angle_rad)
        return rotated + np.array([cx, cy])
    
    def support_function_distance(hex1_center, hex1_angle, hex2_center, hex2_angle):
        """
        Compute minimum distance between two unit hexagons using support functions.
        Returns negative if overlapping, positive if separated.
        For two convex polygons, they are disjoint if and only if there exists
        a separating line. We check all edge normals of both hexagons.
        """
        v1 = get_hex_vertices(hex1_center, hex1_angle)
        v2 = get_hex_vertices(hex2_center, hex2_angle)
        
        # Get edge normals for both hexagons
        # Edge normal for edge i is perpendicular to edge (i, i+1)
        def get_edge_normals(verts):
            normals = []
            for i in range(6):
                e = verts[(i+1)%6] - verts[i]
                # Normal pointing outward
                n = np.array([-e[1], e[0]])
                n = n / (np.linalg.norm(n) + 1e-15)
                normals.append(n)
            return normals
        
        n1 = get_edge_normals(v1)
        n2 = get_edge_normals(v2)
        
        # Check separation along each normal direction
        min_sep = float('inf')
        for n in n1 + n2:
            # Project both hexagons onto this normal
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            # Separation along this normal
            sep = min(proj2) - max(proj1)
            if sep < min_sep:
                min_sep = sep
        
        return min_sep
    
    def hexagon_in_container(center, angle_deg, container_center, container_side, container_angle):
        """
        Check if unit hexagon is inside container hexagon.
        Returns 0 if inside, negative if violating.
        """
        v_inner = get_hex_vertices(center, angle_deg)
        
        # Container hexagon vertices
        ccx, ccy = container_center
        ca_rad = math.radians(container_angle)
        # Scale unit hex to container size
        container_verts = rotate_points(unit_hex_verts, ca_rad) * container_side + np.array([ccx, ccy])
        
        # Get container edge normals (inward pointing)
        def get_inward_normals(verts):
            normals = []
            for i in range(6):
                e = verts[(i+1)%6] - verts[i]
                # Inward normal
                n = np.array([e[1], -e[0]])
                n = n / (np.linalg.norm(n) + 1e-15)
                normals.append(n)
            return normals
        
        in_normals = get_inward_normals(container_verts)
        
        # For each inner vertex, check if it's inside all half-planes of container
        min_margin = float('inf')
        for v in v_inner:
            for n in in_normals:
                # Distance from vertex to container edge along inward normal
                # Reference point on container edge
                # Use container center as reference
                dist = np.dot(v - np.array([ccx, ccy]), n)
                # Inradius of container
                inradius = container_side * sqrt3 / 2
                margin = inradius - dist
                if margin < min_margin:
                    min_margin = margin
        
        return min_margin
    
    def objective(params):
        """
        Objective: minimize container side length + penalties for overlaps and boundary violations.
        params: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx, cy, S, theta]
        where (cx,cy) is container center, S is container side, theta is container angle.
        """
        # Extract parameters
        inner_params = params[:3*N]
        cx, cy = params[3*N], params[3*N+1]
        S = params[3*N+2]
        theta = params[3*N+3]
        
        container_center = (cx, cy)
        
        if S < 2.0:
            return 1e6
        
        penalty = 0.0
        
        # Check pairwise overlaps using support function distance
        for i in range(N):
            xi, yi, ai = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
            for j in range(i+1, N):
                xj, yj, aj = inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]
                sep = support_function_distance((xi, yi), ai, (xj, yj), aj)
                if sep < 0:
                    # Overlap penalty - quadratic for smoothness
                    penalty += 50.0 * sep * sep
        
        # Check boundary violations
        for i in range(N):
            xi, yi, ai = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
            margin = hexagon_in_container((xi, yi), ai, container_center, S, theta)
            if margin < 0:
                penalty += 100.0 * margin * margin
        
        return S + penalty
    
    # Multi-start differential evolution
    best_solution = None
    best_score = float('inf')
    
    # Generate multiple initial configurations
    def generate_initial_config(seed):
        """Generate a random initial configuration."""
        rng = np.random.RandomState(seed)
        
        # Place hexagons in a rough hexagonal grid pattern with some randomness
        positions = []
        angles = []
        
        # Try a hexagonal packing pattern
        # 3-4-4 or 4-3-4 arrangement
        pattern = rng.randint(0, 3)
        
        if pattern == 0:  # 3-4-4
            rows = [3, 4, 4]
        elif pattern == 1:  # 4-3-4
            rows = [4, 3, 4]
        else:  # Random scatter
            rows = None
        
        if rows:
            y_offset = 0.0
            for count in rows:
                x_start = -(count - 1) * sqrt3 / 2
                for j in range(count):
                    x = x_start + j * sqrt3 + rng.uniform(-0.2, 0.2)
                    y = y_offset + rng.uniform(-0.2, 0.2)
                    positions.append([x, y])
                    angles.append(rng.uniform(-30, 30))
                y_offset += 1.5
        else:
            for _ in range(N):
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                positions.append([x, y])
                angles.append(rng.uniform(-180, 180))
        
        # Center the configuration
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in positions:
            p[0] -= cx
            p[1] -= cy
        
        # Estimate container size
        max_dist = max(math.hypot(p[0], p[1]) for p in positions)
        S_est = max_dist + 1.5
        
        return positions, angles, S_est
    
    bounds = []
    for i in range(N):
        bounds.append((-6, 6))  # x
        bounds.append((-6, 6))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-6, 6))  # container cx
    bounds.append((-6, 6))  # container cy
    bounds.append((2.5, 12.0))  # container side
    bounds.append((-180, 180))  # container angle
    
    # Run multiple starts
    n_starts = 5
    for start_idx in range(n_starts):
        pos_init, ang_init, S_init = generate_initial_config(42 + start_idx)
        
        x0 = []
        for i in range(N):
            x0.append(pos_init[i][0])
            x0.append(pos_init[i][1])
            x0.append(ang_init[i])
        x0.append(0.0)  # container cx
        x0.append(0.0)  # container cy
        x0.append(S_init)
        x0.append(0.0)  # container angle
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.8,
                seed=1000 + start_idx,
                polish=True
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
        except Exception as e:
            pass
    
    if best_solution is None:
        # Fallback: simple grid arrangement
        inner = []
        rows = [3, 4, 4]
        y_offset = 0.0
        for count in rows:
            x_start = -(count - 1) * sqrt3 / 2
            for j in range(count):
                x = x_start + j * sqrt3
                y = y_offset
                inner.append([x, y, 0.0])
            y_offset += 1.5
        # Center
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in inner:
            p[0] -= cx
            p[1] -= cy
        max_dist = max(math.hypot(p[0], p[1]) for p in inner)
        return inner, (0.0, 0.0), max_dist + 1.5, 0.0
    
    # Extract solution
    inner = []
    for i in range(N):
        x = best_solution[3*i]
        y = best_solution[3*i+1]
        a = best_solution[3*i+2]
        inner.append([x, y, a])
    
    cx = best_solution[3*N]
    cy = best_solution[3*N+1]
    S = best_solution[3*N+2]
    theta = best_solution[3*N+3]
    
    return inner, (cx, cy), S, theta
# EVOLVE-BLOCK-END
