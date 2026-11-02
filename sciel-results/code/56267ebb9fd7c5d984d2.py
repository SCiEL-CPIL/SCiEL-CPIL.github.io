"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define hexagon vertices for a unit regular hexagon centered at origin
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        # Unit hexagon: circumradius = 1
        angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack([xs, ys])
    
    # Outer hexagon: compute required side length given inner hexagon configurations
    def outer_side_length(config):
        # config: array of shape (N, 3) - [x, y, angle_deg]
        # Outer hexagon centered at origin, we compute the minimum S such that all inner vertices fit
        # For outer hexagon with side S, inradius = S*sqrt(3)/2
        # The 6 normal directions (to edge midpoints) for outer hex with one flat side up:
        # Actually, let's define outer hex with vertices at angles 0, 60, 120, ... degrees
        # Then edge midpoints are at 30, 90, 150, ... degrees
        # The constraint is: for each vertex p of inner hexes, max_i (p . n_i) <= S*sqrt(3)/2
        # where n_i are unit vectors to edge midpoints
        
        # Let's use outer hex with one vertex pointing right (angle 0)
        # Edge midpoint directions: 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for i in range(N):
            x, y, ang = config[i]
            verts = hex_vertices(x, y, ang)
            # Project each vertex onto each normal direction
            projections = verts @ normals.T  # shape (6, 6)
            max_proj = max(max_proj, np.max(projections))
        
        # S*sqrt(3)/2 = max_proj => S = 2*max_proj/sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)
    
    # Check if two hexagons overlap using minimum distance between vertices and edges
    # Simplified: use minimum distance between all vertex pairs, but that's not sufficient for convex shapes
    # Better: for convex polygons, they overlap if and only if there's no separating axis
    # For efficiency, we'll use a penalty based on minimum penetration depth approximation
    
    def hexagon_overlap_penalty(hex1_config, hex2_config):
        """Return penalty if two hexagons overlap, 0 if disjoint."""
        verts1 = hex_vertices(hex1_config[0], hex1_config[1], hex1_config[2])
        verts2 = hex_vertices(hex2_config[0], hex2_config[1], hex2_config[2])
        
        # For two convex polygons, check if they intersect using SAT (Separating Axis Theorem)
        # Get all edges from both polygons
        edges = []
        for i in range(6):
            edges.append((verts1[i], verts1[(i+1)%6]))
            edges.append((verts2[i], verts2[(i+1)%6]))
        
        # For each edge, get the normal direction
        for e1, e2 in edges:
            edge_vec = e2 - e1
            # Normal to edge
            normal = np.array([-edge_vec[1], edge_vec[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            
            # Project all vertices of both hexagons onto this normal
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0  # Separating axis found, no overlap
        
        # No separating axis found, polygons overlap
        # Estimate penetration depth as minimum distance between edges (approximation)
        # For simplicity, use a large penalty
        return 10.0
    
    def total_penalty(config):
        """Sum of overlap penalties for all pairs."""
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                penalty += hexagon_overlap_penalty(config[i], config[j])
        return penalty
    
    def objective(config):
        """Objective: outer side length + penalty for overlaps."""
        side = outer_side_length(config)
        pen = total_penalty(config)
        return side + 100.0 * pen
    
    # Initial configurations for multi-start
    def make_initial_config(seed_idx):
        """Create initial configuration based on seed index."""
        config = np.zeros((N, 3))
        if seed_idx == 0:
            # Hexagonal arrangement: center + ring
            # 1 center, 6 in first ring, 4 more
            config[0] = [0.0, 0.0, 0.0]
            for i in range(1, 7):
                ang = np.deg2rad(60 * i)
                config[i] = [2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0]
            # 4 more in second ring positions
            config[7] = [4.0, 0.0, 0.0]
            config[8] = [2.0 * np.cos(np.deg2rad(60)), 2.0 * np.sin(np.deg2rad(60)), 0.0]
            config[9] = [2.0 * np.cos(np.deg2rad(120)), 2.0 * np.sin(np.deg2rad(120)), 0.0]
            config[10] = [2.0 * np.cos(np.deg2rad(180)), 2.0 * np.sin(np.deg2rad(180)), 0.0]
        elif seed_idx == 1:
            # Row-based arrangement
            for i in range(N):
                row = i // 4
                col = i % 4
                x = col * 2.0 - 3.0
                y = row * np.sqrt(3.0)
                config[i] = [x, y, 0.0]
        elif seed_idx == 2:
            # Random perturbation from hex arrangement
            config[0] = [0.0, 0.0, 0.0]
            for i in range(1, 7):
                ang = np.deg2rad(60 * i)
                config[i] = [2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0]
            np.random.seed(seed_idx)
            for i in range(7, N):
                ang = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(3.0, 4.0)
                config[i] = [r * np.cos(ang), r * np.sin(ang), np.random.uniform(0, 60)]
        else:
            # Random configuration
            np.random.seed(seed_idx)
            for i in range(N):
                config[i] = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0, 60)]
        return config
    
    # Run multi-start optimization
    best_config = None
    best_score = np.inf
    
    for seed_idx in range(4):
        x0 = make_initial_config(seed_idx)
        
        # Define bounds: positions in reasonable range, angles in [0, 60)
        bounds = [(-20, 20), (-20, 20), (0, 60)] * N
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_config = result.x
        except Exception:
            continue
    
    # If optimization failed, use best initial config
    if best_config is None:
        best_config = make_initial_config(0)
    
    # Extract final configuration
    inner = []
    for i in range(N):
        x, y, ang = best_config[i]
        # Normalize angle to [0, 60)
        ang = ang % 60.0
        inner.append([float(x), float(y), float(ang)])
    
    # Compute final outer side length
    final_side = outer_side_length(best_config)
    
    # Outer hexagon center is at origin, rotation is 0 (vertex pointing right)
    return inner, (0.0, 0.0), float(final_side), 0.0
# EVOLVE-BLOCK-END
