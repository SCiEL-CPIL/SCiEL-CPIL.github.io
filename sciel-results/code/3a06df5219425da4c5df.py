"""Packing 12 unit hexagons in a regular hexagon; minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill
SQRT3 = np.sqrt(3.0)

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy)."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def min_outer_side_for_points(points, outer_center, outer_angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at outer_center
    with rotation outer_angle_deg that contains all given points.
    """
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle_deg)
    normal_angles = angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    rel_points = points - np.array([cx, cy])
    dots = rel_points @ normals.T
    max_dot = np.max(dots, axis=0)
    r_needed = np.max(max_dot)
    if r_needed < 0:
        r_needed = 0.0
    s = 2.0 / SQRT3 * r_needed
    return s

def compute_min_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length for given inner configuration."""
    all_points = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        all_points.append(v)
    all_points = np.vstack(all_points)
    return min_outer_side_for_points(all_points, outer_center, outer_angle_deg)

def check_disjoint(inner, tol=1e-6):
    """Check if all inner hexagons are disjoint (no overlapping interiors)."""
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _polygons_overlap(v1, v2):
                return False
    return True

def _polygons_overlap(v1, v2):
    """Check if two convex hexagons (given as (6,2) arrays) have overlapping interiors."""
    # Use SAT: project onto each edge normal
    for verts in [v1, v2]:
        for k in range(6):
            edge = verts[(k+1)%6] - verts[k]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal = normal / np.linalg.norm(normal)
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # If separated with a gap, no overlap
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    return True

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    def try_config(inner, center, angle):
        nonlocal best_score, best_inner, best_center, best_side, best_angle
        s = compute_min_side(inner, center, angle)
        if s < best_score:
            best_score = s
            best_inner = [list(x) for x in inner]
            best_center = center
            best_side = s
            best_angle = angle
    
    # Config 1: 3x4 hexagonal grid (optimal for hexagonal packing)
    positions = []
    for i in range(3):
        for j in range(4):
            x = j * 2 * SQRT3 + (i % 2) * SQRT3
            y = i * 3
            positions.append((x, y))
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    inner_grid = [[p[0] - cx, p[1] - cy, 0.0] for p in positions]
    try_config(inner_grid, (0.0, 0.0), 0.0)
    
    # Config 2: 4x3 hexagonal grid
    positions2 = []
    for i in range(4):
        for j in range(3):
            x = j * 2 * SQRT3 + (i % 2) * SQRT3
            y = i * 3
            positions2.append((x, y))
    cx2 = np.mean([p[0] for p in positions2])
    cy2 = np.mean([p[1] for p in positions2])
    inner_grid2 = [[p[0] - cx2, p[1] - cy2, 0.0] for p in positions2]
    try_config(inner_grid2, (0.0, 0.0), 0.0)
    
    # Config 3: Center + 2 rings
    positions3 = [(0, 0)]
    for k in range(6):
        ang = k * np.pi / 3
        positions3.append((2*SQRT3*np.cos(ang), 2*SQRT3*np.sin(ang)))
    for k in range(5):
        ang = k * 2*np.pi/5 + np.pi/5
        positions3.append((4*SQRT3*np.cos(ang), 4*SQRT3*np.sin(ang)))
    cx3 = np.mean([p[0] for p in positions3])
    cy3 = np.mean([p[1] for p in positions3])
    inner_ring = [[p[0] - cx3, p[1] - cy3, 0.0] for p in positions3]
    try_config(inner_ring, (0.0, 0.0), 0.0)
    
    # Config 4: 2x6 rectangular-ish
    positions4 = []
    for i in range(2):
        for j in range(6):
            x = j * 2 * SQRT3 + (i % 2) * SQRT3
            y = i * 3
            positions4.append((x, y))
    cx4 = np.mean([p[0] for p in positions4])
    cy4 = np.mean([p[1] for p in positions4])
    inner_rect = [[p[0] - cx4, p[1] - cy4, 0.0] for p in positions4]
    try_config(inner_rect, (0.0, 0.0), 0.0)
    
    # Try different outer rotations
    for cfg in [inner_grid, inner_grid2, inner_ring, inner_rect]:
        if time.time() - start_time > DEADLINE:
            break
        for outer_ang in [0, 30, 60, 90, 120, 150]:
            if time.time() - start_time > DEADLINE:
                break
            try_config(cfg, (0.0, 0.0), float(outer_ang))
    
    # Physics-based repulsion optimization
    if best_inner is not None and time.time() - start_time < DEADLINE - 5:
        init_pos = np.array([[x, y] for x, y, a in best_inner])
        
        # Multiple restarts with physics-based repulsion
        n_restarts = 6
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if r == 0:
                pos = init_pos.copy()
            else:
                scale = 0.2 * (r + 1)
                pos = init_pos + scale * np.random.randn(N, 2)
            
            # Physics-based repulsion: iterate until convergence or time limit
            step_size = 0.1
            for iteration in range(200):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Compute forces
                forces = np.zeros((N, 2))
                for i in range(N):
                    v1 = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
                    for j in range(i+1, N):
                        v2 = hexagon_vertices(pos[j, 0], pos[j, 1], 1.0, 0.0)
                        f = hexagon_repulsion_force(v1, v2, strength=50.0)
                        forces[i] += f
                        forces[j] -= f
                
                # Also add a small attraction toward center to keep them bounded
                center = np.mean(pos, axis=0)
                for i in range(N):
                    forces[i] -= 0.1 * (pos[i] - center)
                
                # Update positions
                pos += step_size * forces
                step_size *= 0.95  # Decay step size
                
                # Check convergence
                if np.max(np.abs(forces)) < 1e-4:
                    break
            
            pos_opt = pos
            inner_opt = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), 0.0] for i in range(N)]
            s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
            
            # Verify disjointness
            if check_disjoint(inner_opt, tol=1e-5):
                if s_opt < best_score:
                    best_score = s_opt
                    best_inner = inner_opt
                    best_center = (0.0, 0.0)
                    best_side = s_opt
                    best_angle = 0.0
        
        # Final L-BFGS-B polish from best physics result
        if best_inner is not None and time.time() - start_time < DEADLINE - 3:
            init_pos = np.array([[x, y] for x, y, a in best_inner])
            try:
                result = minimize(
                    objective_full,
                    init_pos.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                pos_opt = result.x.reshape(N, 2)
                inner_opt = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), 0.0] for i in range(N)]
                s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
                
                if check_disjoint(inner_opt, tol=1e-5):
                    if s_opt < best_score:
                        best_score = s_opt
                        best_inner = inner_opt
                        best_center = (0.0, 0.0)
                        best_side = s_opt
                        best_angle = 0.0
            except Exception:
                pass
    
    # Add safety margin
    best_side += 0.05
    
    return best_inner, best_center, best_side, best_angle

def hexagon_distance(v1, v2):
    """Compute minimum distance between two convex hexagons (6,2 arrays). Returns 0 if overlapping."""
    # Use SAT to find minimum separation
    min_sep = float('inf')
    for verts in [v1, v2]:
        for k in range(6):
            edge = verts[(k+1)%6] - verts[k]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # If separated
            if max1 < min2:
                sep = min2 - max1
                min_sep = min(min_sep, sep)
            elif max2 < min1:
                sep = min1 - max2
                min_sep = min(min_sep, sep)
    if min_sep == float('inf'):
        return 0.0  # Overlapping or touching
    return min_sep

def hexagon_repulsion_force(v1, v2, strength=100.0):
    """Compute repulsion force on hexagon 1 due to hexagon 2. Returns (fx, fy)."""
    d = hexagon_distance(v1, v2)
    if d > 0.01:
        # Small repulsion for near-contact
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        diff = c1 - c2
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            return (0.0, 0.0)
        force = strength * 0.01 / (1.0 + dist)
        return (force * diff[0]/dist, force * diff[1]/dist)
    else:
        # Strong repulsion when overlapping - use center difference direction
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        diff = c1 - c2
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            # Random direction if centers coincide
            angle = np.random.uniform(0, 2*np.pi)
            return (strength * np.cos(angle), strength * np.sin(angle))
        force = strength / (0.1 + dist)
        return (force * diff[0]/dist, force * diff[1]/dist)

def objective_full(params):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: 24 values [x1,y1,...,x12,y12]
    """
    pos = params.reshape(N, 2)
    
    all_points = []
    for i in range(N):
        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
        all_points.append(v)
    all_points = np.vstack(all_points)
    
    s = min_outer_side_for_points(all_points, (0.0, 0.0), 0.0)
    
    # Penalty for overlaps using actual polygon distance
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
            v2 = hexagon_vertices(pos[j, 0], pos[j, 1], 1.0, 0.0)
            d = hexagon_distance(v1, v2)
            if d < 0.0:
                # Overlapping - strong penalty
                penalty += 1000.0
            elif d < 0.1:
                # Near contact - moderate penalty
                penalty += (0.1 - d) ** 2 * 50.0
    
    return s + penalty
