"""Packing 12 unit hexagons in a regular hexagon; minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill
MIN_DIST = np.sqrt(3.0)

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
    # Normals to the sides of the outer hexagon
    # Vertices are at angle_rad + k*pi/3
    # Normals are at angle_rad + pi/6 + k*pi/3
    normal_angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    rel_points = points - np.array([cx, cy])
    dots = rel_points @ normals.T
    max_dot = np.max(dots, axis=0)
    r_needed = np.max(max_dot)
    if r_needed < 0:
        r_needed = 0.0
    # Inradius r = s * sqrt(3) / 2 => s = 2 * r / sqrt(3)
    s = 2.0 / np.sqrt(3.0) * r_needed
    return s

def compute_min_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length for given inner configuration."""
    all_points = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        all_points.append(v)
    all_points = np.vstack(all_points)
    return min_outer_side_for_points(all_points, outer_center, outer_angle_deg)

def objective(params):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: 24 values [x1,y1,...,x12,y12]
    Assumes all inner hexagons have angle 0.
    """
    pos = params.reshape(N, 2)
    
    # Calculate outer side length
    all_points = []
    for i in range(N):
        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
        all_points.append(v)
    all_points = np.vstack(all_points)
    
    s = min_outer_side_for_points(all_points, (0.0, 0.0), 0.0)
    
    # Penalty for overlaps (distance < sqrt(3))
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < MIN_DIST:
                penalty += (MIN_DIST - d) ** 2 * 100.0
    
    return s + penalty

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
        # Check for overlaps in the final config
        pos = np.array([[x, y] for x, y, a in inner])
        is_valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < MIN_DIST - 1e-6:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if is_valid and s < best_score:
            best_score = s
            best_inner = [list(x) for x in inner]
            best_center = center
            best_side = s
            best_angle = angle

    sqrt3 = np.sqrt(3)
    
    # Initial Configurations
    
    # 1. 3x4 Grid
    positions = []
    for i in range(3):
        for j in range(4):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions.append((x, y))
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    inner_grid1 = [[p[0] - cx, p[1] - cy, 0.0] for p in positions]
    try_config(inner_grid1, (0.0, 0.0), 0.0)
    
    # 2. 4x3 Grid
    positions2 = []
    for i in range(4):
        for j in range(3):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions2.append((x, y))
    cx2 = np.mean([p[0] for p in positions2])
    cy2 = np.mean([p[1] for p in positions2])
    inner_grid2 = [[p[0] - cx2, p[1] - cy2, 0.0] for p in positions2]
    try_config(inner_grid2, (0.0, 0.0), 0.0)
    
    # 3. Center + 6 + 5 (Defective Hexagon)
    positions3 = [(0, 0)]
    for k in range(6):
        ang = k * np.pi / 3
        positions3.append((sqrt3*np.cos(ang), sqrt3*np.sin(ang)))
    # Add 5 from the next ring
    # The next ring centers are at distance 2*sqrt3
    # We pick 5 that form a compact shape
    # Angles for ring 2: 30, 90, 150, 210, 270 (offset by 30 from ring 1)
    # Or just pick 5 consecutive from the 6 spots?
    # Let's try placing them at angles 0, 60, 120, 180, 240 with distance 2*sqrt3?
    # No, they must touch ring 1.
    # A hexagon at (0,0) touches 6.
    # A hexagon at ring 2 touches 2 or 3 from ring 1.
    # Let's just use a random compact set for now, or a specific one.
    # A known good 12-hexagon packing is the "hexagonal cluster" with one missing from the second ring.
    # Ring 1: 6 hexes.
    # Ring 2: 12 spots. We need 5.
    # Let's pick 5 spots that are adjacent to each other on the perimeter.
    ring2_angles = [np.pi/6 + k * np.pi/3 for k in range(6)] # These are the "corner" directions?
    # Actually, the centers of ring 2 hexagons are at distance 2*sqrt3 from center?
    # No, if they touch ring 1, distance is sqrt3 (from ring 1 center).
    # Ring 1 center is at dist sqrt3.
    # If Ring 2 hex touches Ring 1 hex, dist between centers is sqrt3.
    # So Ring 2 center is at dist 2*sqrt3 from origin?
    # Let's just generate a dense cluster.
    positions3 = []
    # Generate a hexagonal grid of side 3 (which has 19 hexes) and take 12 compact ones.
    # Grid coords (q, r)
    # Center (0,0)
    # Neighbors
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Convert axial to cartesian
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            coords.append((q, r, x, y))
            
    # Sort by distance from center and pick 12
    coords.sort(key=lambda c: c[2]**2 + c[3]**2)
    # The 12 closest points in a hex grid form a compact shape
    selected = coords[:12]
    pos3 = [(x, y) for q, r, x, y in selected]
    cx3 = np.mean([p[0] for p in pos3])
    cy3 = np.mean([p[1] for p in pos3])
    inner_ring = [[p[0] - cx3, p[1] - cy3, 0.0] for p in pos3]
    try_config(inner_ring, (0.0, 0.0), 0.0)
    
    # Optimization
    if best_inner is not None:
        init_pos = np.array([[x, y] for x, y, a in best_inner])
        
        # Multiple restarts with perturbation
        n_restarts = 15
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            if r == 0:
                x0 = init_pos.flatten()
            else:
                scale = 0.5 * np.sqrt(r)
                x0 = init_pos.flatten() + scale * np.random.randn(2*N)
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                pos_opt = result.x.reshape(N, 2)
                inner_opt = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), 0.0] for i in range(N)]
                
                # Verify disjointness strictly
                is_valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pos_opt[i] - pos_opt[j])
                        if d < MIN_DIST - 1e-6:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_inner = inner_opt
                        best_center = (0.0, 0.0)
                        best_side = s_opt
                        best_angle = 0.0
                        
                        # Try to optimize outer rotation for this specific config
                        for outer_ang in [0, 15, 30, 45, 60]:
                            s_rot = compute_min_side(inner_opt, (0.0, 0.0), float(outer_ang))
                            if s_rot < best_score:
                                best_score = s_rot
                                best_side = s_rot
                                best_angle = float(outer_ang)
                                
            except Exception:
                pass
    
    # Add safety margin
    best_side += 0.05
    
    return best_inner, best_center, best_side, best_angle
