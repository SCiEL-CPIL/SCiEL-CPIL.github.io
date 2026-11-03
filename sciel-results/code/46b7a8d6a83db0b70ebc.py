import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 3.0

# Precompute hex vertex offsets for side-1 regular hexagon (flat-top)
def hex_vertices(cx, cy, angle_deg=0.0):
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return pts

def hex_edge_normals(cx, cy, angle_deg=0.0):
    """Outward normals of the 6 edges of a unit hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        # edge midpoint direction is at angle a + pi/6 + k*pi/3
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    return normals

def min_enclosing_hexagon_side(pts, angle_deg=0.0):
    """
    Find the minimum side length S of a regular hexagon (with given rotation)
    that encloses all points. Returns (S, center_x, center_y).
    Uses linear programming via scipy.
    """
    from scipy.optimize import linprog
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx, cy = res.x[0], res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, cx, cy
        else:
            return float('inf'), 0.0, 0.0
    except Exception:
        return float('inf'), 0.0, 0.0

def hexagon_overlap_area_approx(v1, v2, angle1, angle2):
    """
    Approximate overlap area between two unit hexagons using polygon intersection.
    Returns 0 if no overlap, >0 if overlap.
    Uses shapely if available, else a simple distance-based check.
    """
    # For speed, use a conservative distance check first
    d = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
    if d > 2.0:  # max distance for any overlap of unit hexagons
        return 0.0
    if d < 0.01:
        return 1.0  # significant overlap
    # Rough estimate: overlap exists if center distance < 2*apothem = sqrt(3) ~ 1.732
    # More precisely, two unit hexagons (same orientation) don't overlap if d >= 2*APOTHEM*sqrt(3)/1 ... 
    # Actually for same orientation, min distance for no overlap is 2*APOTHEM*sqrt(3)?? 
    # Let's use: if d >= 2.0, no overlap. If d < sqrt(3), likely overlap.
    # For different orientations, it's complex. Use a penalty based on distance.
    if d < 2 * APOTHEM:  # ~1.732
        return (2 * APOTHEM - d) ** 2
    return 0.0

def force_directed_relaxation(initial_positions, max_iter=2000, dt=0.01, repulsion_strength=10.0):
    """
    Relax positions using force-directed method to minimize overlap.
    Returns optimized positions.
    """
    n = len(initial_positions)
    pos = np.array(initial_positions, dtype=float)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    # random nudge
                    dx = (np.random.random() - 0.5) * 0.01
                    dy = (np.random.random() - 0.5) * 0.01
                    dist = math.sqrt(dx*dx + dy*dy)
                
                # Repulsive force: strong when overlapping
                if dist < 2.0:
                    # Overlap region: two unit hexagons can overlap up to distance ~1.732 for same orientation
                    # Use a smooth repulsion
                    overlap_thresh = 1.732  # 2*APOTHEM
                    if dist < overlap_thresh:
                        force_mag = repulsion_strength * (overlap_thresh - dist) ** 2 / (dist + 1e-10)
                    else:
                        force_mag = repulsion_strength * 0.1 * (2.0 - dist) / (dist + 1e-10)
                    
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update positions
        pos += dt * forces
        # Decay step size
        dt *= 0.995
    
    return pos

def greedy_densest_placement():
    """
    Greedy placement: start with a central hexagon, then add each new hexagon
    at the position that maximizes local density while minimizing overlap.
    """
    positions = [(0.0, 0.0)]
    
    # Generate candidate positions on a grid
    grid_range = range(-5, 6)
    candidates = []
    for i in grid_range:
        for j in grid_range:
            # Hex grid coordinates to cartesian
            x = math.sqrt(3) * (i + j / 2.0)
            y = 1.5 * j
            candidates.append((x, y))
    
    # Also add rotated candidates
    for angle in [30, 60]:
        a = math.radians(angle)
        for i in grid_range:
            for j in grid_range:
                x0 = math.sqrt(3) * (i + j / 2.0)
                y0 = 1.5 * j
                x = x0 * math.cos(a) - y0 * math.sin(a)
                y = x0 * math.sin(a) + y0 * math.cos(a)
                candidates.append((x, y))
    
    for _ in range(N - 1):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        best_pos = None
        best_score = float('inf')
        
        # Evaluate candidate positions
        for cand in candidates:
            # Check overlap with existing positions
            overlap_penalty = 0.0
            for px, py in positions:
                d = math.sqrt((cand[0]-px)**2 + (cand[1]-py)**2)
                if d < 2.0:
                    overlap_penalty += (2.0 - d) ** 2
            
            # Score: minimize overlap, prefer closer to center for density
            center_dist = math.sqrt(cand[0]**2 + cand[1]**2)
            score = overlap_penalty * 10.0 + center_dist * 0.1
            
            if score < best_score:
                best_score = score
                best_pos = cand
        
        if best_pos is not None:
            positions.append(best_pos)
    
    return positions

def optimize_with_local_search(positions, max_iter=500):
    """
    Local search: try small perturbations to improve the objective.
    """
    pos = np.array(positions, dtype=float)
    n = len(pos)
    
    def objective(pts):
        # Compute min enclosing hexagon side for all orientations
        # We'll optimize over angle too
        best_S = float('inf')
        for angle_deg in np.linspace(0, 60, 13):
            S, _, _ = min_enclosing_hexagon_side(pts, angle_deg)
            if S < best_S:
                best_S = S
        # Add overlap penalty
        overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < 2.0:
                    overlap += (2.0 - d) ** 2
        return best_S + overlap * 5.0
    
    current_obj = objective(pos)
    step_size = 0.1
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        improved = False
        for i in range(n):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    new_obj = objective(new_pos)
                    if new_obj < current_obj:
                        pos = new_pos
                        current_obj = new_obj
                        improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 0.001:
                break
    
    return pos, current_obj

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Multiple strategies
    best_S = float('inf')
    best_positions = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Strategy 1: Greedy densest placement
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pos1 = greedy_densest_placement()
            pos1_arr = np.array(pos1)
            # Force-directed relaxation
            pos1_relaxed = force_directed_relaxation(pos1_arr, max_iter=1000)
            
            # Optimize angle
            for angle_deg in np.linspace(0, 60, 25):
                S, cx, cy = min_enclosing_hexagon_side(pos1_relaxed, angle_deg)
                if S < best_S:
                    best_S = S
                    best_positions = pos1_relaxed
                    best_angle = angle_deg
                    best_center = (cx, cy)
    except Exception:
        pass
    
    # Strategy 2: Start from hex grid pattern
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # 3x3 grid minus corners = 9, add 2 more
            axial = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
            # Add two more
            axial.append((2,0))
            axial.append((0,2))
            
            pts = []
            for q, r in axial:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pts.append((x, y))
            
            pos2 = np.array(pts)
            pos2_relaxed = force_directed_relaxation(pos2, max_iter=1000)
            
            for angle_deg in np.linspace(0, 60, 25):
                S, cx, cy = min_enclosing_hexagon_side(pos2_relaxed, angle_deg)
                if S < best_S:
                    best_S = S
                    best_positions = pos2_relaxed
                    best_angle = angle_deg
                    best_center = (cx, cy)
    except Exception:
        pass
    
    # Strategy 3: Ring + center
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pts = [(0.0, 0.0)]
            for k in range(6):
                theta = k * math.pi / 3.0
                pts.append((math.sqrt(3) * math.cos(theta), math.sqrt(3) * math.sin(theta)))
            # Add 4 more
            for k in range(4):
                theta = math.pi/6 + k * math.pi / 3.0
                pts.append((2 * math.sqrt(3) * math.cos(theta), 2 * math.sqrt(3) * math.sin(theta)))
            
            pos3 = np.array(pts)
            pos3_relaxed = force_directed_relaxation(pos3, max_iter=1000)
            
            for angle_deg in np.linspace(0, 60, 25):
                S, cx, cy = min_enclosing_hexagon_side(pos3_relaxed, angle_deg)
                if S < best_S:
                    best_S = S
                    best_positions = pos3_relaxed
                    best_angle = angle_deg
                    best_center = (cx, cy)
    except Exception:
        pass
    
    # Final optimization: local search on best
    if best_positions is not None:
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                pos_opt, _ = optimize_with_local_search(best_positions, max_iter=200)
                for angle_deg in np.linspace(0, 60, 50):
                    S, cx, cy = min_enclosing_hexagon_side(pos_opt, angle_deg)
                    if S < best_S:
                        best_S = S
                        best_positions = pos_opt
                        best_angle = angle_deg
                        best_center = (cx, cy)
        except Exception:
            pass
    
    # Fallback
    if best_positions is None or best_S == float('inf'):
        best_positions = np.array([(0.0, 0.0)] + [(math.sqrt(3)*math.cos(k*math.pi/3), math.sqrt(3)*math.sin(k*math.pi/3)) for k in range(6)] + [(2*math.sqrt(3)*math.cos(k*math.pi/3), 2*math.sqrt(3)*math.sin(k*math.pi/3)) for k in range(4)])
        best_angle = 0.0
        S, cx, cy = min_enclosing_hexagon_side(best_positions, best_angle)
        best_S = S
        best_center = (cx, cy)
    
    # Scale down slightly to ensure no overlap
    SCALE = 0.9999
    cx, cy = best_center
    pts_scaled = []
    for x, y in best_positions:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    # Recompute side length
    S_final, cx_final, cy_final = min_enclosing_hexagon_side(np.array(pts_scaled), best_angle)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(best_angle)])
    
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(best_angle)
    
    return inner, center, side, rotation
