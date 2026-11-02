"""Improved packing of 12 unit hexagons inside a regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append([cx + side * math.cos(theta), cy + side * math.sin(theta)])
    return np.array(verts)

def hexagon_overlap_area(v1, v2):
    """Compute overlap area between two convex hexagons using Sutherland-Hodgman clipping."""
    def clip_polygon(subject, clip_edge_p1, clip_edge_p2):
        """Clip subject polygon against edge from p1 to p2 (keep inside)."""
        if len(subject) == 0:
            return []
        
        def inside(p, p1, p2):
            # Check if p is to the left of directed line p1->p2
            return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= -1e-12
        
        def intersect(p1, p2, q1, q2):
            # Line segment p1-p2 intersect with line q1-q2
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = q1
            x4, y4 = q2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-15:
                return p1  # parallel, return p1 as fallback
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
        
        output = []
        n = len(subject)
        for i in range(n):
            current = subject[i]
            previous = subject[i - 1]
            
            if inside(current, clip_edge_p1, clip_edge_p2):
                if not inside(previous, clip_edge_p1, clip_edge_p2):
                    output.append(intersect(previous, current, clip_edge_p1, clip_edge_p2))
                output.append(current)
            elif inside(previous, clip_edge_p1, clip_edge_p2):
                output.append(intersect(previous, current, clip_edge_p1, clip_edge_p2))
        
        return output
    
    # Clip v1 against each edge of v2
    result = [list(v1[i]) for i in range(6)]
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        result = clip_polygon(result, p1, p2)
        if len(result) < 3:
            return 0.0
    
    if len(result) < 3:
        return 0.0
    
    # Compute area of clipped polygon
    area = 0.0
    n = len(result)
    for i in range(n):
        j = (i + 1) % n
        area += result[i][0] * result[j][1]
        area -= result[j][0] * result[i][1]
    return abs(area) / 2.0

def compute_outer_side(inner_positions):
    """
    Given inner hexagon positions (list of [x, y, angle_deg]),
    compute the minimum outer hexagon side length and its center/rotation.
    
    We try multiple outer hexagon rotations and centers to find the minimum enclosing side.
    """
    # Extract inner hexagon vertices
    all_verts = []
    for x, y, ang in inner_positions:
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    
    # For a given outer hexagon (center, side, rotation), check if all inner hexes fit.
    # We'll optimize over outer center and rotation, and binary search on side.
    
    def fits(side, outer_cx, outer_cy, outer_angle):
        outer_verts = hex_vertices(outer_cx, outer_cy, side, outer_angle)
        for inner_verts in all_verts:
            overlap = hexagon_overlap_area(inner_verts, outer_verts)
            # Inner hex area
            inner_area = hexagon_overlap_area(inner_verts, inner_verts)
            if inner_area > 1e-10 and overlap < inner_area - 1e-8:
                return False
        return True
    
    # We need to find minimum side such that there exists a center and rotation where all fit.
    # Strategy: try several rotations, for each do a 2D optimization of center.
    
    best_side = 50.0
    
    for outer_angle in range(0, 60, 10):  # try 6 rotations
        # Initial guess: centroid of all inner vertices
        all_pts = np.vstack(all_verts)
        cx0 = np.mean(all_pts[:, 0])
        cy0 = np.mean(all_pts[:, 1])
        
        # Binary search on side
        lo, hi = 2.0, 50.0
        
        for _ in range(20):  # binary search iterations
            mid = (lo + hi) / 2.0
            if fits(mid, cx0, cy0, outer_angle):
                hi = mid
            else:
                # Try to optimize center for this side
                def objective(center):
                    cx, cy = center
                    if fits(mid, cx, cy, outer_angle):
                        return 0.0
                    # Penalty: sum of overlaps of inner hexes that stick out
                    outer_verts = hex_vertices(cx, cy, mid, outer_angle)
                    penalty = 0.0
                    for inner_verts in all_verts:
                        overlap = hexagon_overlap_area(inner_verts, outer_verts)
                        inner_area = hexagon_overlap_area(inner_verts, inner_verts)
                        if inner_area > 1e-10:
                            penalty += max(0, inner_area - overlap)
                    return penalty
                
                res = minimize(objective, [cx0, cy0], method='Nelder-Mead',
                              options={'maxiter': 50, 'xatol': 0.01, 'fatol': 1e-6})
                cx_opt, cy_opt = res.x
                if fits(mid, cx_opt, cy_opt, outer_angle):
                    hi = mid
                    cx0, cy0 = cx_opt, cy_opt
                else:
                    lo = mid
        
        if hi < best_side:
            best_side = hi
    
    return best_side

def construct():
    global START_TIME
    start = time.time()
    
    # Initial configuration: place hexagons in a honeycomb-like pattern
    # Unit hexagons have width 2 (flat-to-flat) and height sqrt(3) (point-to-point)
    # For flat-top hexagons: width = 2*side, height = sqrt(3)*side
    # For pointy-top: width = sqrt(3)*side, height = 2*side
    
    # Let's use a simple linear arrangement first, then optimize
    # Place 12 hexagons in a 4x3 grid with honeycomb offset
    
    def make_initial_config():
        config = []
        # Honeycomb packing: rows offset by half
        # For flat-top hexagons (angle=0 means flat top):
        # Horizontal spacing: 1.5 * side = 1.5
        # Vertical spacing: sqrt(3) * side = sqrt(3)
        # Row offset: sqrt(3)/2 for odd rows
        
        rows = 3
        cols = 4
        for r in range(rows):
            for c in range(cols):
                x = c * 1.5 + (0.75 if r % 2 == 1 else 0)
                y = r * math.sqrt(3)
                config.append([x, y, 0.0])
        return config
    
    initial_config = make_initial_config()
    
    # Compute initial outer side
    current_side = compute_outer_side(initial_config)
    
    # Try to improve by optimizing inner positions
    # Use a simple local search: perturb positions and check if outer side decreases
    
    best_config = initial_config[:]
    best_side = current_side
    
    # Try different initial arrangements
    arrangements = [
        make_initial_config,
        lambda: [[c * 2.0, 0.0, 0.0] for c in range(12)],  # linear
        lambda: [[(c % 3) * 2.0, (c // 3) * math.sqrt(3), 0.0] for c in range(12)],  # 3x4 grid
    ]
    
    for arr_func in arrangements:
        if time.time() - start > DEADLINE - 30:
            break
        config = arr_func()
        side = compute_outer_side(config)
        if side < best_side:
            best_side = side
            best_config = config[:]
    
    # Local optimization: try small perturbations
    if time.time() - start < DEADLINE - 20:
        for iteration in range(50):
            if time.time() - start > DEADLINE - 20:
                break
            
            improved = False
            for i in range(N):
                for dx in [-0.1, 0.1]:
                    for dy in [-0.1, 0.1]:
                        new_config = best_config[:]
                        new_config[i] = [best_config[i][0] + dx, best_config[i][1] + dy, best_config[i][2]]
                        side = compute_outer_side(new_config)
                        if side < best_side - 1e-6:
                            best_side = side
                            best_config = new_config
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Compute final outer hexagon parameters
    # Use the best configuration to determine outer hexagon
    all_verts = []
    for x, y, ang in best_config:
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    
    all_pts = np.vstack(all_verts)
    cx = np.mean(all_pts[:, 0])
    cy = np.mean(all_pts[:, 1])
    
    # Find best rotation for outer hexagon
    best_outer_angle = 0
    best_final_side = best_side
    
    for outer_angle in range(0, 60, 5):
        # Binary search for minimum side with this rotation
        lo, hi = 2.0, 50.0
        for _ in range(15):
            mid = (lo + hi) / 2.0
            outer_verts = hex_vertices(cx, cy, mid, outer_angle)
            all_fit = True
            for inner_verts in all_verts:
                overlap = hexagon_overlap_area(inner_verts, outer_verts)
                inner_area = hexagon_overlap_area(inner_verts, inner_verts)
                if inner_area > 1e-10 and overlap < inner_area - 1e-8:
                    all_fit = False
                    break
            if all_fit:
                hi = mid
            else:
                lo = mid
        if hi < best_final_side:
            best_final_side = hi
            best_outer_angle = outer_angle
    
    return best_config, (cx, cy), best_final_side, best_outer_angle
