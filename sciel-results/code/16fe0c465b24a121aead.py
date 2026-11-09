import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hexagon_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon with side length s, center (cx, cy), 
    rotated by angle_deg.
    Vertices are in counter-clockwise order.
    """
    angle_rad = math.radians(angle_deg)
    # Standard hexagon vertices at angle 0: (s, 0), (s/2, s*sqrt(3)/2), ...
    # Wait, side length s. Distance from center to vertex is s.
    # Distance from center to edge (apothem) is s * sqrt(3)/2.
    # Let's generate points at distance s from center, angles 0, 60, 120...
    pts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return np.array(pts)

def _point_in_hexagon(px, py, cx, cy, s, angle_deg):
    """
    Check if point (px, py) is inside hexagon defined by center, side, angle.
    Uses support function / half-plane intersection logic for speed.
    A convex polygon contains a point if the point is on the same side 
    (or on) the line defined by each edge as the interior.
    """
    verts = _hexagon_vertices(cx, cy, s, angle_deg)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        # Vector p1 -> p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # Normal pointing inward. 
        # For CCW polygon, inward normal is (dy, -dx) rotated? 
        # Let's check cross product.
        # Vector from p1 to point P: (px-p1x, py-p1y)
        # Cross product (p2-p1) x (P-p1) should have consistent sign for CCW interior.
        # z-component: dx*(py-p1y) - dy*(px-p1x)
        # For CCW, interior is to the left of edge? No, CCW polygon interior is to the left of edges if we traverse?
        # Let's just use distance to line and check against center.
        
        # Simpler: Check if P and Center are on same side of line p1-p2.
        # Line equation: A x + B y + C = 0
        # A = y1 - y2
        # B = x2 - x1
        # C = x1 y2 - x2 y1
        
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0]*p2[1] - p2[0]*p1[1]
        
        val_p = A*px + B*py + C
        val_c = A*cx + B*cy + C
        
        if val_p * val_c < -1e-9:
            return False
    return True

def _hexagons_overlap(cx1, cy1, s1, a1, cx2, cy2, s2, a2, tol=1e-7):
    """
    Check if two hexagons overlap.
    1. Check if any vertex of 1 is in 2.
    2. Check if any vertex of 2 is in 1.
    3. Check if centers are close enough that one might contain the other (unlikely for same size, but safe).
    Actually, for convex polygons, checking vertices is sufficient if they are disjoint.
    If they overlap, either a vertex of one is inside the other, or they intersect at edges (which implies a vertex is inside or on boundary if we consider strict overlap, but for packing we want strict disjointness or touching).
    To be safe and fast, we check vertices.
    """
    v1 = _hexagon_vertices(cx1, cy1, s1, a1)
    v2 = _hexagon_vertices(cx2, cy2, s2, a2)
    
    # Check v1 in hex2
    for p in v1:
        if _point_in_hexagon(p[0], p[1], cx2, cy2, s2, a2):
            return True
            
    # Check v2 in hex1
    for p in v2:
        if _point_in_hexagon(p[0], p[1], cx1, cy1, s1, a1):
            return True
            
    return False

def _compute_outer_side_length(positions, angles, s_inner=1.0):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, rotation 0 for simplicity 
    or we can optimize rotation too, but let's fix outer rotation to 0 for now to reduce variables, 
    or allow outer rotation? The task says outer hexagon center, side, rotation. 
    Usually optimal packing aligns with outer hex. Let's assume outer rotation 0 and center 0 for now, 
    but we should probably check if shifting/rotating outer helps. 
    Given symmetry, center 0 is likely optimal. Rotation 0 is likely optimal if inner hexes are aligned.
    If inner hexes are rotated, maybe outer should rotate too? 
    Let's assume outer is fixed at (0,0) with rotation 0 for the greedy placement to keep it fast.
    We calculate the max distance from origin to any vertex of the inner hexagons, 
    then derive the side length of the outer hexagon that contains all these points.
    
    For a regular hexagon with side L, centered at origin, rotated 0:
    The vertices are at distance L from origin.
    The "width" in x direction is 2L.
    The "height" in y direction is 2 * L * sqrt(3)/2 = L * sqrt(3).
    
    Actually, a point (x,y) is inside the outer hexagon if:
    |x| <= L
    |y| <= L * sqrt(3)/2
    |x| * sqrt(3) + |y| <= L * sqrt(3) ? No.
    
    Let's use the support function.
    The outer hexagon is defined by 6 half-planes.
    For a hexagon of side L, centered at 0, rot 0:
    The normals to the edges are at angles 0, 60, 120, 180, 240, 300.
    The distance from center to edge (apothem) is h = L * sqrt(3)/2.
    So the condition is: for all k in 0..5, dot(P, n_k) <= h.
    n_k = (cos(k*pi/3), sin(k*pi/3)).
    
    So L must satisfy:
    L * sqrt(3)/2 >= max_k ( dot(P, n_k) )
    L >= (2/sqrt(3)) * max_k ( dot(P, n_k) )
    
    We need to check all vertices of all inner hexagons.
    """
    max_proj = 0.0
    normals = []
    for k in range(6):
        a = k * math.pi / 3.0
        normals.append((math.cos(a), math.sin(a)))
    
    for i in range(len(positions)):
        cx, cy = positions[i]
        a_deg = angles[i]
        verts = _hexagon_vertices(cx, cy, s_inner, a_deg)
        for p in verts:
            for nx, ny in normals:
                proj = abs(nx * p[0] + ny * p[1])
                if proj > max_proj:
                    max_proj = proj
                    
    # L * sqrt(3)/2 >= max_proj
    # L >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def _local_search_for_placement(existing_pos, existing_ang, s_inner=1.0, max_iter=50):
    """
    Find a position and angle for a new hexagon that is disjoint from existing ones
    and minimizes the increase in outer side length.
    We start with a random position and angle, and try to nudge it to reduce overlap penalty or fit better.
    Actually, simpler: try a grid of positions and angles around the "ideal" spot (spiral), 
    and pick the one that minimizes the new outer side length, provided it's disjoint.
    """
    best_L = float('inf')
    best_pos = None
    best_ang = None
    
    # Generate candidate positions in a spiral/grid around origin
    # We want to place it close to existing ones to keep L small.
    # Let's try a range of radii and angles.
    
    # To speed up, we can first check if it overlaps.
    # We'll iterate over a set of candidate angles for the new hexagon.
    # Common angles: 0, 60, 120, 180, 240, 300. Maybe also 30, 90...
    candidate_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # Candidate positions: 
    # We don't know where to put it. 
    # Let's try positions on a grid of size ~ 2*s_inner apart, within a bounding box.
    # The bounding box can be estimated from current L.
    current_L = _compute_outer_side_length(existing_pos, existing_ang, s_inner) if existing_pos else 1.0
    # The new hexagon should be within L + s_inner roughly.
    max_r = current_L + s_inner * 2
    
    # Step size for grid search
    step = s_inner * 0.5
    
    # This grid search might be too slow if max_r is large.
    # Optimization: Only check positions that are "close" to the boundary of the current cluster?
    # Or just use a smaller search space initially.
    
    # Let's use a randomized local search with restarts for the new hexagon.
    # Start with a random valid position? No, we need to find a valid position.
    
    # Strategy: 
    # 1. Try to place at a specific set of "good" spots relative to existing hexes.
    # 2. If that fails, expand search.
    
    # For simplicity and robustness within time limit, let's use a coarse grid search 
    # within a radius that is slightly larger than the current bounding radius.
    
    # Actually, let's just try to place it at the "next" spot in a hexagonal lattice 
    # and see if it fits. If not, nudge it.
    
    # Better: Use a simple iterative improvement.
    # Pick a random initial position (e.g., on a circle of radius current_L + s_inner) and angle.
    # Check overlap. If overlap, move away from the overlapping hex.
    # Repeat until no overlap.
    # Then check L.
    
    # Let's do 20 restarts with random initial positions.
    
    for _ in range(20):
        # Random start
        r = current_L + s_inner * 0.5 + random.uniform(0, s_inner)
        theta = random.uniform(0, 2 * math.pi)
        cx = r * math.cos(theta)
        cy = r * math.sin(theta)
        ang = random.choice(candidate_angles)
        
        # Local improvement
        for _iter in range(max_iter):
            # Check overlap
            overlap_found = False
            for j in range(len(existing_pos)):
                if _hexagons_overlap(cx, cy, s_inner, ang, existing_pos[j][0], existing_pos[j][1], s_inner, existing_ang[j]):
                    overlap_found = True
                    # Move away from j
                    dx = cx - existing_pos[j][0]
                    dy = cy - existing_pos[j][1]
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    # Move a small step away
                    step_size = s_inner * 0.2
                    cx += (dx / dist) * step_size
                    cy += (dy / dist) * step_size
                    break # Restart loop for this iteration? Or continue? 
                    # If we move, we might overlap with others. 
                    # Let's just break and re-check in next iteration.
            
            if not overlap_found:
                # Valid position!
                new_L = _compute_outer_side_length(existing_pos + [(cx, cy)], existing_ang + [ang], s_inner)
                if new_L < best_L:
                    best_L = new_L
                    best_pos = (cx, cy)
                    best_ang = ang
                break # Found a valid one, stop improving this restart? 
                # Or continue to see if we can get a smaller L? 
                # Moving away from overlap usually increases L. 
                # So the first valid position found by moving away is likely local min for this restart.
        
    if best_pos is None:
        # Failed to find a valid spot with this heuristic
        return None, None, float('inf')
        
    return best_pos, best_ang, best_L

import random

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    positions = []
    angles = []
    s_inner = 1.0
    
    # Place first hexagon at center
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        # Try to place next hexagon
        pos, ang, L = _local_search_for_placement(
            [p for p in positions], 
            angles, 
            s_inner, 
            max_iter=20 # Keep low to save time
        )
        
        if pos is None:
            # If local search fails, try a brute force grid with larger step?
            # Or just place it far away (bad for score but valid)
            # Let's try a simple radial placement
            r = 2.0 * i
            theta = i * 2.399963 # Golden angle
            pos = (r * math.cos(theta), r * math.sin(theta))
            ang = 0.0
            
            # Check if it overlaps with anything
            valid = True
            for j in range(len(positions)):
                if _hexagons_overlap(pos[0], pos[1], s_inner, ang, positions[j][0], positions[j][1], s_inner, angles[j]):
                    valid = False
                    break
            
            if not valid:
                # Nudge it far away
                pos = (pos[0] + 10.0, pos[1])
                
        positions.append(pos)
        angles.append(ang)
        
        # Optional: Local optimization of all positions?
        # Too expensive.
        
    # Calculate final outer side length
    if positions:
        outer_side = _compute_outer_side_length(positions, angles, s_inner)
    else:
        outer_side = 0.0
        
    # Format output
    inner = []
    for i in range(len(positions)):
        inner.append([positions[i][0], positions[i][1], angles[i]])
        
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-END
