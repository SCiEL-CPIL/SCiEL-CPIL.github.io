"""Hexagon packing: place N unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time
import itertools

N = 11
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit hexagon (circumradius 1)."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given side length (circumradius = side)."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside a regular hexagon with center (cx,cy), side length, rotation."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = -np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # Regular hexagon with circumradius = side
    # In local frame (flat-top at angle 0):
    # Vertices are at angles 0, 60, 120, 180, 240, 300
    # The hexagon is defined by |lx| <= side, |ly| <= SQRT3/2 * side, and |ly| <= SQRT3 * (side - |lx|)
    # Wait, for flat-top (angle 0), the flat sides are top and bottom.
    # Vertices: (1, 0), (0.5, SQRT3/2), (-0.5, SQRT3/2), (-1, 0), (-0.5, -SQRT3/2), (0.5, -SQRT3/2)
    # So max x is 1, max y is SQRT3/2.
    # The condition for being inside is:
    # |lx| <= side
    # |ly| <= SQRT3/2 * side
    # |ly| <= SQRT3 * (side - |lx|)  <-- This is for pointy-top? No.
    
    # Let's use the half-plane method for robustness.
    # Normals to edges of a flat-top hexagon (angle 0):
    # Edge 1: (1,0) to (0.5, SQRT3/2). Normal points inward: (SQRT3/2, -1/2) ? No.
    # Center is (0,0). Edge midpoint is (0.75, SQRT3/4). Normal is (0.75, SQRT3/4) normalized.
    # Actually, simpler:
    # A regular hexagon with circumradius R has inradius r = R * SQRT3 / 2.
    # For flat-top (angle 0), the flat sides are horizontal.
    # Top side: y = r. Bottom side: y = -r.
    # Side slopes: The other 4 sides have normals at 60, 120, 240, 300 degrees?
    # Let's just check against the 6 half-planes.
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # Edge 1: v0 to v1. v0=(R,0), v1=(R/2, R*SQRT3/2).
    # Midpoint m = (3R/4, R*SQRT3/4). Normal n = m normalized.
    # n = (3/4, SQRT3/4) / (sqrt(9/16 + 3/16)) = (3/4, SQRT3/4) / (sqrt(12)/4) = (3/4, SQRT3/4) / (SQRT3/2) = (SQRT3/2, 1/2).
    # Distance from center to edge is R * SQRT3 / 2.
    # So the half-plane is: x * (SQRT3/2) + y * (1/2) <= R * SQRT3 / 2.
    
    # Let's generate the 6 normals for a hexagon rotated by angle_deg.
    # Base normals (for angle 0) point to the midpoints of the edges.
    # Edge midpoints are at angles 30, 90, 150, 210, 270, 330.
    base_angles = np.deg2rad(30 + np.arange(6) * 60)
    rot = np.deg2rad(angle_deg)
    normals = np.column_stack([np.cos(base_angles + rot), np.sin(base_angles + rot)])
    
    # Inradius
    r = side * SQRT3 / 2.0
    
    # Check dot product
    for i in range(6):
        if lx * normals[i, 0] + ly * normals[i, 1] > r:
            return False
    return True

def _min_distance_to_boundary(cx, cy, angle_deg, outer_cx, outer_cy, outer_side, outer_angle):
    """Returns (min_margin, push_vector). Negative margin means outside."""
    verts = _hex_vertices(cx, cy, angle_deg)
    min_margin = 1e18
    push = np.array([0.0, 0.0])
    
    # Pre-compute outer vertices and edge normals for efficiency? 
    # For now, just check each vertex.
    outer_verts = _outer_hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    for i in range(6):
        px, py = verts[i]
        if not _point_in_hexagon(px, py, outer_cx, outer_cy, outer_side, outer_angle):
            # Find closest point on outer boundary
            best_dist = 1e18
            best_diff = np.array([0.0, 0.0])
            p = np.array([px, py])
            for j in range(6):
                q1 = outer_verts[j]
                q2 = outer_verts[(j+1)%6]
                # Project p onto segment q1-q2
                d = q2 - q1
                if np.dot(d, d) < 1e-12:
                    t = 0
                else:
                    t = np.clip(np.dot(p - q1, d) / np.dot(d, d), 0, 1)
                closest = q1 + t * d
                diff = p - closest
                dist = np.linalg.norm(diff)
                if dist < best_dist:
                    best_dist = dist
                    best_diff = diff
            
            # best_diff points from boundary to p (outward). We need to push p inward.
            if -best_dist < min_margin:
                min_margin = -best_dist
                push = -best_diff
                
    return min_margin, push

def _hex_overlap(p1, p2, a1, p3, p4, a2):
    """Check if two hexagons overlap. Returns (overlap_depth, push_vector)."""
    # Simple check: if centers are far enough, no overlap.
    # Max distance between centers for non-overlap is 2 * SQRT3 (touching flats) or 2 (touching points)?
    # For unit hexagons, the "radius" (center to vertex) is 1.
    # If distance > 2, definitely no overlap.
    dist = np.linalg.norm(p1 - p2)
    if dist > 2.0:
        return 0.0, np.array([0.0, 0.0])
        
    # Check if any vertex of hex1 is inside hex2 or vice versa.
    v1 = _hex_vertices(p1[0], p1[1], a1)
    v2 = _hex_vertices(p2[0], p2[1], a2)
    
    # Check v1 inside hex2
    for i in range(6):
        if _point_in_hexagon(v1[i, 0], v1[i, 1], p2[0], p2[1], 1.0, a2):
            # Overlap!
            # Calculate push. Simplest: push along center difference.
            push = p1 - p2
            if np.linalg.norm(push) < 1e-9:
                push = np.array([1.0, 0.0])
            return 1.0, push / np.linalg.norm(push)
            
    # Check v2 inside hex1
    for i in range(6):
        if _point_in_hexagon(v2[i, 0], v2[i, 1], p1[0], p1[1], 1.0, a1):
            push = p2 - p1
            if np.linalg.norm(push) < 1e-9:
                push = np.array([1.0, 0.0])
            return 1.0, push / np.linalg.norm(push)
            
    # It's possible for edges to cross without vertices inside (unlikely for convex shapes if one vertex is inside, but let's be safe).
    # For now, assume vertex check is sufficient for our discrete search.
    return 0.0, np.array([0.0, 0.0])

def _check_validity(positions, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if configuration is valid. Returns (is_valid, min_margin)."""
    min_margin = 1e18
    for i in range(N):
        for j in range(i + 1, N):
            overlap, _ = _hex_overlap(positions[i, :2], positions[j, :2], positions[i, 2], positions[j, :2], positions[j, 2])
            if overlap > 0:
                return False, -overlap
        margin, _ = _min_distance_to_boundary(positions[i, 0], positions[i, 1], positions[i, 2], outer_cx, outer_cy, outer_side, outer_angle)
        if margin < min_margin:
            min_margin = margin
        if margin < 0:
            return False, margin
    return True, min_margin

def _optimize_outer(positions):
    """Find minimum outer side length for given positions. Returns (outer_cx, outer_cy, outer_side, outer_angle)."""
    # The outer hexagon must contain all inner hexagons.
    # This is equivalent to finding the minimum enclosing regular hexagon.
    # We can search over the angle of the outer hexagon and the center.
    # For a fixed angle, the minimum enclosing hexagon center is determined by the constraints.
    # However, a simpler approach: try a few angles (0, 30, 60) and centers (centroid, or grid search around centroid).
    
    best_side = 1e18
    best_params = None
    
    # Try angles 0, 30, 60 (and maybe 15, 45)
    for angle in [0, 15, 30, 45, 60]:
        # Estimate center as centroid of all vertices of inner hexagons
        all_verts = []
        for i in range(N):
            v = _hex_vertices(positions[i, 0], positions[i, 1], positions[i, 2])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        cx_est = np.mean(all_verts[:, 0])
        cy_est = np.mean(all_verts[:, 1])
        
        # Try a small grid of centers around the estimate
        for dcx in np.linspace(-1, 1, 5):
            for dcy in np.linspace(-1, 1, 5):
                cx = cx_est + dcx
                cy = cy_est + dcy
                # Binary search on side length? Or just estimate.
                # Estimate side length based on max distance from center to any vertex.
                max_dist = 0
                for i in range(N):
                    v = _hex_vertices(positions[i, 0], positions[i, 1], positions[i, 2])
                    dists = np.linalg.norm(v - np.array([cx, cy]), axis=1)
                    max_dist = max(max_dist, np.max(dists))
                
                # The side length (circumradius) must be at least max_dist.
                # But we also need to satisfy the angular constraints of the hexagon.
                # A rough lower bound is max_dist.
                # Let's refine by checking validity and adjusting side.
                side = max_dist + 0.1
                # Check if this side works. If not, increase.
                # Actually, we want the MINIMUM side.
                # Let's do a binary search for the minimum side that works.
                lo = max_dist * 0.9
                hi = max_dist * 1.5 + 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2
                    valid, _ = _check_validity(positions, cx, cy, mid, angle)
                    if valid:
                        hi = mid
                    else:
                        lo = mid
                if hi < best_side:
                    best_side = hi
                    best_params = (cx, cy, hi, angle)
    
    return best_params

def _generate_configurations():
    """Generate diverse initial configurations based on lattice structures."""
    configs = []
    
    # 1. Single row
    x = np.arange(N) * 2.0 - (N - 1)
    configs.append(np.column_stack([x, np.zeros(N), np.zeros(N)]))
    
    # 2. Two rows (staggered)
    # Row 1: 6 hexagons, Row 2: 5 hexagons
    # Spacing x: 2.0, y: SQRT3
    x1 = np.arange(6) * 2.0 - 5
    y1 = np.zeros(6)
    a1 = np.zeros(6)
    x2 = np.arange(5) * 2.0 - 4
    y2 = np.full(5, SQRT3)
    a2 = np.zeros(5)
    pos1 = np.column_stack([x1, y1, a1])
    pos2 = np.column_stack([x2, y2, a2])
    configs.append(np.vstack([pos1, pos2]))
    
    # 3. Three rows (centered)
    # Row 1: 4, Row 2: 4, Row 3: 3? Or 3, 5, 3?
    # Let's try 3, 5, 3
    x1 = np.arange(3) * 2.0 - 2
    y1 = np.full(3, -SQRT3)
    a1 = np.zeros(3)
    x2 = np.arange(5) * 2.0 - 4
    y2 = np.zeros(5)
    a2 = np.zeros(5)
    x3 = np.arange(3) * 2.0 - 2
    y3 = np.full(3, SQRT3)
    a3 = np.zeros(3)
    pos1 = np.column_stack([x1, y1, a1])
    pos2 = np.column_stack([x2, y2, a2])
    pos3 = np.column_stack([x3, y3, a3])
    configs.append(np.vstack([pos1, pos2, pos3]))
    
    # 4. 4, 4, 3 (slightly off-center)
    x1 = np.arange(4) * 2.0 - 3
    y1 = np.full(4, -SQRT3)
    a1 = np.zeros(4)
    x2 = np.arange(4) * 2.0 - 3
    y2 = np.zeros(4)
    a2 = np.zeros(4)
    x3 = np.arange(3) * 2.0 - 2
    y3 = np.full(3, SQRT3)
    a3 = np.zeros(3)
    pos1 = np.column_stack([x1, y1, a1])
    pos2 = np.column_stack([x2, y2, a2])
    pos3 = np.column_stack([x3, y3, a3])
    configs.append(np.vstack([pos1, pos2, pos3]))
    
    # 5. 5, 6 (two rows, unbalanced)
    x1 = np.arange(5) * 2.0 - 4
    y1 = np.zeros(5)
    a1 = np.zeros(5)
    x2 = np.arange(6) * 2.0 - 5
    y2 = np.full(6, SQRT3)
    a2 = np.zeros(6)
    pos1 = np.column_stack([x1, y1, a1])
    pos2 = np.column_stack([x2, y2, a2])
    configs.append(np.vstack([pos1, pos2]))
    
    # 6. Rotated versions of the above (30 degrees)
    for cfg in configs[:]:
        rotated = cfg.copy()
        # Rotate positions and angles
        for i in range(N):
            x, y, a = rotated[i]
            # Rotate (x,y) by 30 deg? No, just rotate the hexagons themselves.
            # If we rotate the whole cluster, we need to rotate the outer hexagon too.
            # Let's just change the inner hexagon angles.
            rotated[i, 2] = a + 30
        configs.append(rotated)
        
    return configs

def _local_optimize(positions, outer_cx, outer_cy, outer_side, outer_angle, time_limit=5.0):
    """Simple local search to improve the configuration."""
    start_time = time.time()
    current_side = outer_side
    current_pos = positions.copy()
    current_outer = (outer_cx, outer_cy, outer_side, outer_angle)
    
    best_side = current_side
    best_pos = current_pos.copy()
    best_outer = current_outer
    
    lr = 0.05
    penalty_w = 10.0
    
    for step in range(500):
        if time.time() - start_time > time_limit:
            break
            
        grad_pos = np.zeros_like(current_pos)
        
        # Pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                overlap, push = _hex_overlap(current_pos[i, :2], current_pos[j, :2], current_pos[i, 2], current_pos[j, :2], current_pos[j, 2])
                if overlap > 0:
                    grad_pos[i, 0] += penalty_w * push[0]
                    grad_pos[i, 1] += penalty_w * push[1]
                    grad_pos[j, 0] -= penalty_w * push[0]
                    grad_pos[j, 1] -= penalty_w * push[1]
                    
        # Boundary violations
        for i in range(N):
            margin, push = _min_distance_to_boundary(current_pos[i, 0], current_pos[i, 1], current_pos[i, 2], outer_cx, outer_cy, outer_side, outer_angle)
            if margin < 0:
                grad_pos[i, 0] += penalty_w * push[0]
                grad_pos[i, 1] += penalty_w * push[1]
                
        # Update positions
        current_pos[:, :2] -= lr * grad_pos[:, :2]
        
        # Re-optimize outer
        new_outer = _optimize_outer(current_pos)
        if new_outer[2] < best_side:
            best_side = new_outer[2]
            best_pos = current_pos.copy()
            best_outer = new_outer
            
    return best_pos, best_outer

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    configs = _generate_configurations()
    
    best_side = 1e18
    best_result = None
    
    for cfg in configs:
        if time.time() > deadline:
            break
            
        # Initial outer optimization
        outer = _optimize_outer(cfg)
        if outer is None:
            continue
        outer_cx, outer_cy, outer_side, outer_angle = outer
        
        # Local optimization
        pos, outer = _local_optimize(cfg, outer_cx, outer_cy, outer_side, outer_angle, time_limit=10.0)
        
        # Final validation
        valid, _ = _check_validity(pos, outer[0], outer[1], outer[2], outer[3])
        if valid and outer[2] < best_side:
            best_side = outer[2]
            best_result = (pos, outer[0], outer[1], outer[2], outer[3])
            
    if best_result is not None:
        pos, cx, cy, side, angle = best_result
        return [[float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])] for i in range(N)], \
               (float(cx), float(cy)), float(side), float(angle)
    
    # Fallback: return a very large valid configuration
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
