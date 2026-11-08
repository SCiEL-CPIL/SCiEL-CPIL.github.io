"""Hexagon packing: place N unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)
# Half-width (across flats) of unit hexagon = SQRT3/2
HF = SQRT3 / 2.0
# Full width across flats
FW = SQRT3

# EVOLVE-BLOCK-START

def _hexagon_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit hexagon (circumradius 1) centered at (cx,cy), rotated by angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return cx + np.cos(angles), cy + np.sin(angles)


def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside a regular hexagon with center (cx,cy), side length, rotation."""
    dx = px - cx
    dy = py - cy
    a = -np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # Regular hexagon with circumradius = side
    # The hexagon is the intersection of 6 half-planes
    if abs(lx) > side or abs(ly) > SQRT3 / 2 * side:
        return False
    if abs(ly) > SQRT3 * (side - abs(lx)):
        return False
    return True


def _dist_point_to_edge(px, py, x1, y1, x2, y2):
    """Return (distance, closest_point) from point (px,py) to segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return np.hypot(px - x1, py - y1), (x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = np.clip(t, 0.0, 1.0)
    cx = x1 + t * dx
    cy = y1 + t * dy
    return np.hypot(px - cx, py - cy), (cx, cy)


def _hexagon_min_distance_to_boundary(vertices, outer_vertices):
    """For each vertex of inner hexagon, find min distance to outer hexagon boundary.
    Negative distance means outside. Returns (min_margin, push_direction)."""
    min_margin = 1e18
    best_push = np.array([0.0, 0.0])
    for i in range(6):
        px, py = vertices[i]
        # Check if inside
        # Find closest point on outer boundary
        best_d = 1e18
        best_cp = (0, 0)
        for j in range(6):
            x1, y1 = outer_vertices[j]
            x2, y2 = outer_vertices[(j + 1) % 6]
            d, cp = _dist_point_to_edge(px, py, x1, y1, x2, y2)
            if d < best_d:
                best_d = d
                best_cp = cp
        # Check if point is inside the outer hexagon
        # A point is inside if it's on the same side of all edges
        inside = True
        for j in range(6):
            x1, y1 = outer_vertices[j]
            x2, y2 = outer_vertices[(j + 1) % 6]
            # Cross product to check side
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            # Point is outside; distance is best_d
            margin = -best_d
            if margin < min_margin:
                min_margin = margin
                # Push direction: from closest boundary point toward interior
                # The interior is toward the center of outer hexagon
                # Push the inner hexagon away from the boundary point
                push = np.array([px - best_cp[0], py - best_cp[1]])
                norm = np.linalg.norm(push)
                if norm > 1e-9:
                    push /= norm
                else:
                    push = np.array([0.0, 0.0])
                best_push = push
    return min_margin, best_push


def _hexagon_overlap_distance(v1, v2):
    """Compute minimum distance between two hexagons given their vertices.
    Negative means overlap. Returns (min_dist, push_1, push_2) where pushes separate the hexagons."""
    min_dist = 1e18
    best_push = np.array([0.0, 0.0])
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            q1 = v2[j]
            q2 = v2[(j + 1) % 6]
            # Segment-segment distance
            d1 = p2 - p1
            d2 = q2 - q1
            r = p1 - q1
            a = np.dot(d1, d1)
            b = np.dot(d1, d2)
            c = np.dot(d2, d2)
            d = np.dot(d1, r)
            e = np.dot(d2, r)
            denom = a * c - b * b
            if denom < 1e-12:
                denom = 1e-12
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
            s = np.clip(s, 0, 1)
            t = np.clip(t, 0, 1)
            closest_p = p1 + s * d1
            closest_q = q1 + t * d2
            diff = closest_p - closest_q
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                min_dist = dist
                best_push = diff
    # Also check vertex-vertex
    for i in range(6):
        for j in range(6):
            diff = v1[i] - v2[j]
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                min_dist = dist
                best_push = diff
    return min_dist, best_push


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Try multiple initial configurations
    best_result = None
    best_score = 1e18
    
    # Configuration 1: Row of hexagons
    # Configuration 2: Two rows staggered
    # Configuration 3: Cluster/honeycomb arrangement
    
    # For a row: N hexagons spaced by 2 (touching)
    # Total span in x: (N-1)*2 + 2 = 2N
    # For flat-top hexagon, width across flats = SQRT3 * side
    # So outer side >= 2N / SQRT3
    
    # For honeycomb: 2 rows, staggered
    # Row 1: 6 hexagons, Row 2: 5 hexagons (staggered)
    # Or Row 1: 5, Row 2: 6
    
    # Let's use a systematic approach with gradient-based local search
    # over positions and outer side length
    
    configs = []
    
    # Config A: Single row
    row_positions = []
    for i in range(N):
        x = (i - (N - 1) / 2.0) * 2.0
        y = 0.0
        row_positions.append([x, y, 0.0])
    configs.append(np.array(row_positions))
    
    # Config B: Two rows, 6+5
    two_row_positions = []
    # Row 1 (top): 6 hexagons at y = SQRT3/2
    for i in range(6):
        x = (i - 2.5) * 2.0
        y = HF
        two_row_positions.append([x, y, 0.0])
    # Row 2 (bottom): 5 hexagons at y = -SQRT3/2, staggered
    for i in range(5):
        x = (i - 2.0) * 2.0
        y = -HF
        two_row_positions.append([x, y, 0.0])
    configs.append(np.array(two_row_positions))
    
    # Config C: Two rows, 5+6
    two_row_positions2 = []
    for i in range(5):
        x = (i - 2.0) * 2.0
        y = HF
        two_row_positions2.append([x, y, 0.0])
    for i in range(6):
        x = (i - 2.5) * 2.0
        y = -HF
        two_row_positions2.append([x, y, 0.0])
    configs.append(np.array(two_row_positions2))
    
    # Config D: Honeycomb cluster - 3-4-3-1 or similar
    # Actually for N=11, a good honeycomb is 1-3-5-3-1 but that's 13
    # Let's try 2-3-4-2 = 11
    honeycomb = []
    # Row -2: 2 hexagons
    for i in range(2):
        x = (i - 0.5) * 2.0
        y = -1.5 * HF
        honeycomb.append([x, y, 0.0])
    # Row -1: 3 hexagons
    for i in range(3):
        x = (i - 1.0) * 2.0
        y = -0.5 * HF
        honeycomb.append([x, y, 0.0])
    # Row 0: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * 2.0
        y = 0.0
        honeycomb.append([x, y, 0.0])
    # Row 1: 2 hexagons
    for i in range(2):
        x = (i - 0.5) * 2.0
        y = 0.5 * HF
        honeycomb.append([x, y, 0.0])
    configs.append(np.array(honeycomb))
    
    # Config E: Compact cluster - 3-5-3 = 11
    compact = []
    # Row -1: 3 hexagons
    for i in range(3):
        x = (i - 1.0) * 2.0
        y = -HF
        compact.append([x, y, 0.0])
    # Row 0: 5 hexagons
    for i in range(5):
        x = (i - 2.0) * 2.0
        y = 0.0
        compact.append([x, y, 0.0])
    # Row 1: 3 hexagons
    for i in range(3):
        x = (i - 1.0) * 2.0
        y = HF
        compact.append([x, y, 0.0])
    configs.append(np.array(compact))
    
    # Config F: Compact cluster - 4-3-4 = 11
    compact2 = []
    # Row -1: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * 2.0
        y = -HF
        compact2.append([x, y, 0.0])
    # Row 0: 3 hexagons
    for i in range(3):
        x = (i - 1.0) * 2.0
        y = 0.0
        compact2.append([x, y, 0.0])
    # Row 1: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * 2.0
        y = HF
        compact2.append([x, y, 0.0])
    configs.append(np.array(compact2))
    
    # Config G: Compact cluster - 5-1-5 = 11 (less compact)
    # Config H: 3-4-4 = 11
    compact3 = []
    # Row -1: 3 hexagons
    for i in range(3):
        x = (i - 1.0) * 2.0
        y = -HF
        compact3.append([x, y, 0.0])
    # Row 0: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * 2.0
        y = 0.0
        compact3.append([x, y, 0.0])
    # Row 1: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * 2.0
        y = HF
        compact3.append([x, y, 0.0])
    configs.append(np.array(compact3))
    
    for cfg_idx, positions in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Estimate initial outer side
        # Bounding box of positions
        xs = positions[:, 0]
        ys = positions[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # For flat-top hexagon (angle=0), the containment condition for a point at distance (dx, dy) from center:
        # max(|dx|, |dy|/HF, (|dx| + |dy|/SQRT3)/1.0) <= side * something
        # More precisely, a point is inside a flat-top hexagon of side S centered at origin if:
        # |dx| <= S, |dy| <= HF*S, and |dy| <= SQRT3*(S - |dx|)
        
        # The required S must satisfy for all 6 vertices of all inner hexagons:
        # For each vertex (vx, vy) relative to outer center:
        # |vx| <= S, |vy| <= HF*S, |vy| <= SQRT3*(S - |vx|)
        # S >= |vx|
        # S >= |vy| / HF
        # S >= (|vx| + |vy|/SQRT3)  [from |vy| <= SQRT3*(S - |vx|) => S >= |vx| + |vy|/SQRT3]
        
        # But we need to account for the inner hexagon vertices, not just centers.
        # Inner hexagon vertices are at distance 1 from center in 6 directions.
        
        # Let's just start with a generous estimate and let optimization shrink it.
        # Max extent:
        max_x = max(abs(x_min), abs(x_max)) + 1.0  # +1 for hexagon circumradius
        max_y = max(abs(y_min), abs(y_max)) + HF
        outer_side = max(max_x, max_y / HF, (max_x + max_y / SQRT3)) + 0.5
        
        outer_cx, outer_cy = 0.0, 0.0
        outer_angle = 0.0  # flat-top
        
        # Gradient descent
        lr_pos = 0.02
        lr_side = 0.01
        penalty_weight = 50.0
        margin = 0.005
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Precompute outer hexagon vertices
            a_outer = np.deg2rad(outer_angle)
            outer_angles = a_outer + np.arange(6) * np.pi / 3.0
            ovx = outer_cx + outer_side * np.cos(outer_angles)
            ovy = outer_cy + outer_side * np.sin(outer_angles)
            outer_vertices = np.column_stack([ovx, ovy])
            
            total_loss = 0.0
            grad_pos = np.zeros_like(positions)
            grad_side = 0.0
            
            # Compute all inner hexagon vertices
            inner_vx = np.zeros((N, 6))
            inner_vy = np.zeros((N, 6))
            for i in range(N):
                vx, vy = _hexagon_vertices(positions[i, 0], positions[i, 1], positions[i, 2])
                inner_vx[i] = vx
                inner_vy[i] = vy
            
            # Pairwise overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    v1 = np.column_stack([inner_vx[i], inner_vy[i]])
                    v2 = np.column_stack([inner_vx[j], inner_vy[j]])
                    dist, push = _hexagon_overlap_distance(v1, v2)
                    if dist < margin:
                        penalty = (margin - dist) ** 2
                        total_loss += penalty_weight * penalty
                        g = penalty_weight * 2 * (margin - dist)
                        grad_pos[i, 0] += g * push[0]
                        grad_pos[i, 1] += g * push[1]
                        grad_pos[j, 0] -= g * push[0]
                        grad_pos[j, 1] -= g * push[1]
            
            # Containment
            for i in range(N):
                v = np.column_stack([inner_vx[i], inner_vy[i]])
                min_margin_i, push_i = _hexagon_min_distance_to_boundary(v, outer_vertices)
                if min_margin_i < margin:
                    penalty = (margin - min_margin_i) ** 2
                    total_loss += penalty_weight * penalty
                    g = penalty_weight * 2 * (margin - min_margin_i)
                    grad_pos[i, 0] += g * push_i[0]
                    grad_pos[i, 1] += g * push_i[1]
                    grad_side -= g  # shrinking side makes containment harder
            
            # Objective: outer_side + penalty
            total_loss += outer_side
            
            # Update
            positions -= lr_pos * grad_pos
            outer_side -= lr_side * grad_side
            outer_side = max(outer_side, 5.0)
            
            # Keep centered
            cx_mean = np.mean(positions[:, 0])
            cy_mean = np.mean(positions[:, 1])
            positions[:, 0] -= 0.05 * cx_mean
            positions[:, 1] -= 0.05 * cy_mean
            
            # Periodic validation
            if step % 50 == 0:
                valid = True
                # Check overlaps
                for i in range(N):
                    for j in range(i + 1, N):
                        v1 = np.column_stack([inner_vx[i], inner_vy[i]])
                        v2 = np.column_stack([inner_vx[j], inner_vy[j]])
                        dist, _ = _hexagon_overlap_distance(v1, v2)
                        if dist < -0.01:
                            valid = False
                # Check containment
                if valid:
                    a_outer = np.deg2rad(outer_angle)
                    outer_angles = a_outer + np.arange(6) * np.pi / 3.0
                    ovx = outer_cx + outer_side * np.cos(outer_angles)
                    ovy = outer_cy + outer_side * np.sin(outer_angles)
                    outer_vertices = np.column_stack([ovx, ovy])
                    for i in range(N):
                        v = np.column_stack([inner_vx[i], inner_vy[i]])
                        mm, _ = _hexagon_min_distance_to_boundary(v, outer_vertices)
                        if mm < -0.01:
                            valid = False
                if valid:
                    score = outer_side
                    if score < best_score:
                        best_score = score
                        best_result = (positions.copy(), outer_cx, outer_cy, outer_side, outer_angle)
        
        # Final validation for this config
        if best_result is not None:
            positions_r, ocx, ocy, oside, oang = best_result
            # Verify
            a_outer = np.deg2rad(oang)
            outer_angles = a_outer + np.arange(6) * np.pi / 3.0
            ovx = ocx + oside * np.cos(outer_angles)
            ovy = ocy + oside * np.sin(outer_angles)
            outer_vertices = np.column_stack([ovx, ovy])
            valid = True
            for i in range(N):
                vx, vy = _hexagon_vertices(positions_r[i, 0], positions_r[i, 1], positions_r[i, 2])
                v = np.column_stack([vx, vy])
                mm, _ = _hexagon_min_distance_to_boundary(v, outer_vertices)
                if mm < -0.001:
                    valid = False
            for i in range(N):
                for j in range(i + 1, N):
                    vx1, vy1 = _hexagon_vertices(positions_r[i, 0], positions_r[i, 1], positions_r[i, 2])
                    vx2, vy2 = _hexagon_vertices(positions_r[j, 0], positions_r[j, 1], positions_r[j, 2])
                    v1 = np.column_stack([vx1, vy1])
                    v2 = np.column_stack([vx2, vy2])
                    dist, _ = _hexagon_overlap_distance(v1, v2)
                    if dist < -0.001:
                        valid = False
            if valid:
                return [[float(positions_r[i, 0]), float(positions_r[i, 1]), float(positions_r[i, 2])] for i in range(N)], \
                       (float(ocx), float(ocy)), float(oside), float(oang)
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
