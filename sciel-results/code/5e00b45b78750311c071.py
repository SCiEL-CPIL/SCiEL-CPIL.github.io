"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import sys

N = 12
TIME_LIMIT = 118.0  # safety margin below 120s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _make_hex_lattice():
    """Create a hexagonal lattice arrangement of 12 unit hexagons.
    
    Arrangement: rows of 4, 4, 4 with hexagonal offset.
    Unit hexagons (flat-top) in hexagonal packing have:
    - horizontal spacing: 2 * (3/2) = 3 (for same row)
    - Actually for flat-top hexagons side 1: width = 2, height = sqrt(3)
    - Hexagonal packing: dx = 1.5, dy = sqrt(3)/2 for adjacent
    """
    pts = []
    # Use pointy-top orientation for natural hexagonal packing
    # For pointy-top: width = sqrt(3), height = 2
    # Adjacent in same row: dx = sqrt(3)
    # Adjacent in next row: dx = sqrt(3)/2, dy = 1.5
    
    # Let's use flat-top for consistency with the angle convention
    # Flat-top hexagon side 1: vertices at angles 0, 60, 120, 180, 240, 300
    # Width (horizontal) = 2, Height (vertical) = sqrt(3)
    # For hexagonal packing of flat-top hexagons:
    # Same row: dx = 2 * (3/2) = 3? No...
    # Actually, two flat-top hexagons side by side share an edge, center distance = sqrt(3)
    # Wait, let me reconsider.
    
    # Flat-top hexagon: vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Width = 2, Height = sqrt(3)
    # Two hexagons sharing a vertical edge: center distance = sqrt(3)
    # Two hexagons in a row (sharing top/bottom edges): center distance = 1.5 horizontally, sqrt(3)/2 vertically
    
    # For 12 hexagons in a 4-4-4 arrangement:
    row_height = math.sqrt(3)  # vertical spacing between rows
    col_width = 3.0  # horizontal spacing in same row (for non-adjacent)
    # Actually for hexagonal packing:
    # Row 0: x = 0, 3, 6, 9
    # Row 1: x = 1.5, 4.5, 7.5, 10.5, y = sqrt(3)/2
    # Row 2: x = 0, 3, 6, 9, y = sqrt(3)
    
    # Let me use the standard hexagonal grid spacing
    # For flat-top hexagons with side 1:
    # Horizontal neighbor distance: sqrt(3)
    # Diagonal neighbor: 1.5 horizontal, sqrt(3)/2 vertical
    
    # 4 per row, 3 rows
    h = math.sqrt(3) / 2.0
    for row in range(3):
        for col in range(4):
            x = col * 3.0 + (1.5 if row % 2 == 1 else 0.0)
            y = row * h * 2.0  # row * sqrt(3)
            pts.append([x - 4.5, y - h, 0.0])
    return pts

def _make_optimized_initial():
    """Better initial configuration using known good packing patterns."""
    # Try a 3-4-3-2 or other arrangement
    # For 12 hexagons, a compact arrangement might be:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    pts = []
    h = math.sqrt(3) / 2.0
    # Center the arrangement
    for row in range(3):
        for col in range(4):
            x = col * 3.0 + (1.5 if row % 2 == 1 else 0.0)
            y = row * 2.0 * h
            pts.append([x - 4.5, y - h, 0.0])
    return pts

def _local_search(cfg, center, angle, t0, time_limit, initial_temp=1.0):
    """Run local search (simulated annealing) from initial configuration."""
    T = initial_temp
    T_end = 0.01
    alpha = 0.9995
    
    cur_score = _outer_hex_side(cfg, tuple(center), angle) if _is_valid(cfg) else None
    if cur_score is None:
        # Spread out
        for p in cfg:
            p[0] *= 1.3
            p[1] *= 1.3
        cur_score = _outer_hex_side(cfg, tuple(center), angle) if _is_valid(cfg) else 100.0
    
    best_score = cur_score
    best_cfg = [row[:] for row in cfg]
    best_center = center[:]
    best_angle = angle
    
    while time.time() - t0 < time_limit:
        r = random.random()
        if r < 0.5:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.2 * (T / initial_temp + 0.1)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.7:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 * (T / initial_temp + 0.1))
        elif r < 0.85:
            old_cx, old_cy = center[0], center[1]
            step = 0.1 * (T / initial_temp + 0.1)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        else:
            old_angle = angle
            angle += random.gauss(0, 2.0 * (T / initial_temp + 0.1))
        
        new_score = None
        if _is_valid(cfg):
            new_score = _outer_hex_side(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / max(T, 1e-10))):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.5:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.7:
                    cfg[idx][2] = old_ang
                elif r < 0.85:
                    center[0], center[1] = old_cx, old_cy
                else:
                    angle = old_angle
        else:
            if r < 0.5:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.7:
                cfg[idx][2] = old_ang
            elif r < 0.85:
                center[0], center[1] = old_cx, old_cy
            else:
                angle = old_angle
        
        T = max(T_end, T * alpha)
    
    return best_score, best_cfg, best_center, best_angle

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Try multiple initial configurations and restarts
    best_overall = 1000.0
    best_inner = None
    best_center = None
    best_angle = None
    
    # Initial configurations to try
    initial_configs = []
    
    # Config 1: Hexagonal lattice 4-4-4
    cfg1 = _make_hex_lattice()
    initial_configs.append((cfg1, [0.0, 0.0], 0.0))
    
    # Config 2: Slightly different arrangement
    cfg2 = _make_optimized_initial()
    initial_configs.append((cfg2, [0.0, 0.0], 0.0))
    
    # Config 3: Random spread
    cfg3 = []
    for i in range(N):
        cfg3.append([random.gauss(0, 3.0), random.gauss(0, 3.0), random.uniform(0, 60)])
    initial_configs.append((cfg3, [0.0, 0.0], 0.0))
    
    # Config 4: Tighter hexagonal arrangement
    cfg4 = []
    h = math.sqrt(3) / 2.0
    # 3-4-3-2 arrangement
    rows = [(0, 3), (1, 4), (2, 3), (3, 2)]
    for row_idx, (row, count) in enumerate(rows):
        for col in range(count):
            x = col * 3.0 + (1.5 if row % 2 == 1 else 0.0)
            y = row * 2.0 * h
            cfg4.append([x - 3.0, y - 1.5 * h, 0.0])
    initial_configs.append((cfg4, [0.0, 0.0], 0.0))
    
    # Allocate time across restarts
    n_restarts = len(initial_configs)
    time_per_restart = (TIME_LIMIT - 5.0) / n_restarts
    
    for restart in range(n_restarts):
        remaining = TIME_LIMIT - (time.time() - t0)
        if remaining < 10.0:
            break
        
        cfg = [row[:] for row in initial_configs[restart][0]]
        center = initial_configs[restart][1][:]
        angle = initial_configs[restart][2]
        
        # First, try to make it valid by spreading if needed
        if not _is_valid(cfg):
            for p in cfg:
                p[0] *= 1.5
                p[1] *= 1.5
            if not _is_valid(cfg):
                for i, p in enumerate(cfg):
                    p[0] = (i % 4 - 1.5) * 4.0
                    p[1] = (i // 4 - 1.0) * 4.0
        
        alloc_time = min(time_per_restart, remaining - 5.0)
        if alloc_time < 5.0:
            continue
        
        score, cfg_out, center_out, angle_out = _local_search(
            cfg, center, angle, t0, t0 + alloc_time, initial_temp=0.5
        )
        
        if score < best_overall:
            best_overall = score
            best_inner = [row[:] for row in cfg_out]
            best_center = center_out[:]
            best_angle = angle_out
    
    # If we still have time, do more restarts from the best solution
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > 15.0 and best_inner is not None:
        # Perturb best and search again
        cfg = [row[:] for row in best_inner]
        for p in cfg:
            p[0] += random.gauss(0, 0.3)
            p[1] += random.gauss(0, 0.3)
            p[2] += random.gauss(0, 3.0)
        
        center = best_center[:]
        angle = best_angle
        
        if not _is_valid(cfg):
            for p in cfg:
                p[0] *= 1.2
                p[1] *= 1.2
            if not _is_valid(cfg):
                cfg = [row[:] for row in best_inner]
        
        alloc_time = min(remaining - 5.0, 40.0)
        if alloc_time > 5.0:
            score, cfg_out, center_out, angle_out = _local_search(
                cfg, center, angle, t0, t0 + alloc_time, initial_temp=0.3
            )
            if score < best_overall:
                best_overall = score
                best_inner = [row[:] for row in cfg_out]
                best_center = center_out[:]
                best_angle = angle_out
    
    # Final check
    if best_inner is None:
        best_inner = _make_hex_lattice()
        best_center = [0.0, 0.0]
        best_angle = 0.0
        best_overall = _outer_hex_side(best_inner, tuple(best_center), best_angle)
    
    inner = [[p[0], p[1], p[2]] for p in best_inner]
    return inner, (best_center[0], best_center[1]), best_overall, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
