"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random

N = 12
TIME_LIMIT = 115.0  # safety margin below 120s

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
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
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

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_initial():
        """Hexagonal lattice arrangement: rows of 4, 4, 4 with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring_initial():
        """12 hexagons arranged in a ring around the center."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_grid_initial():
        """Grid-like arrangement with more spacing."""
        pts = []
        row_h = math.sqrt(3)
        col_w = 2.0
        for r in range(3):
            for c in range(4):
                x = c * col_w + (col_w / 2 if r % 2 else 0.0)
                y = r * row_h
                pts.append([x - 3.0, y - row_h, 0.0])
        return pts
    
    def make_tight_lattice():
        """Tighter hexagonal lattice with proper spacing."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        # Center the configuration
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for p in pts:
            p[0] -= cx
            p[1] -= cy
        return pts
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_initial(), make_ring_initial(), make_grid_initial(), make_tight_lattice()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    # Simulated annealing with adaptive step size
    T = 1.0
    T_end = 0.001
    alpha = 0.9997
    
    # Track for adaptive step sizing
    accepted = 0
    rejected = 0
    step_scale = 1.0
    
    while time.time() - t0 < TIME_LIMIT:
        r = random.random()
        
        if r < 0.45:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.2 * step_scale * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 * step_scale + 3.0 * T)
        elif r < 0.75:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.1 * step_scale * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 2.0 * step_scale + 2.0 * T)
        elif r < 0.95:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.15 * step_scale * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        else:
            # Larger move: perturb 3 hexagons
            idx1, idx2, idx3 = random.sample(range(N), 3)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            old_x3, old_y3 = cfg[idx3][0], cfg[idx3][1]
            step = 0.12 * step_scale * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
            cfg[idx3][0] += random.gauss(0, step)
            cfg[idx3][1] += random.gauss(0, step)
        
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                accepted += 1
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                rejected += 1
                # Revert
                if r < 0.45:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.65:
                    cfg[idx][2] = old_ang
                elif r < 0.75:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.85:
                    angle = old_angle
                elif r < 0.95:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                    cfg[idx3][0], cfg[idx3][1] = old_x3, old_y3
        else:
            rejected += 1
            # Invalid: revert
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            elif r < 0.95:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                cfg[idx3][0], cfg[idx3][1] = old_x3, old_y3
        
        # Adaptive step sizing
        total = accepted + rejected
        if total > 100:
            accept_rate = accepted / total
            if accept_rate < 0.1:
                step_scale = max(0.3, step_scale * 1.2)
            elif accept_rate > 0.8:
                step_scale = min(3.0, step_scale * 0.8)
            accepted = 0
            rejected = 0
        
        T = max(T_end, T * alpha)
        
        # Final polish: when T is low, do finer moves
        if T < 0.01:
            r2 = random.random()
            if r2 < 0.7:
                idx = random.randrange(N)
                old_x, old_y = cfg[idx][0], cfg[idx][1]
                step = 0.02 * step_scale
                cfg[idx][0] += random.gauss(0, step)
                cfg[idx][1] += random.gauss(0, step)
                new_score = full_score(cfg, tuple(center), angle)
                if new_score is not None and new_score < cur_score:
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_cfg = [row[:] for row in cfg]
                        best_center = center[:]
                        best_angle = angle
                else:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
            else:
                idx = random.randrange(N)
                old_ang = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 1.0 * step_scale)
                new_score = full_score(cfg, tuple(center), angle)
                if new_score is not None and new_score < cur_score:
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_cfg = [row[:] for row in cfg]
                        best_center = center[:]
                        best_angle = angle
                else:
                    cfg[idx][2] = old_ang
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
