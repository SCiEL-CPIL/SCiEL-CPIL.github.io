import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 115.0  # Safety margin below 150s

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
            return False  # Found separating axis -> no overlap
    return True  # No separating axis -> overlap

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # 6 outward normals of the outer hexagon
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
    
    def make_flower_initial():
        """
        Create a 'flower' like arrangement.
        1 center, 6 around, 5 in gaps.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons touching the center
        r1 = math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        # We have 7. Need 5 more.
        # Place them in the 'valleys' of the first ring, further out.
        r2 = r1 + math.sqrt(3)
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
            
        return pts[:12]

    def make_lattice_initial():
        """
        Hexagonal lattice 3-4-3 arrangement (10 hexes) + 2 extra.
        """
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # Rows: 3, 4, 3
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        
        x_end1 = -col_width*2 - col_width/2 - col_width
        x_end2 = col_width*2 - col_width/2 + col_width
        pts.append([x_end1, 0.0, 0.0])
        pts.append([x_end2, 0.0, 0.0])
        
        return pts

    def make_random_initial():
        """Random scatter."""
        pts = []
        for _ in range(N):
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    def make_hex_tiling_initial():
        """
        Use hexagonal tiling coordinates.
        Place hexagons on a triangular lattice.
        """
        pts = []
        # Hexagonal tiling: centers form a triangular lattice
        # Basis vectors for triangular lattice with spacing sqrt(3)
        # (distance between centers of adjacent hexagons in tiling)
        dx = math.sqrt(3)
        dy = 0.0
        dx2 = math.sqrt(3) / 2
        dy2 = 1.5
        
        # Use axial coordinates (q, r) for hex grid
        # Convert to cartesian: x = dx * (q + r/2), y = dy2 * r
        # Pick 12 positions in a compact cluster
        positions = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        
        for q, r in positions:
            x = dx * (q + r / 2.0)
            y = dy2 * r
            pts.append([x, y, 0.0])
            
        return pts[:12]

    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Generate diverse initial configurations
    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_hex_tiling_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    # First pass: evaluate initial configs
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 15:
            break
            
        cfg = init_func()
        if len(cfg) != N:
            continue
            
        if _is_valid(cfg):
            side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = [row[:] for row in cfg]
                best_center = (0.0, 0.0)
                best_angle = 0.0

    if best_cfg is None:
        # Fallback: use a loose configuration
        cfg = make_lattice_initial()
        best_cfg = cfg
        best_side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # Simulated Annealing with multiple restarts
    def run_sa(cfg_init, max_time):
        """Run simulated annealing starting from cfg_init."""
        nonlocal best_side, best_cfg, best_center, best_angle
        
        cfg = [row[:] for row in cfg_init]
        cur_score = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
        
        if not _is_valid(cfg):
            # Try to fix by spreading out
            for i in range(N):
                cfg[i][0] *= 1.2
                cfg[i][1] *= 1.2
            if not _is_valid(cfg):
                return None, None
            
            cur_score = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
        
        T = 0.3
        T_end = 0.001
        alpha = 0.995
        local_best = cur_score
        local_best_cfg = [row[:] for row in cfg]
        
        while time.time() - t0 < TIME_LIMIT - 5:
            if time.time() - t0 > max_time:
                break
                
            # Move type: 80% position, 20% rotation
            if random.random() < 0.8:
                idx = random.randrange(N)
                old_x, old_y, old_a = cfg[idx]
                cfg[idx][0] += random.gauss(0, 0.15)
                cfg[idx][1] += random.gauss(0, 0.15)
                cfg[idx][2] += random.gauss(0, 3.0)
                move_type = 'pos'
            else:
                idx = random.randrange(N)
                old_a = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 8.0)
                move_type = 'rot'
            
            if not _is_valid(cfg):
                # Revert
                if move_type == 'pos':
                    cfg[idx] = [old_x, old_y, old_a]
                else:
                    cfg[idx][2] = old_a
                continue
            
            new_score = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
            
            # Accept if better or with probability based on temperature
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < local_best:
                    local_best = new_score
                    local_best_cfg = [row[:] for row in cfg]
            else:
                if move_type == 'pos':
                    cfg[idx] = [old_x, old_y, old_a]
                else:
                    cfg[idx][2] = old_a
            
            T = max(T_end, T * alpha)
        
        return local_best, local_best_cfg

    # Run multiple SA restarts from different starting points
    sa_starts = []
    if best_cfg:
        sa_starts.append(best_cfg)
    
    for init_func in [make_flower_initial, make_lattice_initial, make_hex_tiling_initial]:
        cfg = init_func()
        if _is_valid(cfg):
            sa_starts.append(cfg)
    
    # Add some random starts
    for _ in range(3):
        cfg = make_random_initial()
        if _is_valid(cfg):
            sa_starts.append(cfg)
    
    # Run SA with time budgeting
    num_sa = len(sa_starts)
    if num_sa > 0:
        time_per_sa = min(15.0, (TIME_LIMIT - 10 - (time.time() - t0)) / num_sa)
        time_per_sa = max(5.0, time_per_sa)
        
        for start_cfg in sa_starts:
            if time.time() - t0 > TIME_LIMIT - 5:
                break
                
            sa_end_time = time.time() + time_per_sa
            local_best, local_best_cfg = run_sa(start_cfg, sa_end_time)
            
            if local_best is not None and local_best < best_side:
                best_side = local_best
                best_cfg = local_best_cfg
                best_center = (0.0, 0.0)
                best_angle = 0.0

    # Final refinement: very short SA from the best found
    if best_cfg and time.time() - t0 < TIME_LIMIT - 3:
        cfg = [row[:] for row in best_cfg]
        cur_score = best_side
        T = 0.05
        T_end = 0.001
        alpha = 0.99
        
        while time.time() - t0 < TIME_LIMIT - 2:
            idx = random.randrange(N)
            old_x, old_y, old_a = cfg[idx]
            cfg[idx][0] += random.gauss(0, 0.05)
            cfg[idx][1] += random.gauss(0, 0.05)
            cfg[idx][2] += random.gauss(0, 1.0)
            
            if not _is_valid(cfg):
                cfg[idx] = [old_x, old_y, old_a]
                continue
            
            new_score = _outer_hex_side(cfg, (0.0, 0.0), 0.0)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [row[:] for row in cfg]
            else:
                cfg[idx] = [old_x, old_y, old_a]
            
            T = max(T_end, T * alpha)

    # Final verification
    if not _is_valid(best_cfg):
        cfg = make_lattice_initial()
        best_cfg = cfg
        best_side = _outer_hex_side(cfg, (0.0, 0.0), 0.0)

    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
