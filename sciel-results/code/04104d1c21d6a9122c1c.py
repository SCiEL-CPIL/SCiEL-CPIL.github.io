import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]] * N, [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle + i * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        for k in range(6):
            a = angle + k * math.pi / 3 + math.pi / 6
            nx = math.cos(a)
            ny = math.sin(a)
            if dx * nx + dy * ny > APOTHEM * side:
                return False
        return True

    def seg_seg_dist(p1, p2, p3, p4):
        d = min(
            point_seg_dist(p1, p3, p4),
            point_seg_dist(p2, p3, p4),
            point_seg_dist(p3, p1, p2),
            point_seg_dist(p4, p1, p2)
        )
        return d

    def point_seg_dist(p, a, b):
        ab = b - a
        ap = p - a
        denom = np.dot(ab, ab)
        if denom < 1e-12:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / denom
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def min_dist_between_hexes(c1, a1, c2, a2):
        dc = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        if dc >= 2.0:
            return dc - 2.0
        
        v1 = hex_vertices(c1[0], c1[1], 1.0, a1)
        v2 = hex_vertices(c2[0], c2[1], 1.0, a2)
        
        min_dist = dc
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, c2[0], c2[1], 1.0, a2):
                return -1.0
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, c1[0], c1[1], 1.0, a1):
                return -1.0
        
        edges1 = [(v1[i], v1[(i+1)%6]) for i in range(6)]
        edges2 = [(v2[i], v2[(i+1)%6]) for i in range(6)]
        
        for e1 in edges1:
            for e2 in edges2:
                d = seg_seg_dist(e1, e2)
                if d < min_dist:
                    min_dist = d
        
        return min_dist - 2.0

    def outer_hexagon_size(positions, angles, center, rotation_deg):
        max_proj = 0.0
        rot = math.radians(rotation_deg)
        for k in range(6):
            a = rot + k * math.pi / 3 + math.pi / 6
            nx = math.cos(a)
            ny = math.sin(a)
            for i in range(N):
                cx, cy = positions[i]
                v = hex_vertices(cx, cy, 1.0, angles[i])
                for vx, vy in v:
                    proj = (vx - center[0]) * nx + (vy - center[1]) * ny
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
        S = max_proj / APOTHEM
        return S

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        pos = np.array(params[:2*N]).reshape(N, 2)
        angs = params[2*N:3*N]
        cx, cy = params[3*N:3*N+2]
        outer_rot = params[3*N+2]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_between_hexes(pos[i], angs[i], pos[j], angs[j])
                if d < 0:
                    penalty += 50.0 * (-d) ** 2
        
        S = outer_hexagon_size(pos, angs, (cx, cy), outer_rot)
        return S + penalty

    def make_hex_tiling_config(radius_rings=2):
        """Generate positions on a hexagonal grid (axial coordinates)."""
        positions = []
        angles = []
        # Center
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        for ring in range(1, radius_rings + 1):
            # Walk around the ring
            # For hex grid, ring r has 6*r hexagons
            # Start at (r, 0) and walk
            x, y = ring, 0
            for i in range(6 * ring):
                # Convert axial to cartesian
                cx = math.sqrt(3) * (x + y/2.0)
                cy = 1.5 * y
                positions.append([cx, cy])
                angles.append(0.0)
                
                # Next position in ring
                if i < ring:
                    x -= 1
                elif i < 2 * ring:
                    y -= 1
                elif i < 3 * ring:
                    y += 1
                elif i < 4 * ring:
                    x += 1
                elif i < 5 * ring:
                    y += 1
                else:
                    x -= 1
        
        # We need exactly N positions. If we have more, take the first N.
        # If we have less, we need to add more or use a different layout.
        # For N=11, ring 1 has 6, center has 1. Total 7.
        # Ring 2 has 12. Total 19.
        # So we take center + 6 from ring 1 + 4 from ring 2.
        
        if len(positions) > N:
            positions = positions[:N]
            angles = angles[:N]
            
        return positions, angles

    def make_compact_config():
        """A manually tuned compact configuration."""
        # This is a heuristic starting point.
        # Center at (0,0)
        pos = [[0.0, 0.0]]
        angs = [0.0]
        
        # 6 neighbors at distance 2
        for i in range(6):
            a = i * math.pi / 3
            pos.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
            angs.append(0.0)
            
        # 4 more at distance ~3.5, offset angles
        for i in range(4):
            a = math.pi/6 + i * math.pi / 2
            pos.append([3.5 * math.cos(a), 3.5 * math.sin(a)])
            angs.append(0.0)
            
        return pos, angs

    def make_de_config():
        """Random configuration for differential evolution."""
        pos = []
        angs = []
        for i in range(N):
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            pos.append([x, y])
            angs.append(0.0)
        return pos, angs

    # Initial configurations
    configs = [
        make_hex_tiling_config(2),
        make_compact_config(),
        make_de_config(),
        make_de_config(),
        make_de_config()
    ]

    best_S = float('inf')
    best_params = None

    for pos, angs in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pos_arr = np.array(pos)
        angs_arr = np.array(angs)
        
        cx = np.mean(pos_arr[:, 0])
        cy = np.mean(pos_arr[:, 1])
        
        params = np.concatenate([
            pos_arr.ravel(),
            angs_arr,
            [cx, cy],
            [0.0]
        ])
        
        # Try Nelder-Mead
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Try Powell
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                res = minimize(objective, params, method='Powell',
                             options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Differential Evolution for global search
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Bounds: positions [-5, 5], angles [0, 360], center [-5, 5], outer rot [0, 360]
        bounds = [(-5, 5)] * (2*N) + [(0, 360)] * N + [(-5, 5), (-5, 5), (0, 360)]
        
        try:
            res = differential_evolution(objective, bounds, maxiter=200, popsize=30, 
                                        tol=1e-6, seed=42, polish=True)
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback
        pos, angs = make_compact_config()
        pos_arr = np.array(pos)
        angs_arr = np.array(angs)
        cx = np.mean(pos_arr[:, 0])
        cy = np.mean(pos_arr[:, 1])
        best_params = np.concatenate([pos_arr.ravel(), angs_arr, [cx, cy], [0.0]])
        best_S = objective(best_params)

    # Extract final result
    pos = best_params[:2*N].reshape(N, 2)
    angs = best_params[2*N:3*N]
    cx, cy = best_params[3*N:3*N+2]
    outer_rot = best_params[3*N+2]

    # Compute actual outer hexagon size
    S = outer_hexagon_size(pos, angs, (cx, cy), outer_rot)

    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(angs[i])])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(outer_rot)
    
    return inner, center, side, rotation
