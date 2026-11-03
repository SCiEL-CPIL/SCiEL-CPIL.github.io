import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

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
        dx = px - cx
        dy = py - cy
        for k in range(6):
            a = math.radians(angle_deg) + k * math.pi / 3 + math.pi / 6
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

        # Bounding box check for speed
        min_x1, min_y1 = np.min(v1, axis=0)
        max_x1, max_y1 = np.max(v1, axis=0)
        min_x2, min_y2 = np.min(v2, axis=0)
        max_x2, max_y2 = np.max(v2, axis=0)
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return min_x2 - max_x1 if max_x1 < min_x2 else min_x1 - max_x2 # Approx dist if separated in x
            # Actually, if bboxes don't overlap, min dist is at least min of gaps
            dx_gap = max(0, min_x2 - max_x1, min_x1 - max_x2)
            dy_gap = max(0, min_y2 - max_y1, min_y1 - max_y2)
            return math.hypot(dx_gap, dy_gap) - 2.0 # This is an underestimate of true dist if bboxes don't overlap? No, it's the dist between bboxes.
            # Wait, if bboxes don't overlap, the distance between hexes is >= distance between bboxes.
            # But we want exact min dist. If bboxes don't overlap, we can still compute edge dist, but it's likely large.
            # For now, let's just proceed to edge check if bboxes overlap or are close.
            # If bboxes are far, return a large positive number to indicate no overlap.
            if dx_gap > 0.1 or dy_gap > 0.1:
                 return 1.0 # Safe positive value, means "not overlapping" for penalty purposes

        for vx, vy in v1:
            if point_in_hexagon(vx, vy, c2[0], c2[1], 1.0, a2):
                return -1.0
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, c1[0], c1[1], 1.0, a1):
                return -1.0

        min_dist = dc
        edges1 = [(v1[i], v1[(i+1)%6]) for i in range(6)]
        edges2 = [(v2[i], v2[(i+1)%6]) for i in range(6)]
        for e1 in edges1:
            for e2 in edges2:
                d = seg_seg_dist(e1[0], e1[1], e2[0], e2[1])
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
                    penalty += 100.0 * (-d) ** 2

        S = outer_hexagon_size(pos, angs, (cx, cy), outer_rot)
        return S + penalty

    # Discrete orientation patterns
    # Pattern 1: All 0
    # Pattern 2: All 30
    # Pattern 3: Center 0, others 0 (same as 1)
    # Pattern 4: Alternating?
    
    def config_flower():
        pos = [[0.0, 0.0]]
        angs = [0.0]
        for i in range(6):
            a = i * math.pi / 3
            pos.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
            angs.append(0.0)
        for i in range(4):
            a = math.pi/6 + i * math.pi / 2
            pos.append([3.8 * math.cos(a), 3.8 * math.sin(a)])
            angs.append(0.0)
        return pos, angs

    def config_lattice():
        pos = []
        angs = []
        for q in range(-1, 2):
            for r in range(-1, 2):
                if abs(q + r) <= 1 or (q, r) in [(1,1), (-1,-1), (2,0), (-2,0), (0,2), (0,-2)]:
                    x = math.sqrt(3) * (q + r/2.0)
                    y = 1.5 * r
                    pos.append([x, y])
                    angs.append(0.0)
        if len(pos) > N:
            pos = pos[:N]
            angs = angs[:N]
        return pos, angs

    configs = [config_flower(), config_lattice()]
    
    # Discrete rotation patterns to try
    rotation_patterns = [
        [0.0]*N,
        [0.0]*N, # Duplicate for now, can add [30.0]*N
    ]
    # Add a pattern with 30 degree rotation for all
    rotation_patterns.append([30.0]*N)
    
    best_S = float('inf')
    best_params = None

    for pos, angs in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        pos_arr = np.array(pos, dtype=float)
        
        for rot_pattern in rotation_patterns:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            angs_arr = np.array(rot_pattern, dtype=float)
            cx = np.mean(pos_arr[:, 0])
            cy = np.mean(pos_arr[:, 1])

            # Try a few outer rotations
            for outer_rot_deg in [0, 30]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break

                params = np.concatenate([
                    pos_arr.ravel(),
                    angs_arr,
                    [cx, cy],
                    [outer_rot_deg]
                ])

                # Use L-BFGS-B for better performance
                try:
                    res = minimize(objective, params, method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-6})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass

    if best_params is None:
        pos, angs = config_flower()
        pos_arr = np.array(pos, dtype=float)
        angs_arr = np.array(angs, dtype=float)
        cx = np.mean(pos_arr[:, 0])
        cy = np.mean(pos_arr[:, 1])
        best_params = np.concatenate([pos_arr.ravel(), angs_arr, [cx, cy], [0.0]])

    pos = best_params[:2*N].reshape(N, 2)
    angs = best_params[2*N:3*N]
    cx, cy = best_params[3*N:3*N+2]
    outer_rot = best_params[3*N+2]

    S = outer_hexagon_size(pos, angs, (cx, cy), outer_rot)

    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(angs[i])])

    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(outer_rot)

    return inner, center, side, rotation
