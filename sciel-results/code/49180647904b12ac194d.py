import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]] * N, [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle + i * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point is inside hexagon."""
        angle = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        # Rotate point into hexagon's local frame
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # Hexagon with flat top: |lx| <= side, |ly| <= side*sqrt(3)/2, |lx + ly/sqrt(3)| <= side
        # Actually for pointy-top hexagon:
        # Check all 6 half-planes
        for k in range(6):
            a = angle + k * math.pi / 3 + math.pi / 6
            nx = math.cos(a)
            ny = math.sin(a)
            if dx * nx + dy * ny > APOTHEM * side:
                return False
        return True

    def min_dist_between_hexes(c1, a1, c2, a2):
        """
        Compute minimum distance between two unit hexagons.
        If they overlap, return negative (penalty).
        """
        # For same orientation, min center dist is 2
        # For different orientations, it's more complex
        # Approximate: check if any vertex of one is inside the other
        # and compute edge-to-edge distances
        
        # Simple approach: if centers are far enough, they're disjoint
        dc = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        if dc >= 2.0:
            return dc - 2.0
        
        # Check vertex containment
        v1 = hex_vertices(c1[0], c1[1], 1.0, a1)
        v2 = hex_vertices(c2[0], c2[1], 1.0, a2)
        
        min_dist = dc
        # Check if any vertex of hex1 is inside hex2
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, c2[0], c2[1], 1.0, a2):
                return -1.0  # Overlap
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, c1[0], c1[1], 1.0, a1):
                return -1.0  # Overlap
        
        # Compute edge-edge distances (approximate)
        # Use a simpler check: if dc < 2, they might overlap
        # More precise: check all edge pairs
        edges1 = [(v1[i], v1[(i+1)%6]) for i in range(6)]
        edges2 = [(v2[i], v2[(i+1)%6]) for i in range(6)]
        
        for e1 in edges1:
            for e2 in edges2:
                d = seg_seg_dist(e1, e2)
                if d < min_dist:
                    min_dist = d
        
        return min_dist - 2.0  # Negative means overlap

    def seg_seg_dist(p1, p2, p3, p4):
        """Distance between two line segments."""
        # Project and check
        d = min(
            point_seg_dist(p1, p3, p4),
            point_seg_dist(p2, p3, p4),
            point_seg_dist(p3, p1, p2),
            point_seg_dist(p4, p1, p2)
        )
        return d

    def point_seg_dist(p, a, b):
        """Distance from point to line segment."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
        t = max(0, min(1, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def outer_hexagon_size(positions, angles, center, rotation_deg):
        """
        Compute the minimum side length of a regular hexagon (with given rotation)
        that contains all unit hexagons.
        """
        max_proj = 0.0
        rot = math.radians(rotation_deg)
        # The 6 normals of the outer hexagon
        for k in range(6):
            a = rot + k * math.pi / 3 + math.pi / 6
            nx = math.cos(a)
            ny = math.sin(a)
            for i in range(N):
                cx, cy = positions[i]
                # For each hexagon, find max projection of its vertices
                v = hex_vertices(cx, cy, 1.0, angles[i])
                for vx, vy in v:
                    proj = (vx - center[0]) * nx + (vy - center[1]) * ny
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
        
        # Side length S: apothem = S * sqrt(3)/2, so S = apothem / (sqrt(3)/2)
        # The outer hexagon has apothem = max_proj
        S = max_proj / APOTHEM
        return S

    def objective(params):
        """Objective: outer hexagon side length, with penalty for overlaps."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        # Unpack: positions (2N), angles (N), center (2), outer rotation (1)
        # Total: 3N + 3 = 36 params
        pos = np.array(params[:2*N]).reshape(N, 2)
        angs = params[2*N:3*N]
        cx, cy = params[3*N:3*N+2]
        outer_rot = params[3*N+2]
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_between_hexes(pos[i], angs[i], pos[j], angs[j])
                if d < 0:
                    penalty += 10.0 * (-d) ** 2
        
        S = outer_hexagon_size(pos, angs, (cx, cy), outer_rot)
        return S + penalty

    # Initial configurations
    def make_initial_config_1():
        """Central hex + 6 ring + 4 outer."""
        pos = []
        angs = []
        # Center
        pos.append([0.0, 0.0])
        angs.append(0.0)
        # Ring of 6
        for i in range(6):
            a = i * math.pi / 3
            pos.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
            angs.append(0.0)
        # 4 more at radius ~4
        for i in range(4):
            a = math.pi/6 + i * math.pi / 2
            pos.append([4.0 * math.cos(a), 4.0 * math.sin(a)])
            angs.append(0.0)
        return pos, angs

    def make_initial_config_2():
        """Alternative: 3 rows."""
        pos = []
        angs = []
        # Row 1: 4 hexagons
        for i in range(4):
            pos.append([i * 2.0, 0.0])
            angs.append(0.0)
        # Row 2: 4 hexagons offset
        for i in range(4):
            pos.append([i * 2.0 + 1.0, 1.5 * math.sqrt(3)])
            angs.append(0.0)
        # Row 3: 3 hexagons
        for i in range(3):
            pos.append([i * 2.0 + 1.0, 3.0 * math.sqrt(3)])
            angs.append(0.0)
        return pos, angs

    def make_initial_config_3():
        """Symmetric cross pattern."""
        pos = []
        angs = []
        # Center
        pos.append([0.0, 0.0])
        angs.append(0.0)
        # 4 cardinal
        for a in [0, math.pi/2, math.pi, 3*math.pi/2]:
            pos.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
            angs.append(0.0)
        # 4 diagonal
        for a in [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]:
            pos.append([2.5 * math.cos(a), 2.5 * math.sin(a)])
            angs.append(0.0)
        return pos, angs

    def make_initial_config_4():
        """Tight cluster."""
        pos = []
        angs = []
        # Use axial-like but with optimization freedom
        # Center + 6 neighbors + 4 at second ring
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,0), (-2,0), (0,2), (0,-2)]
        for q, r in axial:
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pos.append([x, y])
            angs.append(0.0)
        return pos, angs

    def make_initial_config_5():
        """Another ring-based."""
        pos = []
        angs = []
        pos.append([0.0, 0.0])
        angs.append(0.0)
        for i in range(6):
            a = i * math.pi / 3
            pos.append([2.0 * math.cos(a), 2.0 * math.sin(a)])
            angs.append(0.0)
        # 4 more at 120-degree spacing at radius 3.5
        for i in range(4):
            a = math.pi/3 + i * math.pi / 2
            pos.append([3.5 * math.cos(a), 3.5 * math.sin(a)])
            angs.append(0.0)
        return pos, angs

    configs = [make_initial_config_1(), make_initial_config_2(), 
               make_initial_config_3(), make_initial_config_4(), make_initial_config_5()]

    best_S = float('inf')
    best_params = None

    for pos, angs in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pos_arr = np.array(pos)
        angs_arr = np.array(angs)
        
        # Compute initial center and outer rotation
        cx = np.mean(pos_arr[:, 0])
        cy = np.mean(pos_arr[:, 1])
        
        # Try a few outer rotations
        for outer_rot_deg in [0, 30, 15, 45]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            params = np.concatenate([
                pos_arr.ravel(),
                angs_arr,
                [cx, cy],
                [outer_rot_deg]
            ])
            
            # Nelder-Mead optimization
            try:
                res = minimize(objective, params, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            
            # Also try Powell
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                try:
                    res = minimize(objective, params, method='Powell',
                                 options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass

    if best_params is None:
        # Fallback
        pos, angs = make_initial_config_4()
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

    # Small safety scale
    SCALE = 1.0 + 1e-10
    pos_scaled = pos * SCALE  # Scale positions from center
    # Actually, we should scale the whole configuration
    # But the outer hexagon size already accounts for this

    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(angs[i])])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(outer_rot)
    
    return inner, center, side, rotation
