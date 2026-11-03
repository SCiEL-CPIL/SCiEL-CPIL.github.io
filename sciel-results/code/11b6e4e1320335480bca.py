import math
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
PENALTY_WEIGHT = 500.0

def hex_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def polygon_area(verts):
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
    return abs(area) / 2.0

def point_in_convex_polygon(px, py, verts):
    n = len(verts)
    sign = 0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) > 1e-9:
            if sign == 0:
                sign = 1 if cross > 0 else -1
            elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                return False
    return True

def polygons_disjoint(v1, v2):
    # Check if any vertex of v1 is inside v2 or vice versa
    for i in range(6):
        if point_in_convex_polygon(v1[i][0], v1[i][1], v2):
            return False
        if point_in_convex_polygon(v2[i][0], v2[i][1], v1):
            return False
    # Check edge intersections (simple segment intersection)
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def segments_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    return False

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def is_valid_cluster(pts, angles):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            v1 = hex_vertices(pts[i][0], pts[i][1], 1.0, angles[i])
            v2 = hex_vertices(pts[j][0], pts[j][1], 1.0, angles[j])
            if not polygons_disjoint(v1, v2):
                return False
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a simple valid configuration
        seed_axial = [(0,0),(2,0),(-2,0),(0,2),(0,-2),(2,2),(-2,-2),(2,-2),(-2,2),(4,0),(0,4)]
        pts = axial_to_cartesian(seed_axial)
        angles = [0.0] * N
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        # Compute minimal enclosing hexagon for this configuration
        best_S = float('inf')
        for beta_deg in range(0, 60, 5):
            beta = math.radians(beta_deg)
            angles_norm = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles_norm), np.sin(angles_norm)])
            D = float(np.max(np.abs((pts - np.array([cx, cy])) @ normals.T)))
            S = (D + APOTHEM) / APOTHEM
            if S < best_S:
                best_S = S
        inner = [[float(x), float(y), float(a)] for (x, y), a in zip(pts, angles)]
        return inner, [cx, cy], best_S, 0.0

    random.seed(42)
    np.random.seed(42)

    # Initial valid configuration: hexagonal lattice
    seed_axial = [
        (0, 0),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (2, 2), (-2, -2),
        (2, -2), (-2, 2),
        (4, 0),
        (0, 4)
    ]
    pts_init = axial_to_cartesian(seed_axial)
    angles_init = [0.0] * N

    # State: positions (N*2), outer center (2), outer side (1), outer rotation (1)
    # Total dimensions: 2N + 4 = 26
    best_state = None
    best_score = float('inf')

    # Objective function for scipy minimize
    def objective(state):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        pts = state[:2*N].reshape(N, 2)
        cx, cy = state[2*N], state[2*N+1]
        S = state[2*N+2]
        beta = state[2*N+3]
        
        # Ensure S is positive
        if S < 0.5:
            S = 0.5
        
        penalty = 0.0
        
        # 1. Non-overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                v1 = hex_vertices(pts[i][0], pts[i][1], 1.0, angles_init[i])
                v2 = hex_vertices(pts[j][0], pts[j][1], 1.0, angles_init[j])
                # Approximate distance penalty: if centers are too close, penalize
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < math.sqrt(3) - 1e-6:
                    penalty += (math.sqrt(3) - dist) ** 2 * PENALTY_WEIGHT
        
        # 2. Containment penalty
        outer_verts = hex_vertices(cx, cy, S, math.degrees(beta))
        for i in range(N):
            # Check if all vertices of inner hex i are inside outer hex
            inner_verts = hex_vertices(pts[i][0], pts[i][1], 1.0, angles_init[i])
            for v in inner_verts:
                if not point_in_convex_polygon(v[0], v[1], outer_verts):
                    # Distance to nearest edge (approximate)
                    penalty += 100.0
        
        return S + penalty

    # Local search loop
    current_pts = pts_init.copy()
    current_angles = angles_init.copy()
    current_cx = float(np.mean(current_pts[:, 0]))
    current_cy = float(np.mean(current_pts[:, 1]))
    current_S = 10.0  # Start large
    current_beta = 0.0

    # Initial sizing: find minimal S for current configuration
    def find_min_S(pts, angles, cx, cy, beta):
        best_S = float('inf')
        for b_deg in range(0, 60, 2):
            b = math.radians(b_deg)
            angles_norm = b + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles_norm), np.sin(angles_norm)])
            D = float(np.max(np.abs((pts - np.array([cx, cy])) @ normals.T)))
            S = (D + APOTHEM) / APOTHEM
            if S < best_S:
                best_S = S
        return best_S

    current_S = find_min_S(current_pts, current_angles, current_cx, current_cy, current_beta)
    best_score = current_S
    best_state = (current_pts.copy(), current_angles.copy(), current_cx, current_cy, current_S, current_beta)

    # Iterative local search
    max_iters = 500
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        # Build state vector
        state_vec = np.concatenate([current_pts.flatten(), [current_cx, current_cy, current_S, current_beta]])
        
        # Optimize
        try:
            res = minimize(objective, state_vec, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < float('inf'):
                new_pts = res.x[:2*N].reshape(N, 2)
                new_cx, new_cy = res.x[2*N], res.x[2*N+1]
                new_S = res.x[2*N+2]
                new_beta = res.x[2*N+3]
                
                # Validate: check if configuration is actually valid (no overlaps, all inside)
                if is_valid_cluster(new_pts, current_angles):
                    # Check containment
                    outer_verts = hex_vertices(new_cx, new_cy, new_S, math.degrees(new_beta))
                    valid = True
                    for i in range(N):
                        inner_verts = hex_vertices(new_pts[i][0], new_pts[i][1], 1.0, current_angles[i])
                        for v in inner_verts:
                            if not point_in_convex_polygon(v[0], v[1], outer_verts):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid and new_S < best_score:
                        best_score = new_S
                        best_state = (new_pts.copy(), current_angles.copy(), new_cx, new_cy, new_S, new_beta)
                        current_pts = new_pts
                        current_cx, current_cy = new_cx, new_cy
                        current_S = new_S
                        current_beta = new_beta
        except Exception:
            pass

        # Periodic perturbation: move one random hexagon
        if it % 10 == 0:
            idx = random.randint(0, N-1)
            current_pts[idx][0] += random.uniform(-0.5, 0.5)
            current_pts[idx][1] += random.uniform(-0.5, 0.5)
            # Reset S to a slightly larger value to allow re-optimization
            current_S = best_score * 1.1
            # Recenter
            current_cx = float(np.mean(current_pts[:, 0]))
            current_cy = float(np.mean(current_pts[:, 1]))

    # Final refinement: try rotating individual hexagons
    for i in range(N):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        for angle_deg in [15, 30, 45]:
            test_angles = current_angles.copy()
            test_angles[i] = angle_deg
            test_pts = current_pts.copy()
            if is_valid_cluster(test_pts, test_angles):
                test_S = find_min_S(test_pts, test_angles, current_cx, current_cy, current_beta)
                if test_S < best_score:
                    # Verify containment
                    outer_verts = hex_vertices(current_cx, current_cy, test_S, math.degrees(current_beta))
                    valid = True
                    for j in range(N):
                        inner_verts = hex_vertices(test_pts[j][0], test_pts[j][1], 1.0, test_angles[j])
                        for v in inner_verts:
                            if not point_in_convex_polygon(v[0], v[1], outer_verts):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_score = test_S
                        best_state = (test_pts.copy(), test_angles.copy(), current_cx, current_cy, test_S, current_beta)

    # Extract best state
    if best_state is None:
        # Fallback
        pts = axial_to_cartesian(seed_axial)
        angles = [0.0] * N
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        best_S = find_min_S(pts, angles, cx, cy, 0.0)
        inner = [[float(x), float(y), float(a)] for (x, y), a in zip(pts, angles)]
        return inner, [cx, cy], best_S, 0.0

    best_pts, best_angles, best_cx, best_cy, best_S, best_beta = best_state
    
    # Final containment check and adjustment
    outer_verts = hex_vertices(best_cx, best_cy, best_S, math.degrees(best_beta))
    for i in range(N):
        inner_verts = hex_vertices(best_pts[i][0], best_pts[i][1], 1.0, best_angles[i])
        for v in inner_verts:
            if not point_in_convex_polygon(v[0], v[1], outer_verts):
                # Expand S slightly if needed
                best_S *= 1.01
                outer_verts = hex_vertices(best_cx, best_cy, best_S, math.degrees(best_beta))
                break

    inner = [[float(x), float(y), float(a)] for (x, y), a in zip(best_pts, best_angles)]
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    
    return inner, center, side, rotation
