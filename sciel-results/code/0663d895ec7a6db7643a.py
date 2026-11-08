import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_polygons_fast(v1, v2):
        min_d = float('inf')
        # Check edges of v1 against v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        # Check edges of v2 against v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        return min_d

    def evaluate_violation(params, outer_s):
        # params: 36D vector [x1, y1, a1, x2, y2, a2, ...]
        inner_verts = []
        for i in range(12):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            v = hex_verts(x, y, 1.0, a)
            inner_verts.append(v)
        
        violation = 0.0
        r = outer_s * np.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        # Containment penalty
        for i in range(12):
            for v in inner_verts[i]:
                dists = np.dot(v, n_vecs.T)
                out = np.maximum(0, dists - r)
                violation += np.sum(out)
        
        # Non-overlap penalty
        for i in range(12):
            for j in range(i+1, 12):
                cx1, cy1 = params[i*3], params[i*3+1]
                cx2, cy2 = params[j*3], params[j*3+1]
                c_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                # Minimum distance between centers of two unit hexagons is sqrt(3) if they touch at vertices.
                # Actually, if they are parallel, min dist is sqrt(3). If rotated, it can be smaller.
                # A safe lower bound for center distance to guarantee non-overlap is 2*apothem = sqrt(3).
                # But if rotated, they can interlock? No, convex polygons.
                # The minimum distance between centers of two unit regular hexagons is sqrt(3).
                # If c_dist < sqrt(3), they must overlap.
                threshold = np.sqrt(3)
                if c_dist < threshold:
                    violation += (threshold - c_dist) ** 2 * 10.0
        return violation

    def objective(params):
        return evaluate_violation(params, current_outer_s)

    def get_initial_positions():
        # Hexagonal lattice initialization
        # Side length 1 hexagons.
        # Flat top hexagon: width = 2, height = sqrt(3).
        # Pointy top hexagon: width = sqrt(3), height = 2.
        # Let's use flat top (rotation 0).
        # Center to center distance for adjacent flat top hexagons:
        # Horizontal: 2 * apothem = sqrt(3) * 2? No.
        # Flat top hex vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), ...
        # Apothem is sqrt(3)/2.
        # To tile, horizontal spacing is 2 * apothem = sqrt(3).
        # Vertical spacing is 3/2 (height of 3 rows of equilateral triangles? No).
        # Let's just use a grid that is definitely valid for a large outer hexagon.
        # Spacing 2.0 in x and 1.8 in y is safe.
        pos = []
        for i in range(12):
            x = (i % 4) * 2.0 - 3.0
            y = (i // 4) * 1.8 - 1.5
            a = 0.0
            pos.append([x, y, a])
        return np.array(pos).flatten()

    LB = 3.0
    UB = 10.0
    best_solution = None
    best_s = float('inf')
    
    # Bounds for Differential Evolution
    # x, y in [-5, 5]
    # angle in [0, 60] (symmetry)
    bounds = []
    for _ in range(12):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 60.0))

    x0 = get_initial_positions()

    for step in range(20):
        if time.time() > deadline:
            break
        mid_s = (LB + UB) / 2.0
        current_outer_s = mid_s
        
        found_valid = False
        best_violation = float('inf')
        best_params = x0.copy()

        # Use Differential Evolution
        try:
            res = differential_evolution(objective, bounds, maxiter=20, popsize=10, tol=1e-4, seed=42)
            if res.fun < best_violation:
                best_violation = res.fun
                best_params = res.x
            if res.fun < 1e-3:
                found_valid = True
        except Exception:
            pass

        if found_valid:
            UB = mid_s
            best_solution = best_params.copy()
            best_s = mid_s
        else:
            LB = mid_s

    if best_solution is None:
        # Fallback to initial grid
        inner = []
        for i in range(12):
            x = (i % 4) * 2.0 - 3.0
            y = (i // 4) * 1.8 - 1.5
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    inner = []
    for i in range(12):
        x = best_solution[i*3]
        y = best_solution[i*3+1]
        a = best_solution[i*3+2]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), best_s, 0.0
