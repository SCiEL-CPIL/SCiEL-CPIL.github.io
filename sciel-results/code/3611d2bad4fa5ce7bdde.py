# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(hex_verts):
    normals = []
    centroid = hex_verts.mean(axis=0)
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _separated(v1, n1, v2, n2):
    all_normals = np.vstack([n1, n2])
    for k in range(12):
        n = all_normals[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if (max1 < min2 - 1e-6) or (max2 < min1 - 1e-6):
            return True
    return False


def _all_separated(inner_hexes):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(inner_hexes[i][0], inner_hexes[i][1],
                              inner_hexes[j][0], inner_hexes[j][1]):
                return False
    return True


def _min_outer_side(inner_hexes):
    def _fits(S, angle_deg):
        outer_verts = _hex_vertices(0, 0, S, angle_deg)
        for hex_verts, normals in inner_hexes:
            for p in hex_verts:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    max_r = 0
    for hex_verts, normals in inner_hexes:
        r = np.max(np.linalg.norm(hex_verts, axis=1))
        max_r = max(max_r, r)

    S_low = max_r
    S_high = S_low * 2.0

    for _ in range(50):
        S_mid = (S_low + S_high) / 2.0
        if _fits(S_mid, 0.0):
            S_high = S_mid
        else:
            S_low = S_mid

    best_S = S_high
    best_angle = 0.0

    for angle_deg in [0, 10, 20, 30, 40, 50, 60]:
        sl = max_r * 0.95
        sh = max_r * 2.0
        for _ in range(50):
            sm = (sl + sh) / 2.0
            if _fits(sm, angle_deg):
                sh = sm
            else:
                sl = sm
        if sh < best_S:
            best_S = sh
            best_angle = angle_deg

    return best_S, best_angle


def _build_inner_hexes(positions, angles):
    hexes = []
    for i in range(11):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        norms = _edge_normals(verts)
        hexes.append((verts, norms))
    return hexes


def _hex_lattice_positions(rings):
    positions = [(0.0, 0.0)]
    for ring in range(1, rings + 1):
        dist = ring * 2.5
        for i in range(6 * ring):
            angle = np.deg2rad(60.0 * i / ring)
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            positions.append((x, y))
    return positions


def _generate_symmetric_arrangements():
    arrangements = []
    
    ring1_dist = 2.5
    ring2_dist = 5.0

    ring1 = []
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        x = ring1_dist * np.cos(angle)
        y = ring1_dist * np.sin(angle)
        ring1.append((x, y))

    for offset in [0, 15, 30, 45, 60]:
        ring2_sel = []
        for i in [0, 3, 6, 9]:
            angle = np.deg2rad(30.0 * i + offset)
            x = ring2_dist * np.cos(angle)
            y = ring2_dist * np.sin(angle)
            ring2_sel.append((x, y))
        
        positions = [(0.0, 0.0)] + ring1 + ring2_sel
        angles = [0.0] * 11
        arrangements.append((np.array(positions), np.array(angles)))

    for r1_scale in [1.0, 0.95, 0.9, 1.05, 1.1]:
        for r2_scale in [1.0, 0.95, 0.9, 1.05, 1.1]:
            r1d = ring1_dist * r1_scale
            r2d = ring2_dist * r2_scale
            
            ring1_s = []
            for i in range(6):
                angle = np.deg2rad(60.0 * i)
                x = r1d * np.cos(angle)
                y = r1d * np.sin(angle)
                ring1_s.append((x, y))
            
            ring2_s = []
            for i in [0, 3, 6, 9]:
                angle = np.deg2rad(30.0 * i)
                x = r2d * np.cos(angle)
                y = r2d * np.sin(angle)
                ring2_s.append((x, y))
            
            positions = [(0.0, 0.0)] + ring1_s + ring2_s
            angles = [0.0] * 11
            arrangements.append((np.array(positions), np.array(angles)))

    for r1_scale in [1.0, 0.95, 1.05]:
        for r2_scale in [1.0, 0.95, 1.05]:
            r1d = ring1_dist * r1_scale
            r2d = ring2_dist * r2_scale
            
            ring1_s = []
            for i in range(5):
                angle = np.deg2rad(60.0 * i + 30)
                x = r1d * np.cos(angle)
                y = r1d * np.sin(angle)
                ring1_s.append((x, y))
            
            ring2_s = []
            for i in range(5):
                angle = np.deg2rad(60.0 * i)
                x = r2d * np.cos(angle)
                y = r2d * np.sin(angle)
                ring2_s.append((x, y))
            
            positions = [(0.0, 0.0)] + ring1_s + ring2_s
            angles = [0.0] * 11
            arrangements.append((np.array(positions), np.array(angles)))

    for dx in [0, 0.5, 1.0, 1.5, 2.0]:
        for dy in [0, 0.5, 1.0, 1.5, 2.0]:
            if dx == 0 and dy == 0:
                continue
            c1 = (dx, dy)
            c2 = (-dx, -dy)
            
            outer_dist = 4.5
            outer = []
            for i in range(9):
                angle = np.deg2rad(40.0 * i + 20)
                x = outer_dist * np.cos(angle)
                y = outer_dist * np.sin(angle)
                outer.append((x, y))
            
            positions = [c1, c2] + outer
            angles = [0.0] * 11
            arrangements.append((np.array(positions), np.array(angles)))

    seed_positions = np.array(
        [
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.165],
            [1.25, 2.165],
            [-1.25, -2.165],
            [1.25, -2.165],
            [-3.75, 2.165],
            [3.75, 2.165],
            [-3.75, -2.165],
            [3.75, -2.165],
        ]
    )
    arrangements.append((seed_positions, np.zeros(11)))

    for rot in [0, 15, 30, 45, 60]:
        ring1_s = []
        for i in range(6):
            angle = np.deg2rad(60.0 * i)
            x = ring1_dist * np.cos(angle)
            y = ring1_dist * np.sin(angle)
            ring1_s.append((x, y))
        
        ring2_s = []
        for i in [0, 3, 6, 9]:
            angle = np.deg2rad(30.0 * i)
            x = ring2_dist * np.cos(angle)
            y = ring2_dist * np.sin(angle)
            ring2_s.append((x, y))
        
        positions = [(0.0, 0.0)] + ring1_s + ring2_s
        angles = [0.0] + [rot] * 6 + [rot] * 4
        arrangements.append((np.array(positions), np.array(angles)))

    return arrangements


def hexagon_packing_11():
    t_start = time.time()
    deadline = 600.0

    arrangements = _generate_symmetric_arrangements()
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_angle = 0.0

    for positions, angles in arrangements:
        if time.time() > deadline - 60:
            break
        
        inner_hexes = _build_inner_hexes(positions, angles)
        if not _all_separated(inner_hexes):
            continue
        
        S, angle = _min_outer_side(inner_hexes)
        if np.isfinite(S) and S < best_S:
            best_S = S
            best_positions = positions.copy()
            best_angles = angles.copy()
            best_angle = angle

    if best_positions is None:
        seed_positions = np.array(
            [
                [0, 0],
                [-2.5, 0],
                [2.5, 0],
                [-1.25, 2.165],
                [1.25, 2.165],
                [-1.25, -2.165],
                [1.25, -2.165],
                [-3.75, 2.165],
                [3.75, 2.165],
                [-3.75, -2.165],
                [3.75, -2.165],
            ]
        )
        best_positions = seed_positions
        best_angles = np.zeros(11)
        inner_hexes = _build_inner_hexes(best_positions, best_angles)
        best_S, best_angle = _min_outer_side(inner_hexes)

    # Use differential evolution for global search
    if time.time() < deadline - 180:
        def _objective(params):
            if time.time() > deadline - 30:
                return 1e10
            
            positions = params[:22].reshape(11, 2)
            angles = params[22:]
            
            inner_hexes = _build_inner_hexes(positions, angles)
            if not _all_separated(inner_hexes):
                return 1e10
            
            S, angle = _min_outer_side(inner_hexes)
            if not np.isfinite(S):
                return 1e10
            return S
        
        # Define bounds for differential evolution
        bounds = [(-5, 5)] * 22 + [(0, 60)] * 11
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                polish=False
            )
            if result.fun < best_S:
                best_S = result.fun
                best_positions = result.x[:22].reshape(11, 2)
                best_angles = result.x[22:]
                inner_hexes = _build_inner_hexes(best_positions, best_angles)
                best_S, best_angle = _min_outer_side(inner_hexes)
        except Exception:
            pass

    # Local optimization from best arrangement
    if time.time() < deadline - 120:
        x0 = np.zeros(33)
        x0[:22] = best_positions.flatten()
        x0[22:] = best_angles
        
        def _objective(params):
            if time.time() > deadline - 30:
                return 1e10
            
            positions = params[:22].reshape(11, 2)
            angles = params[22:]
            
            inner_hexes = _build_inner_hexes(positions, angles)
            if not _all_separated(inner_hexes):
                return 1e10
            
            S, angle = _min_outer_side(inner_hexes)
            if not np.isfinite(S):
                return 1e10
            return S
        
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_S:
                best_S = res.fun
                best_positions = res.x[:22].reshape(11, 2)
                best_angles = res.x[22:]
                inner_hexes = _build_inner_hexes(best_positions, best_angles)
                best_S, best_angle = _min_outer_side(inner_hexes)
        except Exception:
            pass

    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
