"""Rewritten program for hexagon packing with a stronger structural approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    apothem = np.sqrt(3) / 2.0

    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S(vertices_list, ocx=0.0, ocy=0.0, oangle=0.0):
        """Minimum side length S of outer hexagon (center ocx,ocy, rotation oangle) to contain all vertices."""
        S = 0.0
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        for verts in vertices_list:
            for v in verts:
                lx = v[0] - ocx
                ly = v[1] - ocy
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_overlap_depth(v1, v2):
        """Return overlap depth (0 if no overlap) between two convex polygons using SAT."""
        polys = [v1, v2]
        max_depth = 0.0
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-12:
                    continue
                normal = np.array([-edge[1], edge[0]]) / edge_len
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                if overlap < 0:
                    return 0.0  # separating axis found
                max_depth = max(max_depth, overlap)
        return max_depth

    def objective(params, use_penalty=True):
        """Compute objective: outer side length + overlap penalty."""
        n = len(params) // 3
        vertices_list = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)

        S = compute_S(vertices_list)

        if not use_penalty:
            return S

        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:
                    depth = sat_overlap_depth(vertices_list[i], vertices_list[j])
                    if depth > 0:
                        penalty += 500.0 * depth
        return S + penalty

    def make_initial_lattice():
        """Create initial hexagonal lattice placement."""
        params = np.zeros(3 * N)
        # Center hexagon
        params[0:3] = [0, 0, 0]
        idx = 1
        # Ring 1: 6 hexagons at distance sqrt(3)
        for k in range(6):
            theta = k * np.pi / 3
            cx = np.sqrt(3) * np.cos(theta)
            cy = np.sqrt(3) * np.sin(theta)
            params[3 * idx] = cx
            params[3 * idx + 1] = cy
            params[3 * idx + 2] = 0
            idx += 1
        # Ring 2: 4 more hexagons (we need 11 total, have 7, need 4 more)
        # Place at distance 2*sqrt(3) in directions that avoid the ring-1 gaps
        for k in range(4):
            theta = (k + 0.5) * np.pi / 3
            cx = 2 * np.sqrt(3) * np.cos(theta)
            cy = 2 * np.sqrt(3) * np.sin(theta)
            params[3 * idx] = cx
            params[3 * idx + 1] = cy
            params[3 * idx + 2] = 0
            idx += 1
        return params

    def optimize_from(params, maxiter=3000, maxfev=15000):
        """Run Powell optimization from given initial params."""
        if time.time() > deadline:
            return None, np.inf
        try:
            result = minimize(objective, params, method='Powell',
                              options={'maxiter': maxiter, 'maxfev': maxfev, 'xatol': 1e-8, 'fatol': 1e-8})
            return result.x, result.fun
        except Exception:
            return None, np.inf

    # Multi-start strategy with diverse initializations
    best_params = None
    best_score = np.inf

    # Start 1: standard lattice
    p0 = make_initial_lattice()
    res_params, res_score = optimize_from(p0, maxiter=4000, maxfev=20000)
    if res_params is not None and res_score < best_score:
        best_params = res_params
        best_score = res_score

    # Start 2: slightly rotated lattice
    if time.time() < deadline - 20:
        p1 = p0.copy()
        p1[2] = np.pi / 12  # rotate center hexagon
        for i in range(1, N):
            p1[3 * i + 2] = np.pi / 6  # rotate others
        res_params, res_score = optimize_from(p1, maxiter=3000, maxfev=15000)
        if res_params is not None and res_score < best_score:
            best_params = res_params
            best_score = res_score

    # Start 3: random perturbation of best
    if best_params is not None and time.time() < deadline - 15:
        p2 = best_params + np.random.normal(0, 0.05, size=best_params.size)
        res_params, res_score = optimize_from(p2, maxiter=3000, maxfev=15000)
        if res_params is not None and res_score < best_score:
            best_params = res_params
            best_score = res_score

    # Start 4: different lattice orientation
    if time.time() < deadline - 15:
        p3 = make_initial_lattice()
        # Rotate all by 30 degrees
        for i in range(N):
            p3[3 * i + 2] = np.pi / 6
        # Also shift positions slightly
        p3[1:3:N*3] += 0.1  # small y shift
        res_params, res_score = optimize_from(p3, maxiter=3000, maxfev=15000)
        if res_params is not None and res_score < best_score:
            best_params = res_params
            best_score = res_score

    # Start 5: tighter initial packing
    if time.time() < deadline - 15:
        p4 = np.zeros(3 * N)
        # More compact arrangement: use axial coordinates
        # Hex grid with axial coords (q, r) -> Cartesian
        def axial_to_cartesian(q, r):
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            return x, y

        # 11 hexagons in a compact cluster
        coords = [(0, 0),
                  (1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, -1), (-1, 1), (1, 1), (-1, -1),
                  (2, -1), (-2, 1)]
        for i, (q, r) in enumerate(coords):
            x, y = axial_to_cartesian(q, r)
            p4[3 * i] = x
            p4[3 * i + 1] = y
            p4[3 * i + 2] = 0
        res_params, res_score = optimize_from(p4, maxiter=3000, maxfev=15000)
        if res_params is not None and res_score < best_score:
            best_params = res_params
            best_score = res_score

    # Final refinement: local search on best
    if best_params is not None and time.time() < deadline - 10:
        # Try small perturbations in each direction
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            p = best_params + np.random.normal(0, 0.02, size=best_params.size)
            res_params, res_score = optimize_from(p, maxiter=2000, maxfev=10000)
            if res_params is not None and res_score < best_score:
                best_params = res_params
                best_score = res_score

    if best_params is None:
        best_params = make_initial_lattice()

    # Compute final result
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = np.degrees(best_params[3 * i + 2])
        inner.append([cx, cy, angle])

    vertices_list = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)

    S = compute_S(vertices_list)

    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
