import numpy as np
import math
import time
import random

N = 11
DEADLINE = 120.0
SAFETY_MARGIN = 15.0

def hex_vertices(cx, cy, side, angle_deg):
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def project_polygon(vertices, axis):
    dots = vertices @ axis
    return dots.min(), dots.max()

def polygons_overlap_sat(v1, v2):
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            min1, max1 = project_polygon(v1, normal)
            min2, max2 = project_polygon(v2, normal)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    return True

def check_all_disjoint_fast(inner_hexes):
    n = len(inner_hexes)
    verts = [hex_vertices(x, y, 1.0, a) for x, y, a in inner_hexes]
    for i in range(n):
        xi, yi, _ = inner_hexes[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner_hexes[j]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy > 4.0:
                continue
            if polygons_overlap_sat(verts[i], verts[j]):
                return False
    return True

def compute_min_side(inner_hexes, center=(0.0, 0.0), outer_angle=0.0):
    cx, cy = center
    max_radius = 0.0
    for i in range(len(inner_hexes)):
        xi, yi, ai = inner_hexes[i]
        vi = hex_vertices(xi, yi, 1.0, ai)
        for k in range(6):
            a = np.radians(outer_angle) + k * np.pi / 3
            nx, ny = np.cos(a), np.sin(a)
            center_proj = cx * nx + cy * ny
            proj = vi[:, 0] * nx + vi[:, 1] * ny
            max_radius = max(max_radius, abs(proj.max() - center_proj), abs(proj.min() - center_proj))
    return 2.0 * max_radius / np.sqrt(3)

def construct():
    start_time = time.time()
    random.seed(42)

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    d3 = np.sqrt(3)

    configs = [
        [(0.0, 0.0),
         (1.5, 0.0), (0.75, d3 / 2), (-0.75, d3 / 2),
         (-1.5, 0.0), (-0.75, -d3 / 2), (0.75, -d3 / 2),
         (2.25, d3 / 2), (0.0, d3), (-2.25, d3 / 2), (0.0, -d3)],
        [(0.0, 0.0),
         (1.5, 0.0), (0.75, d3 / 2), (-0.75, d3 / 2),
         (-1.5, 0.0), (-0.75, -d3 / 2), (0.75, -d3 / 2),
         (2.25, -d3 / 2), (0.0, -d3), (-2.25, -d3 / 2), (0.0, d3)],
        [(0.0, 0.0),
         (1.5, 0.0), (0.75, d3 / 2), (-0.75, d3 / 2),
         (-1.5, 0.0), (-0.75, -d3 / 2), (0.75, -d3 / 2),
         (2.25, d3 / 2), (2.25, -d3 / 2), (-2.25, d3 / 2), (-2.25, -d3 / 2)],
        [(0.0, 0.0),
         (1.5, 0.0), (0.75, d3 / 2), (-0.75, d3 / 2),
         (-1.5, 0.0), (-0.75, -d3 / 2), (0.75, -d3 / 2),
         (1.5, d3), (0.0, d3), (-1.5, d3), (0.0, -d3)],
    ]

    for config_idx, base_positions in enumerate(configs):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break

        xs = [p[0] for p in base_positions]
        ys = [p[1] for p in base_positions]
        cx0, cy0 = np.mean(xs), np.mean(ys)
        centered = [(x - cx0, y - cy0) for x, y in base_positions]

        current = [(x, y, 0.0) for x, y in centered]

        if not check_all_disjoint_fast(current):
            continue

        current_side = compute_min_side(current, (0.0, 0.0), 0.0)
        best_in_sa = current[:]
        best_side_in_sa = current_side

        T = 0.3
        cooling = 0.9995
        n_steps = 30000

        for step in range(n_steps):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break

            idx = random.randint(0, N - 1)
            dx = random.gauss(0, 0.25)
            dy = random.gauss(0, 0.25)

            new_state = current[:]
            x, y, a = new_state[idx]
            new_state[idx] = (x + dx, y + dy, a)

            if not check_all_disjoint_fast(new_state):
                T *= cooling
                continue

            new_side = compute_min_side(new_state, (0.0, 0.0), 0.0)
            delta = new_side - current_side
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current = new_state
                current_side = new_side
                if current_side < best_side_in_sa:
                    best_side_in_sa = current_side
                    best_in_sa = current[:]

            T *= cooling

        if best_in_sa is not None:
            for outer_ang in range(0, 180, 15):
                side = compute_min_side(best_in_sa, (0.0, 0.0), outer_ang)
                if side < best_side:
                    best_side = side
                    best_inner = best_in_sa[:]
                    best_center = (0.0, 0.0)
                    best_outer_angle = outer_ang

    if best_inner is None:
        inner = [[k * 1.5, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 20.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
