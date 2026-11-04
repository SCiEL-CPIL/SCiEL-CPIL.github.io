import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    dx = px - cx
    dy = py - cy
    theta = np.radians(angle_deg)
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    normals = np.pi / 6 + np.arange(6) * np.pi / 3
    inradius = side * np.sqrt(3) / 2
    for n_angle in normals:
        proj = lx * np.cos(n_angle) + ly * np.sin(n_angle)
        if proj > inradius + 1e-9:
            return False
    return True

def _overlap_penalty(h1, h2):
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    inr1 = s1 * np.sqrt(3) / 2
    inr2 = s2 * np.sqrt(3) / 2
    if d < inr1 + inr2 - 1e-9:
        verts1 = _hex_vertices(cx1, cy1, s1, a1)
        verts2 = _hex_vertices(cx2, cy2, s2, a2)
        for v in verts1:
            if _point_in_hexagon(v[0], v[1], cx2, cy2, s2, a2):
                return 1.0
        for v in verts2:
            if _point_in_hexagon(v[0], v[1], cx1, cy1, s1, a1):
                return 1.0
        if d < (inr1 + inr2) * 0.95:
            return 0.5
    return 0.0

def _compute_outer(positions):
    best_side = 1e10
    best_cx, best_cy, best_angle = 0, 0, 0
    for outer_angle_deg in range(0, 60, 5):
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        max_proj = 0
        normals = np.radians(outer_angle_deg) + np.pi / 6 + np.arange(6) * np.pi / 3
        for pos in positions:
            px, py, pangle = pos
            verts = _hex_vertices(px, py, 1.0, pangle)
            for v in verts:
                dx = v[0] - cx
                dy = v[1] - cy
                for n_angle in normals:
                    proj = dx * np.cos(n_angle) + dy * np.sin(n_angle)
                    if proj > max_proj:
                        max_proj = proj
        inradius_needed = max_proj
        side_needed = inradius_needed / (np.sqrt(3) / 2)
        if side_needed < best_side:
            best_side = side_needed
            best_cx, best_cy, best_angle = cx, cy, outer_angle_deg
    return best_side, best_cx, best_cy, best_angle

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initial hexagonal cluster layout
    positions = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = i * np.pi / 3
        positions.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle), 0.0])
    for angle in [np.pi / 6, np.pi / 2, 5 * np.pi / 6, 7 * np.pi / 6, 3 * np.pi / 2]:
        positions.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle), 0.0])

    outer_side, ocx, ocy, oangle = _compute_outer(positions)
    best_side = outer_side
    best_positions = [p[:] for p in positions]
    best_center = (ocx, ocy)
    best_angle = oangle

    np.random.seed(42)
    for iteration in range(2000):
        if time.time() > deadline - 10:
            break
        idx = np.random.randint(0, N)
        new_pos = best_positions[idx][:]
        new_pos[0] += np.random.randn() * 0.15
        new_pos[1] += np.random.randn() * 0.15
        new_pos[2] = (np.random.randn() * 15) % 60

        trial = [p[:] for p in best_positions]
        trial[idx] = new_pos

        has_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                if _overlap_penalty(
                    (trial[i][0], trial[i][1], 1.0, trial[i][2]),
                    (trial[j][0], trial[j][1], 1.0, trial[j][2])
                ) > 0:
                    has_overlap = True
                    break
            if has_overlap:
                break

        if has_overlap:
            continue

        new_side, ncx, ncy, nangle = _compute_outer(trial)
        if new_side < best_side:
            best_side = new_side
            best_positions = trial
            best_center = (ncx, ncy)
            best_angle = nangle

    inner = [[p[0], p[1], p[2]] for p in best_positions]
    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
