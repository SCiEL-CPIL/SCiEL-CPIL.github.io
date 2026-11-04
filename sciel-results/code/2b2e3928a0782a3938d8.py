import numpy as np
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    theta = np.radians(angle_deg)
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(p, cx, cy, side, angle_deg):
    theta = np.radians(angle_deg)
    a = side * np.sqrt(3) / 2.0
    for k in range(6):
        na = theta + np.pi / 6 + k * np.pi / 3
        nx, ny = np.cos(na), np.sin(na)
        if (p[0] - cx) * nx + (p[1] - cy) * ny > a + 1e-9:
            return False
    return True

def hexagons_overlap(h1, h2):
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    verts1 = hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = hex_vertices(cx2, cy2, 1.0, a2)
    for v in verts1:
        if point_in_hexagon(v, cx2, cy2, 1.0, a2):
            return True
    for v in verts2:
        if point_in_hexagon(v, cx1, cy1, 1.0, a1):
            return True
    return False

def min_enclosing_hexagon_side(points, angle_deg):
    theta = np.radians(angle_deg)
    normals = np.array([[np.cos(theta + np.pi/6 + k*np.pi/3), 
                         np.sin(theta + np.pi/6 + k*np.pi/3)] for k in range(6)])
    dots = points @ normals.T
    max_dots = dots.max(axis=0)
    min_dots = dots.min(axis=0)
    a_min = np.max((max_dots - min_dots) / 2.0)
    return a_min * 2.0 / np.sqrt(3.0)

def find_center_for_hexagon(points, angle_deg):
    theta = np.radians(angle_deg)
    normals = np.array([[np.cos(theta + np.pi/6 + k*np.pi/3), 
                         np.sin(theta + np.pi/6 + k*np.pi/3)] for k in range(6)])
    dots = points @ normals.T
    max_dots = dots.max(axis=0)
    min_dots = dots.min(axis=0)
    targets = (max_dots + min_dots) / 2.0
    c, _, _, _ = np.linalg.lstsq(normals.T, targets, rcond=None)
    return c

def build_cluster(topology_type=0):
    d = np.sqrt(3)
    positions = []
    if topology_type == 0:
        for i in range(4): positions.append((d*i, 0.0))
        for i in range(3): positions.append((d*(i+0.5), 1.5))
        for i in range(3): positions.append((d*(i+1.0), 3.0))
        for i in range(2): positions.append((d*(i+1.5), 4.5))
    elif topology_type == 1:
        for row in range(3):
            n = 5 - row
            y = row * 1.5
            xo = row * d / 2
            for i in range(n): positions.append((xo + d*i, y))
    elif topology_type == 2:
        for row in range(3):
            y = row * 1.5
            xo = (row % 2) * d / 2
            for i in range(4): positions.append((xo + d*i, y))
    elif topology_type == 3:
        for row in range(2):
            y = row * 1.5
            xo = (row % 2) * d / 2
            for i in range(6): positions.append((xo + d*i, y))
    elif topology_type == 4:
        for row in range(4):
            y = row * 1.5
            xo = (row % 2) * d / 2
            for i in range(3): positions.append((xo + d*i, y))
    elif topology_type == 5:
        positions.extend([(d*1.5,0.0),(d*0.5,0.0)])
        for i in range(4): positions.append((d*i, 1.5))
        for i in range(4): positions.append((d*i, 3.0))
        positions.extend([(d*1.5,4.5),(d*0.5,4.5)])
    return [[x, y, 0.0] for x, y in positions]

def check_validity(hexagons, cx, cy, side, angle_deg):
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if hexagons_overlap(hexagons[i], hexagons[j]):
                return False
    for h in hexagons:
        verts = hex_vertices(h[0], h[1], 1.0, h[2])
        for v in verts:
            if not point_in_hexagon(v, cx, cy, side, angle_deg):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    best_result = None
    best_side = np.inf
    
    for topo in range(6):
        if time.time() > deadline:
            break
        hexagons = build_cluster(topo)
        if len(hexagons) != N:
            continue
        all_verts = np.vstack([hex_vertices(h[0], h[1], 1.0, h[2]) for h in hexagons])
        best_angle = 0.0
        best_s = np.inf
        for angle_deg in np.linspace(0, 60, 121):
            s = min_enclosing_hexagon_side(all_verts, angle_deg)
            if s < best_s:
                best_s = s
                best_angle = angle_deg
        c = find_center_for_hexagon(all_verts, best_angle)
        shifted = [[h[0]-c[0], h[1]-c[1], h[2]] for h in hexagons]
        if check_validity(shifted, 0.0, 0.0, best_s, best_angle):
            if best_s < best_side:
                best_side = best_s
                best_result = (shifted, (0.0, 0.0), best_s, best_angle)
        else:
            for extra in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
                s2 = best_s + extra
                if check_validity(shifted, 0.0, 0.0, s2, best_angle):
                    if s2 < best_side:
                        best_side = s2
                        best_result = (shifted, (0.0, 0.0), s2, best_angle)
                    break
    
    if best_result is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return best_result
