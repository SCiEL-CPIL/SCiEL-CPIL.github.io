import numpy as np
from scipy.spatial import ConvexHull
import time

N = 11

def _hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    rad = np.radians(angle_deg)
    angles = rad + np.radians(np.arange(0, 360, 60))
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _cluster_centers():
    d = np.sqrt(3)
    centers = [(0.0, 0.0)]
    for k in range(6):
        ang = np.radians(30 + 60 * k)
        centers.append((d * np.cos(ang), d * np.sin(ang)))
    for k in range(4):
        ang = np.radians(90 * k)
        centers.append((2 * d * np.cos(ang), 2 * d * np.sin(ang)))
    return np.array(centers)

def _min_side_length(hull_points, theta):
    ct, st = np.cos(theta), np.sin(theta)
    rotated = hull_points @ np.array([[ct, -st], [st, ct]]).T
    normal_angles = np.radians(30) + np.radians(60) * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    projections = rotated @ normals.T
    max_proj = np.max(np.abs(projections))
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    centers = _cluster_centers()
    all_vertices = np.vstack([_hex_vertices(cx, cy) for cx, cy in centers])
    hull = ConvexHull(all_vertices)
    hull_points = all_vertices[hull.vertices]

    a, b = 0.0, np.pi / 3.0
    golden = (np.sqrt(5) - 1) / 2

    best_theta = 0.0
    best_side = float('inf')

    for theta in np.linspace(0, np.pi / 3, 21):
        side = _min_side_length(hull_points, theta)
        if side < best_side:
            best_side = side
            best_theta = theta

    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc = _min_side_length(hull_points, c)
    fd = _min_side_length(hull_points, d)
    if fc < best_side:
        best_side, best_theta = fc, c
    if fd < best_side:
        best_side, best_theta = fd, d

    for _ in range(200):
        if time.time() > deadline:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = _min_side_length(hull_points, c)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = _min_side_length(hull_points, d)
        if fc < best_side:
            best_side, best_theta = fc, c
        if fd < best_side:
            best_side, best_theta = fd, d
        if abs(b - a) < 1e-12:
            break

    angle_deg = float(np.degrees(best_theta))
    inner = [[float(cx), float(cy), angle_deg] for cx, cy in centers]
    return inner, (0.0, 0.0), float(best_side), angle_deg
