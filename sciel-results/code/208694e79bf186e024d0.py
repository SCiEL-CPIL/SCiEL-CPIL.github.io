"""Hexagon packing: 12 unit hexagons in minimal enclosing regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, s, rot_deg=0.0):
    rot = math.radians(rot_deg)
    verts = []
    for k in range(6):
        a = rot + k * math.pi / 3.0
        px = cx + s * math.cos(a)
        py = cy + s * math.sin(a)
        verts.append((px, py))
    return verts


def _hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _cluster_points(rot_deg=0.0):
    rot = math.radians(rot_deg)
    c, sn = math.cos(rot), math.sin(rot)
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = k * math.pi / 3.0
        centers.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    for k in range(6):
        a = math.pi/6.0 + k * math.pi / 3.0
        centers.append((math.sqrt(3.0) * math.cos(a), math.sqrt(3.0) * math.sin(a)))
    pts = []
    for (cx, cy) in centers:
        for k in range(6):
            a = rot + k * math.pi / 3.0
            vx = cx * c - cy * sn + math.cos(a)
            vy = cx * sn + cy * c + math.sin(a)
            pts.append((vx, vy))
    return pts


def _min_hex_side(hull_pts, n_angles=360):
    hx = np.array([p[0] for p in hull_pts])
    hy = np.array([p[1] for p in hull_pts])
    best = float('inf')
    best_base = 0.0
    for i in range(n_angles):
        base = i * math.pi / n_angles
        cos_n = np.cos(base + np.arange(6) * math.pi / 3.0)
        sin_n = np.sin(base + np.arange(6) * math.pi / 3.0)
        proj = hx[None, :] * cos_n[:, None] + hy[None, :] * sin_n[:, None]
        ranges = proj.max(axis=1) - proj.min(axis=1)
        s_candidate = ranges.max() / math.sqrt(3.0)
        if s_candidate < best:
            best = s_candidate
            best_base = base
    return best, best_base


def _score_for_rotation(rot_deg):
    pts = _cluster_points(rot_deg)
    hull = _hull(pts)
    s, _ = _min_hex_side(hull)
    return s


def construct():
    t0 = time.time()
    deadline = 140.0

    best_rot = 0.0
    best_s = float('inf')
    best_hex_base = 0.0

    # Coarse search: 0.5 degree steps
    n_coarse = 720
    for i in range(n_coarse):
        if time.time() - t0 > deadline:
            break
        rot = i * 360.0 / n_coarse
        pts = _cluster_points(rot)
        hull = _hull(pts)
        s, base = _min_hex_side(hull)
        if s < best_s:
            best_s = s
            best_rot = rot
            best_hex_base = base

    # Refinement passes
    for pass_i in range(5):
        if time.time() - t0 > deadline - 10:
            break
        window = [1.0, 0.3, 0.1, 0.03, 0.01][pass_i]
        n_fine = 200
        for i in range(n_fine):
            if time.time() - t0 > deadline - 10:
                break
            rot = best_rot - window + 2 * window * i / (n_fine - 1)
            pts = _cluster_points(rot)
            hull = _hull(pts)
            s, base = _min_hex_side(hull)
            if s < best_s:
                best_s = s
                best_rot = rot
                best_hex_base = base

    # Build inner positions
    rot = math.radians(best_rot)
    c, sn = math.cos(rot), math.sin(rot)
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = k * math.pi / 3.0
        centers.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    for k in range(6):
        a = math.pi/6.0 + k * math.pi / 3.0
        centers.append((math.sqrt(3.0) * math.cos(a), math.sqrt(3.0) * math.sin(a)))

    inner = []
    for (cx, cy) in centers:
        rx = cx * c - cy * sn
        ry = cx * sn + cy * c
        inner.append([rx, ry, best_rot])

    outer_side = best_s
    outer_center = (0.0, 0.0)
    outer_rot = best_hex_base * 180.0 / math.pi

    return inner, outer_center, outer_side, outer_rot

# EVOLVE-BLOCK-END
