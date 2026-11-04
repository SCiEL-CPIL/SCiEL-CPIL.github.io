import numpy as np
from scipy.optimize import minimize_scalar

def _hexagon_vertices(cx, cy, angle_deg=0.0):
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + k * np.pi / 3.0
        verts.append((cx + np.cos(theta), cy + np.sin(theta)))
    return verts

def _all_cluster_vertices(inner):
    pts = []
    for x, y, ang in inner:
        verts = _hexagon_vertices(x, y, ang)
        pts.extend(verts)
    return np.array(pts)

def _min_enclosing_hexagon_side(points, phi):
    max_proj = 0.0
    for k in range(6):
        beta = phi + k * np.pi / 3.0
        proj = points[:, 0] * np.cos(beta) + points[:, 1] * np.sin(beta)
        max_abs = float(np.max(np.abs(proj)))
        if max_abs > max_proj:
            max_proj = max_abs
    return max_proj * 2.0 / np.sqrt(3.0)

def _optimal_side_for_arrangement(inner):
    pts = _all_cluster_vertices(inner)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    def objective(phi):
        return _min_enclosing_hexagon_side(pts_centered, phi)
    result = minimize_scalar(objective, bounds=(0.0, np.pi / 3.0), method='bounded',
                             options={'xatol': 1e-12})
    return float(result.fun), float(result.x), centroid

def construct():
    N = 11
    e1 = np.array([np.sqrt(3.0), 0.0])
    e2 = np.array([np.sqrt(3.0) / 2.0, 3.0 / 2.0])

    base7 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    # Several candidate sets of 4 additional hexagons to try
    candidate_extras = [
        [(1, 1), (-1, -1), (1, -2), (-1, 2)],
        [(1, 1), (-1, -1), (2, -1), (-2, 1)],
        [(2, 0), (-2, 0), (0, 2), (0, -2)],
        [(1, 1), (1, -2), (-1, 2), (-1, -1)],
        [(2, 0), (0, 2), (-2, 0), (0, -2)],
        [(2, -1), (-2, 1), (1, 1), (-1, -1)],
        [(2, -1), (-2, 1), (2, 0), (-2, 0)],
        [(0, 2), (0, -2), (1, 1), (-1, -1)],
    ]

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_rot = 0.0

    for extra in candidate_extras:
        lattice_pts = base7 + extra
        centers = []
        for i, j in lattice_pts:
            c = i * e1 + j * e2
            centers.append((float(c[0]), float(c[1]), 0.0))
        side, phi, centroid = _optimal_side_for_arrangement(centers)
        if side < best_side:
            best_side = side
            best_inner = centers
            best_center = (float(centroid[0]), float(centroid[1]))
            best_rot = float(np.degrees(phi))

    return best_inner, best_center, float(best_side), best_rot
