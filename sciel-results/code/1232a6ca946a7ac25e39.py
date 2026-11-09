# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 620.0  # safety margin before 630s kill


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(th), cy + d * np.sin(th)])


def _inner_positions():
    """Honeycomb 3-2-3-2-2 arrangement of 12 unit hexagons, centered on centroid."""
    dy = 1.5 * np.sqrt(3.0)
    pos = np.array([
        # row 3
        [-1.5, dy], [0.0, dy], [1.5, dy],
        # row 2
        [-0.75, 0.0], [0.75, 0.0],
        # row 3
        [-1.5, -dy], [0.0, -dy], [1.5, -dy],
        # row 2
        [-0.75, -2 * dy], [0.75, -2 * dy],
        # row 2
        [0.0, -3 * dy], [1.5, -3 * dy],
    ])
    pos = pos - pos.mean(axis=0)  # center cluster at origin
    return pos


def _cluster_vertices(pos, rot_deg):
    """All 72 vertices of the 12 inner hexagons after a global rotation (deg)."""
    r = np.deg2rad(rot_deg)
    c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s], [s, c]])
    verts = []
    for (cx, cy) in pos:
        base = _hex_vertices(0.0, 0.0, 1.0, 0.0)  # axis-aligned unit hexagon
        v = base + np.array([cx, cy])
        v = v @ R.T
        verts.append(v)
    return np.vstack(verts)


def _min_side_for_orientation(verts, theta):
    """Required outer side length for a given outer orientation theta (deg)."""
    t = np.deg2rad(theta)
    # 6 outward normals of the outer hexagon
    normals = np.column_stack([np.cos(t + np.deg2rad(60.0 * np.arange(6))),
                               np.sin(t + np.deg2rad(60.0 * np.arange(6)))])
    # support function: max projection of cluster vertices onto each normal
    proj = verts @ normals.T  # (nverts, 6)
    R = proj.max(axis=0)  # (6,)
    # side length s satisfies s*cos(60k) = R_k  => s = R_k / cos(60k)
    cosk = np.cos(np.deg2rad(60.0 * np.arange(6)))
    return (R / cosk).max()


def _objective(params, verts_by_rot=None, pos=None):
    """Objective: required outer side length. params = [theta_deg, rot_deg]."""
    theta, rot = params
    if abs(rot) > 1e-12:
        verts = _cluster_vertices(pos, rot)
    else:
        verts = verts_by_rot
    return _min_side_for_orientation(verts, theta)


def hexagon_packing_12():
    t0 = time.time()

    pos = _inner_positions()
    verts0 = _cluster_vertices(pos, 0.0)

    # Best-known fallback: axis-aligned outer hexagon, side from support function.
    s_fallback = _min_side_for_orientation(verts0, 0.0)
    best_theta, best_rot, best_s = 0.0, 0.0, s_fallback

    # Nelder-Mead over (outer orientation, global cluster rotation).
    def obj(p):
        return _objective(p, verts_by_rot=verts0, pos=pos)

    try:
        x0 = np.array([0.0, 0.0])
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12,
                                'adaptive': True})
        if res.x is not None and np.all(np.isfinite(res.x)):
            theta, rot = float(res.x[0]), float(res.x[1])
            s = _min_side_for_orientation(_cluster_vertices(pos, rot), theta)
            if np.isfinite(s) and s < best_s:
                best_theta, best_rot, best_s = theta, rot, s
    except Exception:
        pass

    # Coarse scan over theta (and a few cluster rotations) as a robust backup.
    for rot in (0.0,):
        verts = _cluster_vertices(pos, rot)
        for theta in np.arange(0.0, 60.0, 1.0):
            s = _min_side_for_orientation(verts, theta)
            if np.isfinite(s) and s < best_s:
                best_theta, best_rot, best_s = float(theta), rot, s

    if time.time() - t0 > _DEADLINE:
        # bail to the safe fallback
        best_theta, best_rot, best_s = 0.0, 0.0, s_fallback

    # Build final inner hexagon data (apply global rotation to centers).
    r = np.deg2rad(best_rot)
    c, s_ = np.cos(r), np.sin(r)
    R = np.array([[c, -s_], [s_, c]])
    centers = pos @ R.T
    inner_hex_data = np.column_stack([centers, np.zeros(len(centers))])  # (12,3) angle 0

    outer_hex_data = np.array([0.0, 0.0, best_theta])
    return inner_hex_data, outer_hex_data, float(best_s)


# EVOLVE-BLOCK-END
