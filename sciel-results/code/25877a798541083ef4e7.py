import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
d = math.sqrt(3.0)
TIME_LIMIT = 150.0
SAFETY = 8.0

# Unit regular hexagon vertices (side 1, centered at origin), angles 0,60,...,300
_VERTS = np.array([[math.cos(math.pi * i / 3.0), math.sin(math.pi * i / 3.0)] for i in range(6)])


def _rot(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return pts @ np.array([[c, -s], [s, c]])


def _build_cluster(theta):
    """Build the 11 unit hexagons as a rigid cluster (non-overlapping by construction).
    Returns (66,2) array of all vertices and (11,2) centers."""
    core = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi / 3.0
        core.append((2.0 * d * math.cos(a), 2.0 * d * math.sin(a)))
    base = [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0, math.pi]
    outer = []
    for k in range(4):
        a = base[k] + theta
        outer.append((3.0 * d * math.cos(a), 3.0 * d * math.sin(a)))
    centers = np.array(core + outer)

    verts = np.empty((N, 6, 2))
    for i, (cx, cy) in enumerate(centers):
        verts[i] = _rot(_VERTS, theta) + np.array([cx, cy])
    return verts.reshape(-1, 2), centers


def _side_from_verts(verts, alpha):
    """Enclosing regular-hexagon side length whose side normals are at alpha + k*pi/3."""
    ca = np.cos(alpha + np.arange(6) * math.pi / 3.0)
    sa = np.sin(alpha + np.arange(6) * math.pi / 3.0)
    dots = verts @ np.stack([ca, sa], axis=1)      # (66,6)
    extents = dots.max(axis=0) - dots.min(axis=0)
    return extents.max() / d


def _objective(p):
    x, y, theta, alpha = p
    verts, _ = _build_cluster(theta)
    verts = verts + np.array([x, y])
    return _side_from_verts(verts, alpha)


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT - SAFETY

    bounds = [(-4.0, 4.0), (-4.0, 4.0), (0.0, math.pi / 3.0), (0.0, math.pi / 3.0)]

    # Coarse global search over the rigid-cluster parameterization + container rotation
    best = None
    try:
        res = differential_evolution(
            _objective, bounds, seed=12345, maxiter=400, tol=1e-10,
            popsize=18, mutation=(0.4, 1.2), recombination=0.9, polish=True,
            init='sobol', updating='immediate', workers=-1,
            options={'init_opts': {'n_skip': 256}})
        if res.success or res.fun < np.inf:
            best = res.x
    except Exception:
        pass

    if best is None:
        # Fallback: symmetric flower with 4 outer hexagons at 45,135,225,315 deg
        best = np.array([0.0, 0.0, math.pi / 4.0, 0.0])

    x, y, theta, alpha = best
    verts, centers = _build_cluster(theta)
    verts = verts + np.array([x, y])

    # Final tight refinement of the container rotation for the fixed packing
    try:
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(lambda a: _side_from_verts(verts, a),
                            bounds=(alpha - math.pi / 18.0, alpha + math.pi / 18.0),
                            method='bounded', options={'xatol': 1e-12})
        alpha = r.x
    except Exception:
        pass

    S = _side_from_verts(verts, alpha)

    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0] + x), float(centers[i, 1] + y),
                      float(math.degrees(theta))])

    return inner, (float(x), float(y)), float(S), float(math.degrees(alpha))
