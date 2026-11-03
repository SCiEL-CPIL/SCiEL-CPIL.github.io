import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

# 6 unit-hexagon vertex directions (pointy-top, radius = side = 1)
_HEX_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in
    [30, 90, 150, 210, 270, 330]
])

# 6 outward normal directions of the container (pointy-top, angle = 90 + 60k)
_NORMALS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in
    [90, 150, 210, 270, 330, 30]
])


def hexagon_vertices(cx, cy, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return np.column_stack([
        cx + _HEX_VERTS[:, 0] * c - _HEX_VERTS[:, 1] * s,
        cy + _HEX_VERTS[:, 0] * s + _HEX_VERTS[:, 1] * c,
    ])


def min_side(points):
    """Smallest side length of a pointy-top hexagon (centered at origin,
    rotatable) that contains all points."""
    # project all points onto the 6 normal directions
    # D = max over directions of |projection| (centered)
    # We optimize center too via a quick analytic: for a given rotation,
    # the minimal enclosing hexagon centered optimally has D = half the
    # range of projections on each axis; take max over axes.
    best = float('inf')
    # coarse + fine scan of rotation (0..60 deg by symmetry)
    for beta_deg in np.arange(0.0, 60.0, 1.0):
        b = math.radians(beta_deg)
        c, s = math.cos(b), math.sin(b)
        R = np.array([[c, -s], [s, c]])
        proj = points @ R.T  # (n,2)
        # rotate normals by beta: normal_k = R @ _NORMALS[k]
        # projection of point p onto normal n = p . n
        # = (R^T p) . n ... easier: proj_n = points @ (R @ n)
        # = points @ (R @ _NORMALS[k])
        # Compute max |dot| over all points and all 6 normals
        RN = R @ _NORMALS.T  # (2,6)
        P = points @ RN      # (n,6)
        D = np.max(np.abs(P))
        best = min(best, D)
    # refine around best
    return best + APOTHEM  # side = (D + apothem)/apothem * apothem? see below


def min_side_side(points):
    """Return minimal side length S. S = (D + APOTHEM)/APOTHEM where D is
    the max |projection| distance from center. We allow center optimization:
    for a fixed rotation the optimal center minimizes the max |proj|, which
    equals half the range of projections on the worst axis."""
    best = float('inf')
    for beta_deg in np.arange(0.0, 60.0, 1.0):
        b = math.radians(beta_deg)
        c, s = math.cos(b), math.sin(b)
        R = np.array([[c, -s], [s, c]])
        RN = R @ _NORMALS.T  # (2,6)
        P = points @ RN      # (n,6)
        # optimal center along each normal: midpoint of min/max projection
        # D_k = (max_k - min_k)/2 ; D = max_k D_k
        D = np.max((np.max(P, axis=0) - np.min(P, axis=0)) / 2.0)
        S = (D + APOTHEM) / APOTHEM
        best = min(best, S)
    return best


def objective(params):
    """params: [x0,y0, x1,y1, ..., x10,y10, angle_deg]"""
    centers = params[:2 * N].reshape(N, 2)
    angle_deg = params[2 * N]
    angle_deg = ((angle_deg % 60.0) + 60.0) % 60.0

    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    verts_rot = _HEX_VERTS @ R.T  # (6,2)

    # overlap penalty: for each pair, check min distance between hexagons.
    # Two regular hexagons (same orientation, unit side) are disjoint iff
    # their center distance >= 2*APOTHEM * (some factor) along each of the
    # 3 relevant directions. For same-orientation unit hexagons, the Minkowski
    # difference is a hexagon of side 2, so they are disjoint iff the center
    # difference is outside that hexagon, i.e. max_k |dot(diff, n_k)| >= 2*APOTHEM.
    # We use the 6 normals (pointy-top).
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            proj = np.abs(diff @ _NORMALS.T)  # (6,)
            m = np.max(proj)
            if m < 2.0 * APOTHEM:
                pen += (2.0 * APOTHEM - m) ** 2

    # container side
    all_pts = np.vstack([centers[i] + verts_rot for i in range(N)])
    # minimal enclosing hexagon side (with center optimization)
    best_S = float('inf')
    for beta_deg in np.arange(0.0, 60.0, 2.0):
        b = math.radians(beta_deg)
        cc, ss = math.cos(b), math.sin(b)
        RB = np.array([[cc, -ss], [ss, cc]])
        RN = RB @ _NORMALS.T
        P = all_pts @ RN
        D = np.max((np.max(P, axis=0) - np.min(P, axis=0)) / 2.0)
        S = (D + APOTHEM) / APOTHEM
        best_S = min(best_S, S)

    return best_S + 100.0 * pen


def lattice_init(seed=0):
    """Generate a starting configuration from a hexagonal lattice."""
    rng = np.random.default_rng(seed)
    # hex lattice: axial (q,r) -> cartesian
    def axial_to_cart(q, r):
        return np.array([math.sqrt(3) * (q + r / 2.0), 1.5 * r])

    # pick 11 cells near origin in a compact shape
    cells = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    cells += ring1  # 7
    # add 4 more from ring 2 in a compact way
    ring2 = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 1), (-2, 2), (1, 2), (-1, -2)]
    # choose 4
    extra = rng.choice(len(ring2), size=4, replace=False)
    cells += [ring2[i] for i in extra]
    centers = np.array([axial_to_cart(q, r) for q, r in cells])
    angle_deg = rng.uniform(0, 60)
    params = np.concatenate([centers.ravel(), [angle_deg]])
    return params


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_params = None
    best_val = float('inf')
    best_side = float('inf')

    # Multi-start
    n_starts = 40
    for k in range(n_starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if k == 0:
            params0 = lattice_init(0)
        else:
            # perturb the best so far, or a new lattice with different seed
            if best_params is not None and k % 2 == 0:
                rng = np.random.default_rng(k)
                params0 = best_params + rng.normal(0, 0.1, size=best_params.shape)
            else:
                params0 = lattice_init(k)

        try:
            res = minimize(objective, params0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        best_params = lattice_init(0)

    # Extract centers and angle
    centers = best_params[:2 * N].reshape(N, 2)
    angle_deg = ((best_params[2 * N] % 60.0) + 60.0) % 60.0

    # Compute the minimal enclosing hexagon precisely (center + rotation)
    all_pts = np.vstack([centers[i] + hexagon_vertices(0, 0, angle_deg) for i in range(N)])

    # Find best rotation and center
    best_S = float('inf')
    best_beta = 0.0
    best_center = np.array([0.0, 0.0])
    for beta_deg in np.arange(0.0, 60.0, 0.5):
        b = math.radians(beta_deg)
        cc, ss = math.cos(b), math.sin(b)
        RB = np.array([[cc, -ss], [ss, cc]])
        RN = RB @ _NORMALS.T
        P = all_pts @ RN
        # optimal center: midpoint of projections on each axis -> in 2D,
        # we need a center that minimizes max |proj - c_proj|.
        # For a hexagon, the center in the rotated frame: cx_k = (max+min)/2
        # for each of the 6 normals; the 2D center is the intersection.
        # Approximate: use the center of mass, then compute D.
        # Better: solve for center via the fact that D = max_k (max_k - min_k)/2
        # and the center is at the midpoint in each normal direction.
        mids = (np.max(P, axis=0) + np.min(P, axis=0)) / 2.0  # (6,)
        # center in rotated frame: project midpoints back. The center c_rot
        # satisfies R @ c_rot = c. For each normal n, n . c = mid_k.
        # Solve least squares: _NORMALS_rot @ c_rot = mids
        c_rot, _, _, _ = np.linalg.lstsq(_NORMALS.T, mids, rcond=None)
        c = RB.T @ c_rot
        D = np.max((np.max(P, axis=0) - np.min(P, axis=0)) / 2.0)
        S = (D + APOTHEM) / APOTHEM
        if S < best_S:
            best_S = S
            best_beta = beta_deg
            best_center = c

    # Center the inner hexagons relative to the container center
    inner = []
    for i in range(N):
        x = float(centers[i, 0] - best_center[0])
        y = float(centers[i, 1] - best_center[1])
        inner.append([x, y, float(angle_deg)])

    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
