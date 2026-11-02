"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area; returns 0.0 on failure."""
    try:
        h = ConvexHull(pts)
        return float(h.volume)
    except Exception:
        return 0.0

def _min_tri_area(pts):
    """Minimum triangle area over all triples (vectorized cross products)."""
    n = len(pts)
    # Build all (i,j,k) with i<j<k
    idx_i = np.repeat(np.arange(n), n*(n-1)//2)
    # For each i, pairs (j,k) with j>k>i
    pairs = []
    for i in range(n-2):
        js = np.arange(i+1, n)
        ks = np.arange(i+2, n)
        for j in js:
            for k in ks:
                pairs.append((i, j, k))
    pairs = np.array(pairs)
    pi = pts[pairs[:, 0]]
    pj = pts[pairs[:, 1]]
    pk = pts[pairs[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(areas.min())

def _score(pts):
    """min triangle area / convex hull area."""
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_tri_area(pts) / ha

def _smooth_min(x, beta):
    """Soft-min via log-sum-exp (negative log-sum-exp of -x)."""
    m = np.min(x)
    if m == 0:
        return 0.0
    s = np.sum(np.exp(-(x - m) * beta))
    return m - np.log(s) / beta

def _neg_smooth_score(pts, beta=50.0):
    """Negative smooth surrogate of score for minimization."""
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e10
    n = len(pts)
    # all triples
    pairs = []
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                pairs.append((i, j, k))
    pairs = np.array(pairs)
    pi = pts[pairs[:, 0]]
    pj = pts[pairs[:, 1]]
    pk = pts[pairs[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # normalize by hull area to make scale-invariant
    norm_areas = areas / ha
    sm = _smooth_min(norm_areas, beta)
    return -sm

def _ring_init(center_mode, rng):
    """Generate a symmetric ring-based initialization."""
    if center_mode == 0:
        # 1 center + 12 on ring
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        r = 1.0
        pts[1:, 0] = r * np.cos(angles)
        pts[1:, 1] = r * np.sin(angles)
    elif center_mode == 1:
        # 1 center + 6 inner + 6 outer
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
        angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        r_in, r_out = 0.55, 1.0
        pts[1:7, 0] = r_in * np.cos(angles_in)
        pts[1:7, 1] = r_in * np.sin(angles_in)
        pts[7:13, 0] = r_out * np.cos(angles_out)
        pts[7:13, 1] = r_out * np.sin(angles_out)
    else:
        # 1 center + 12 ring with random perturbation
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False) + rng.uniform(-0.1, 0.1)
        r = 1.0
        pts[1:, 0] = r * np.cos(angles)
        pts[1:, 1] = r * np.sin(angles)
    return pts

def construct():
    best_pts = None
    best_score = -1.0
    rng = np.random.RandomState(N)

    inits = [
        _ring_init(0, rng),
        _ring_init(1, rng),
        _ring_init(2, rng),
        _ring_init(2, rng),
        _ring_init(2, rng),
    ]

    for init in inits:
        if time.time() > _DEADLINE - 15.0:
            break
        # Normalize to unit area
        ha = _hull_area(init)
        if ha > 1e-12:
            init = init / np.sqrt(ha)

        # Multi-start local optimization with smooth surrogate
        for beta in [30.0, 60.0, 100.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            x0 = init.flatten()
            try:
                res = minimize(
                    _neg_smooth_score,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                cand = res.x.reshape(N, 2)
            except Exception:
                continue

            # Polish with tighter beta
            if time.time() > _DEADLINE - 5.0:
                break
            try:
                res2 = minimize(
                    _neg_smooth_score,
                    cand.flatten(),
                    args=(200.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand = res2.x.reshape(N, 2)
            except Exception:
                pass

            sc = _score(cand)
            if sc > best_score:
                best_score = sc
                best_pts = cand.copy()

    if best_pts is None:
        # Fallback
        pts = _ring_init(0, rng)
        ha = _hull_area(pts)
        if ha > 1e-12:
            pts = pts / np.sqrt(ha)
        return pts

    return best_pts
# EVOLVE-BLOCK-END
