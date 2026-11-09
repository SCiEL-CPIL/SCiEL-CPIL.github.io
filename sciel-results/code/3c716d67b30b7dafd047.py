"""Packing 12 unit hexagons in a minimal regular hexagon: symmetry-reduced Nelder-Mead."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)
APO = SQRT3 / 2.0  # apothem of unit hexagon

# Unit flat-top hexagon vertices and outward normals
_HEX_V = np.array([
    [1.0, 0.0], [0.5, APO], [-0.5, APO],
    [-1.0, 0.0], [-0.5, -APO], [0.5, -APO],
])
_HEX_N = np.array([
    [SQRT3/2, 0.5], [0.0, 1.0], [-SQRT3/2, 0.5],
    [-SQRT3/2, -0.5], [0.0, -1.0], [SQRT3/2, -0.5],
])
_OUTER_N = _HEX_N

_DEADLINE = 135.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _hex_vertices(x, y, a):
    """Return (6,2) array of vertices for a hexagon at (x,y) with angle a."""
    ca = np.cos(a); sa = np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return R @ _HEX_V.T + np.array([x, y])[:, None]

def _hex_normals(x, y, a):
    """Return (6,2) array of outward normals for a hexagon at (x,y) with angle a."""
    ca = np.cos(a); sa = np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return R @ _HEX_N.T

def _min_separation(v1, v2, n1, n2):
    """Compute minimum separation between two hexagons. Returns negative if overlapping."""
    # For each pair of edges (one from each hex), compute the separation
    # Using the support function approach
    min_sep = np.inf
    for i in range(6):
        for j in range(6):
            # Edge i of hex1, edge j of hex2
            # The separation along normal ni of hex1
            d1 = np.dot(v1[i], n1[i])  # support of hex1 along n1[i]
            d2 = np.dot(v2[j], n2[j])  # support of hex2 along n2[j]
            # But we need to project both hexagons onto the same direction
            # Actually, use the standard convex polygon separation
            pass
    # Better: use the convex polygon intersection test
    # For two convex polygons, they are disjoint iff there exists a separating axis
    # The candidate axes are the edge normals of both polygons
    axes = np.vstack([n1, n2])  # (12, 2)
    for ax in axes:
        proj1 = v1 @ ax
        proj2 = v2 @ ax
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Separation: gap between the two intervals
        sep = max(min1 - max2, min2 - max1)
        if sep < 0:
            # Overlapping on this axis
            min_sep = min(min_sep, sep)
        else:
            # Separated on this axis, so no overlap
            return sep
    return min_sep  # Negative if overlapping

def _compute_S_and_overlap(x, y, a, S):
    """Compute containment violation and overlap penalty."""
    # Containment: for each inner hexagon, max projection onto outer normals <= S*APO
    total_viol = 0.0
    for i in range(N):
        vi = _hex_vertices(x[i], y[i], a[i])
        for j in range(6):
            proj = vi[:, j] @ _OUTER_N[j]
            if proj > S * APO:
                total_viol += (proj - S * APO) ** 2
    
    # Overlap: for each pair, compute separation
    overlap_pen = 0.0
    for i in range(N):
        vi = _hex_vertices(x[i], y[i], a[i])
        ni = _hex_normals(x[i], y[i], a[i])
        for j in range(i + 1, N):
            vj = _hex_vertices(x[j], y[j], a[j])
            nj = _hex_normals(x[j], y[j], a[j])
            sep = _min_separation(vi, vj, ni, nj)
            if sep < 0:
                overlap_pen += (-sep) ** 2
    return total_viol, overlap_pen

def _symmetric_params_to_full(p):
    """Convert 7 symmetric parameters to full 36-dim array.
    p = [r1, theta1, alpha1, r2, theta2, alpha2, S]
    """
    r1, th1, a1, r2, th2, a2, S = p
    x = np.zeros(N); y = np.zeros(N); ang = np.zeros(N)
    for k in range(6):
        rot = k * np.pi / 3.0
        x[k] = r1 * np.cos(th1 + rot)
        y[k] = r1 * np.sin(th1 + rot)
        ang[k] = a1 + rot
        x[k + 6] = r2 * np.cos(th2 + rot)
        y[k + 6] = r2 * np.sin(th2 + rot)
        ang[k + 6] = a2 + rot
    return np.concatenate([x, y, ang, [S]])

def _symmetric_objective(p):
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x, y, a, S = _symmetric_params_to_full(p)
    x = x[:-1]; y = y[:-1]; a = a[:-1]; S = float(S[-1])
    viol, ov = _compute_S_and_overlap(x, y, a, S)
    return S + 10.0 * viol + 10.0 * ov

def _full_objective(v, lam_viol=10.0, lam_ov=10.0):
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]; S = float(v[3*N])
    viol, ov = _compute_S_and_overlap(x, y, a, S)
    return S + lam_viol * viol + lam_ov * ov

def _is_valid(x, y, a, S, tol=1e-6):
    viol, ov = _compute_S_and_overlap(x, y, a, S)
    return viol < tol and ov < tol, S

def construct():
    _T0[0] = time.time()
    rng = np.random.default_rng(42)
    
    best_full = None
    best_S = np.inf
    
    # Symmetric initial guesses
    # Orbit 1: hexagons near the edges
    # Orbit 2: hexagons in the interior
    inits = [
        [1.5, 0.0, 0.0, 0.8, np.pi/6, 0.0, 3.0],
        [1.8, 0.0, 0.0, 0.5, np.pi/6, 0.0, 3.5],
        [2.0, 0.0, 0.0, 0.3, np.pi/6, 0.0, 4.0],
        [1.2, 0.0, 0.0, 1.0, np.pi/6, 0.0, 2.8],
        [1.5, np.pi/6, 0.0, 0.8, 0.0, 0.0, 3.0],
        [1.5, 0.0, 0.0, 0.8, np.pi/3, 0.0, 3.0],
    ]
    
    for p0 in inits:
        if _elapsed() > _DEADLINE - 15:
            break
        try:
            res = minimize(_symmetric_objective, p0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = float(res.fun)
                best_full = _symmetric_params_to_full(res.x)
        except Exception:
            pass
    
    if best_full is None:
        # Fallback: simple grid
        x = np.array([SQRT3 * (c - 2.5) for r in range(2) for c in range(6)])
        y = np.array([1.5 * r for r in range(2) for c in range(6)])
        a = np.zeros(N)
        S = 4.0
        best_full = np.concatenate([x, y, a, [S]])
        best_S = S
    
    # Polish with full 37-dim optimization
    v0 = best_full.copy()
    try:
        res2 = minimize(lambda v: _full_objective(v), v0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
        if res2.fun < best_S:
            best_full = res2.x
            best_S = float(res2.fun)
    except Exception:
        pass
    
    x = best_full[0:N]; y = best_full[N:2*N]; a = best_full[2*N:3*N]; S = float(best_full[3*N])
    ok, S = _is_valid(x, y, a, S)
    
    if not ok:
        # Try higher penalties
        try:
            res3 = minimize(lambda v: _full_objective(v, 50.0, 50.0), best_full, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            x = res3.x[0:N]; y = res3.x[N:2*N]; a = res3.x[2*N:3*N]; S = float(res3.x[3*N])
            ok, S = _is_valid(x, y, a, S)
        except Exception:
            pass
    
    if not ok:
        # Return best effort even if slightly invalid
        pass
    
    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
