"""Packing 12 unit hexagons in a minimal regular hexagon: Lattice-based multi-start Powell."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)
APO = SQRT3 / 2.0  # apothem of unit hexagon

# Unit flat-top hexagon: vertices (radius 1) and outward unit edge-normals
_HEX_V = np.array([
    [1.0, 0.0], [0.5, APO], [-0.5, APO],
    [-1.0, 0.0], [-0.5, -APO], [0.5, -APO],
])
_HEX_N = np.array([
    [SQRT3/2, 0.5], [0.0, 1.0], [-SQRT3/2, 0.5],
    [-SQRT3/2, -0.5], [0.0, -1.0], [SQRT3/2, -0.5],
])
_OUTER_N = _HEX_N  # outer hexagon (flat-top) unit normals

_DEADLINE = 145.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _compute(x, y, a, lam):
    """Return (S, overlap_penalty). x,y,a are (N,) arrays."""
    ca = np.cos(a); sa = np.sin(a)
    vx = _HEX_V[:, 0]; vy = _HEX_V[:, 1]
    # all vertices (N,6,2)
    all_vx = ca[:, None] * vx[None, :] - sa[:, None] * vy[None, :] + x[:, None]
    all_vy = sa[:, None] * vx[None, :] + ca[:, None] * vy[None, :] + y[:, None]
    # S: outer center origin, outer rotation 0
    on0 = _OUTER_N[:, 0]; on1 = _OUTER_N[:, 1]
    dots = all_vx[:, :, None] * on0[None, None, :] + all_vy[:, :, None] * on1[None, None, :]
    S = float(np.max(dots)) / APO
    if S < 0.0:
        S = 0.0
    if lam == 0.0:
        return S, 0.0
    # normals (N,6,2)
    nx = _HEX_N[:, 0]; ny = _HEX_N[:, 1]
    all_nx = ca[:, None] * nx[None, :] - sa[:, None] * ny[None, :]
    all_ny = sa[:, None] * nx[None, :] + ca[:, None] * ny[None, :]
    pen = 0.0
    for i in range(N):
        vi_x = all_vx[i]; vi_y = all_vy[i]
        ni_x = all_nx[i]; ni_y = all_ny[i]
        for j in range(i + 1, N):
            ax = np.empty(12); ay = np.empty(12)
            ax[:6] = ni_x; ay[:6] = ni_y
            ax[6:] = all_nx[j]; ay[6:] = all_ny[j]
            pi = vi_x[:, None] * ax[None, :] + vi_y[:, None] * ay[None, :]  # (6,12)
            pj = all_vx[j, :, None] * ax[None, :] + all_vy[j, :, None] * ay[None, :]
            min_i = pi.min(axis=0); max_i = pi.max(axis=0)
            min_j = pj.min(axis=0); max_j = pj.max(axis=0)
            sep = np.maximum(min_i - max_j, min_j - max_i).max()
            if sep < 0.0:
                d = -sep
                pen += d * d
    return S, lam * pen

def _objective(v, lam):
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
    S, pen = _compute(x, y, a, lam)
    return S + pen

def _is_valid(x, y, a, tol=1e-6):
    S, raw_pen = _compute(x, y, a, 1.0)
    return raw_pen < tol, S

def _fallback():
    # guaranteed-valid: 2 rows x 6, wide spacing
    inner = []
    for r in range(2):
        for c in range(6):
            inner.append([3.2 * (c - 2.5), 2.9 * r, 0.0])
    return inner, (0.0, 0.0), 22.0, 0.0

def _generate_lattice_initials():
    """Generate initial configurations based on honeycomb lattice."""
    initials = []
    # Honeycomb lattice basis
    # Flat-top hexagons touch at distance sqrt(3)
    # Basis vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    v1 = np.array([SQRT3, 0.0])
    v2 = np.array([SQRT3/2, 1.5])
    
    # Generate a patch of lattice points
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            points.append(p)
    points = np.array(points)
    
    # We want to select 12 points that form a compact cluster.
    # Strategy: Center the cluster at origin, pick 12 closest to a random center?
    # Or just pick specific known good clusters.
    
    # Cluster 1: 7 central + 5 from next ring
    # Central 7:
    c1 = np.array([
        [0,0],
        [1,0], [1,-1], [0,-1], [-1,0], [-1,1], [0,1]
    ])
    # Next ring points (distance 2 in axial coords)
    # Axial to Cartesian: x = sqrt(3)*(i + j/2), y = 1.5*j
    # Wait, my v1, v2 correspond to axial (1,0) and (0,1)?
    # v1 = (sqrt(3), 0). Axial (1,0) -> x=sqrt(3), y=0. Correct.
    # v2 = (sqrt(3)/2, 1.5). Axial (0,1) -> x=sqrt(3)/2, y=1.5. Correct.
    
    def axial_to_cart(i, j):
        return np.array([SQRT3 * (i + 0.5*j), 1.5 * j])
        
    ring2_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            d = 0.5 * (abs(i) + abs(j) + abs(i+j))
            if d == 2:
                ring2_points.append(axial_to_cart(i, j))
    ring2_points = np.array(ring2_points)
    
    # Try different subsets of 5 from ring2
    # To be efficient, we'll just create a few random subsets and one symmetric-ish one
    rng = np.random.default_rng(42)
    
    for k in range(10):
        if k == 0:
            # Symmetric-ish: pick every other one?
            # Just pick first 5
            subset = ring2_points[:5]
        else:
            idx = rng.choice(len(ring2_points), 5, replace=False)
            subset = ring2_points[idx]
            
        cluster = np.vstack([c1, subset])
        # Center the cluster
        center = cluster.mean(axis=0)
        cluster = cluster - center
        
        x = cluster[:, 0]
        y = cluster[:, 1]
        a = np.zeros(N)
        
        v = np.concatenate([x, y, a])
        initials.append(v)
        
    # Also add a pure random start
    for k in range(5):
        x = rng.uniform(-4, 4, N)
        y = rng.uniform(-4, 4, N)
        a = rng.uniform(0, np.pi/3, N)
        v = np.concatenate([x, y, a])
        initials.append(v)
        
    return initials

def construct():
    _T0[0] = time.time()
    rng = np.random.default_rng(42)
    
    best = None
    best_f = np.inf
    
    initials = _generate_lattice_initials()
    
    # Phase 1: Local refinement from lattice-based and random starts
    # Use Powell method which handles non-smoothness (due to max/abs in S and overlap) well
    for c0 in initials:
        if _elapsed() > _DEADLINE - 5:
            break
        try:
            # Start with moderate penalty
            r = minimize(lambda v: _objective(v, 10.0), c0, method='Powell',
                         options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9})
            if r.x is not None and r.fun < best_f:
                best_f = float(r.fun)
                best = np.asarray(r.x, dtype=float)
        except Exception:
            pass
            
    # Phase 2: Polish the best candidate with high penalty to ensure validity
    if best is not None:
        if _elapsed() < _DEADLINE - 5:
            try:
                r2 = minimize(lambda v: _objective(v, 100.0), best, method='Powell',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if r2.x is not None:
                    # Check if it's valid and better
                    x2 = r2.x[0:N]; y2 = r2.x[N:2*N]; a2 = r2.x[2*N:3*N]
                    ok, S2 = _is_valid(x2, y2, a2)
                    if ok and S2 < best_f:
                        best_f = S2
                        best = r2.x
                    elif not ok and r2.fun < best_f:
                        # Keep the lower objective even if not perfectly valid yet, 
                        # but we need to return a valid one. 
                        # So we stick with the last known valid or try to fix.
                        pass
            except Exception:
                pass

    if best is None:
        return _fallback()

    x = best[0:N]; y = best[N:2*N]; a = best[2*N:3*N]
    ok, S = _is_valid(x, y, a)
    if not ok:
        # Final attempt to fix validity with very high penalty
        try:
            r3 = minimize(lambda v: _objective(v, 500.0), best, method='Powell',
                          options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-12})
            if r3.x is not None:
                x = r3.x[0:N]; y = r3.x[N:2*N]; a = r3.x[2*N:3*N]
                ok, S = _is_valid(x, y, a)
        except Exception:
            pass
            
    if not ok:
        return _fallback()

    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
