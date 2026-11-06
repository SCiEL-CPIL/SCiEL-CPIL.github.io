"""
Packing 50 equal circles in a unit circle.

Strategy:
1. Multiple diverse initial layouts (hexagonal lattice, concentric rings, random).
2. Smooth log-sum-exp objective with vectorized gradient for fast L-BFGS-B.
3. Nelder-Mead refinement on the kink.
4. Direct radius maximization via binary search with feasibility check.
5. Deadline guard to ensure we return within time limit.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _compute_radius(P):
    """Compute the packing radius for configuration P. Returns 0 if invalid."""
    n = P.shape[0]
    # Check all points are inside the unit circle
    dists = np.sqrt(np.sum(P * P, axis=1))
    if np.any(dists > 1.0 + 1e-10):
        return 0.0
    # Pairwise distances
    diff = P[:, None, :] - P[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    inside = 1.0 - dists.max()
    r = min(min_pair, inside)
    if r < 0.0:
        return 0.0
    return float(r)


def _soft_min(vals, t):
    """Log-sum-exp soft minimum."""
    m = vals.min()
    if not np.isfinite(m):
        return 0.0
    return m - t * np.log(np.sum(np.exp(-(vals - m) / t)))


def _objective(P, t):
    """Negative soft-min radius objective."""
    n = P.shape[0]
    diff = P[:, None, :] - P[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-15)
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    inside = 1.0 - np.sqrt(np.sum(P * P, axis=1))
    vals = np.concatenate([pair.ravel(), inside.ravel()])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return -_soft_min(vals, t)


def _objective_grad(P, t):
    """Gradient of the negative soft-min radius objective."""
    n = P.shape[0]
    diff = P[:, None, :] - P[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-15)
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    inside = 1.0 - np.sqrt(np.sum(P * P, axis=1))
    vals = np.concatenate([pair.ravel(), inside.ravel()])
    vals = vals[np.isfinite(vals)]
    m = vals.min()
    w = np.exp(-(vals - m) / t)
    ssum = np.sum(w)
    
    grad = np.zeros_like(P)
    
    # Pair contributions
    pw = w[:n * n].reshape(n, n)
    invd = np.zeros_like(d)
    mask = d > 1e-12
    invd[mask] = 1.0 / d[mask]
    # d(pair_ij)/dP_i = (P_i - P_j) / (2 * d_ij)
    gpair = np.sum(diff * (pw * invd / 2.0)[:, :, None], axis=1)
    
    # Inside contributions
    iw = w[n * n:]
    rad = np.sqrt(np.sum(P * P, axis=1))
    rad_safe = np.where(rad > 1e-12, rad, 1e-12)
    # d(inside_i)/dP_i = -P_i / |P_i|
    ginside = -iw[:, None] * (P / rad_safe[:, None])
    
    grad = gpair + ginside
    # grad of objective = -grad of soft_min = -(1/ssum) * sum(w * d(vals)/dP)
    grad = -grad / ssum
    return grad


def _hex_seed(r, jitter_seed=None):
    """Generate hexagonal lattice points clipped to unit disk."""
    s = 2.0 * r
    K = int(np.ceil(1.0 / s)) + 2
    h = s * np.sqrt(3.0) / 2.0
    pts = []
    for j in range(-K, K + 1):
        y = j * h
        if abs(y) > 1.0:
            continue
        xoff = (s / 2.0) if (j % 2) else 0.0
        for i in range(-K, K + 1):
            x = i * s + xoff
            if x * x + y * y <= 1.0:
                pts.append((x, y))
    pts = np.array(pts, dtype=np.float64)
    if pts.shape[0] < N:
        return None
    # Select N closest to center
    order = np.argsort(np.sum(pts * pts, axis=1), kind="stable")
    sel = pts[order[:N]].copy()
    if jitter_seed is not None:
        rng = np.random.default_rng(jitter_seed)
        sel = sel + rng.uniform(-0.3, 0.3, size=sel.shape) * r
    return sel


def _ring_seed():
    """Concentric rings with hexagonal offsets."""
    pts = []
    # Center point
    pts.append((0.0, 0.0))
    # Ring 1: 6 points
    r1 = 0.15
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts.append((r1 * np.cos(ang), r1 * np.sin(ang)))
    # Ring 2: 12 points
    r2 = 0.30
    for k in range(12):
        ang = 2 * np.pi * k / 12 + np.pi / 12
        pts.append((r2 * np.cos(ang), r2 * np.sin(ang)))
    # Ring 3: 18 points
    r3 = 0.45
    for k in range(18):
        ang = 2 * np.pi * k / 18
        pts.append((r3 * np.cos(ang), r3 * np.sin(ang)))
    # Ring 4: 15 points to reach 50
    r4 = 0.60
    for k in range(15):
        ang = 2 * np.pi * k / 15 + np.pi / 15
        pts.append((r4 * np.cos(ang), r4 * np.sin(ang)))
    
    pts = np.array(pts[:N], dtype=np.float64)
    return pts


def _random_seed(seed):
    """Random points inside unit disk."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < N:
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            pts.append((x, y))
    return np.array(pts, dtype=np.float64)


def _optimize_from_seed(P0, t, deadline):
    """Run L-BFGS-B then Nelder-Mead from seed P0."""
    best = P0.copy()
    best_r = _compute_radius(P0)
    
    # L-BFGS-B
    try:
        res = minimize(
            _objective, P0, jac=_objective_grad, method="L-BFGS-B",
            args=(t,),
            options={"maxiter": 80, "gtol": 1e-6, "maxfun": 500},
        )
        P1 = np.asarray(res.x, dtype=np.float64)
        # Project back inside unit circle
        rad = np.sqrt(np.sum(P1 * P1, axis=1))
        mask = rad > 1.0
        P1[mask] = P1[mask] / rad[mask, None] * 0.999
        r1 = _compute_radius(P1)
        if r1 > best_r:
            best, best_r = P1, r1
    except Exception:
        pass
    
    # Nelder-Mead refinement
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                _objective, best, method="Nelder-Mead", args=(t,),
                options={"maxiter": 60, "xatol": 1e-8, "fatol": 1e-10},
            )
            P2 = np.asarray(res2.x, dtype=np.float64)
            rad = np.sqrt(np.sum(P2 * P2, axis=1))
            mask = rad > 1.0
            P2[mask] = P2[mask] / rad[mask, None] * 0.999
            r2 = _compute_radius(P2)
            if r2 > best_r:
                best, best_r = P2, r2
        except Exception:
            pass
    
    return best, best_r


def _binary_search_radius(P_init, deadline):
    """Binary search for maximum feasible radius using local optimization."""
    r_lo = 0.1
    r_hi = 0.14
    
    # First, try to improve from a good starting point
    best_P = P_init.copy()
    best_r = _compute_radius(P_init)
    
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        mid_r = (r_lo + r_hi) / 2
        # Try to find feasible configuration with radius mid_r
        # Use the current best as starting point and optimize
        P_opt, r_opt = _optimize_from_seed(best_P, 0.003, deadline)
        if r_opt > best_r:
            best_r, best_P = r_opt, P_opt
        # Update bounds based on achieved radius
        if r_opt >= mid_r:
            r_lo = mid_r
        else:
            r_hi = mid_r
    
    return best_P, best_r


def construct():
    """Main entry point. Returns (N, 2) array of circle centers."""
    deadline = time.time() + 145.0  # Safety margin below 150s
    
    best_P = None
    best_r = 0.0
    
    # Generate diverse initial seeds
    seeds = []
    
    # Hexagonal lattice seeds at different radii
    for base_r in [0.12, 0.125, 0.13]:
        for js in [None, 0, 1, 2]:
            P = _hex_seed(base_r, js)
            if P is not None:
                seeds.append(P)
    
    # Ring seed
    seeds.append(_ring_seed())
    
    # Random seeds
    for s in [42, 123, 456]:
        seeds.append(_random_seed(s))
    
    # Optimize from each seed
    t_values = [0.003, 0.005, 0.01]
    
    for seed_idx, P0 in enumerate(seeds):
        if time.time() > deadline - 20:
            break
        
        # Try different temperature values
        for t in t_values:
            if time.time() > deadline - 10:
                break
            
            P_opt, r_opt = _optimize_from_seed(P0, t, deadline)
            if r_opt > best_r:
                best_r, best_P = r_opt, P_opt
    
    # Final refinement with binary search on the best configuration
    if best_P is not None and time.time() < deadline - 15:
        P_final, r_final = _binary_search_radius(best_P, deadline)
        if r_final > best_r:
            best_r, best_P = r_final, P_final
    
    # Fallback: simple ring packing
    if best_P is None:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_P = np.column_stack([0.6 * np.cos(ang), 0.6 * np.sin(ang)])
    
    return best_P
