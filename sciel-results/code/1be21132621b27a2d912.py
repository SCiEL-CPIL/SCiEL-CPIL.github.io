"""
Lennard-Jones 55-atom cluster global minimum search.

Strategy: build the two-shell Mackay icosahedron with optimised shell radii
and a small symmetry-breaking radial displacement on the outer shell, then
polish with a high-precision multi-stage L-BFGS run.  The initial geometry
is already extremely close to the global minimum, so the expensive search
is dominated by the final convergence.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55

# ---------------------------------------------------------------------------
# Icosahedral geometry
# ---------------------------------------------------------------------------

def _get_icosahedron_vectors():
    """
    12 unit vectors along the 5-fold axes (icosahedron vertices) and
    30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    verts = np.array([
        [0,  1,  phi],
        [0, -1,  phi],
        [0,  1, -phi],
        [0, -1, -phi],
        [ 1, phi, 0],
        [-1, phi, 0],
        [ 1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0,  1],
        [-phi, 0,  1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=np.float64)

    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    dists = np.linalg.norm(
        verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2
    )
    min_dist = np.min(dists[dists > 1e-8])

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) * 0.5
                mid /= np.linalg.norm(mid)
                edge_mids.append(mid)

    return verts, np.array(edge_mids)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _mackay_icosahedron_55(r1, r2, r3, outer_delta=0.0):
    """
    55-atom two-shell Mackay icosahedron.
    1 centre + 12 @ r1 + 30 @ r2 + 12 @ r3.
    outer_delta: small per-atom radial jitter on the outer 12 (symmetry break).
    """
    v5, v2 = _get_icosahedron_vectors()
    pts = [np.zeros(3)]
    pts += [v * r1 for v in v5]
    pts += [v * r2 for v in v2]
    if outer_delta == 0.0:
        pts += [v * r3 for v in v5]
    else:
        # deterministic small jitter to break perfect icosahedral symmetry
        rng = np.random.default_rng(12345)
        for k, v in enumerate(v5):
            pts.append(v * (r3 + outer_delta * rng.standard_normal()))
    return np.array(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# LJ energy / gradient (vectorised)
# ---------------------------------------------------------------------------

def _lj_energy(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3.0)
    r12 = r6 * r6
    e = 4.0 * np.sum(r12 - r6)
    # the diagonal contributed 4*(1-1)=0, so no correction needed
    return e


def _lj_energy_grad(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3.0)
    r12 = r6 * r6
    e = 4.0 * np.sum(r12 - r6)
    # grad_i = sum_{j != i} 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    #       = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff_ij
    coeff = 4.0 * (-12.0 * r2 ** (-7.0) + 6.0 * r2 ** (-4.0))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad


def _is_valid(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.min(d2)) >= 0.5


# ---------------------------------------------------------------------------
# Relaxation helpers
# ---------------------------------------------------------------------------

def _relax(pos, maxiter=20000, gtol=1e-12, maxls=200):
    x0 = pos.ravel()

    def f(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            f, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': maxls},
        )
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _lj_energy(pos)


def _polish(pos, deadline=None):
    """Multi-stage L-BFGS with progressively tighter tolerances."""
    t0 = time.time()
    best_p = pos.copy()
    best_e = _lj_energy(pos)
    x0 = pos.ravel()

    stages = [
        dict(maxiter=5000,  gtol=1e-12, maxls=100),
        dict(maxiter=10000, gtol=1e-14, maxls=200),
        dict(maxiter=20000, gtol=1e-15, maxls=300),
        dict(maxiter=50000, gtol=1e-16, maxls=500),
    ]

    def f(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_grad(p)
        return e, g.ravel()

    for opts in stages:
        if deadline is not None and time.time() > deadline - 3.0:
            break
        try:
            res = minimize(f, x0, jac=True, method='L-BFGS-B', options=opts)
            if res.fun < best_e:
                best_e = res.fun
                best_p = res.x.reshape(-1, 3)
                x0 = res.x
        except Exception:
            pass

    return best_p, best_e


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def construct():
    t_start = time.time()
    deadline = t_start + 142.0   # 8 s safety margin before the 150 s kill

    # ---- 1. Build a set of near-optimal Mackay seeds -----------------------
    # The two known-good radii sets (from literature / CSD):
    seeds = [
        _mackay_icosahedron_55(1.135, 1.162, 1.283),
        _mackay_icosahedron_55(1.135, 1.162, 1.283, outer_delta=0.003),
        _mackay_icosahedron_55(1.130, 1.158, 1.280),
        _mackay_icosahedron_55(1.140, 1.165, 1.286),
    ]

    best_pos = seeds[0].copy()
    best_e = _lj_energy(seeds[0])

    # ---- 2. Relax each seed, keep the best --------------------------------
    for i, s in enumerate(seeds):
        if time.time() > deadline - 30.0:
            break
        p, e = _relax(s, maxiter=15000, gtol=1e-12, maxls=100)
        if _is_valid(p) and e < best_e:
            best_e = e
            best_pos = p

    # ---- 3. High-precision multi-stage polish ------------------------------
    if time.time() < deadline - 5.0:
        best_pos, best_e = _polish(best_pos, deadline=deadline)

    # ---- 4. If we still have time, try a few more radius variants ----------
    extra_radii = [
        (1.132, 1.160, 1.281),
        (1.138, 1.164, 1.285),
        (1.128, 1.155, 1.278),
        (1.142, 1.168, 1.288),
    ]
    for r1, r2, r3 in extra_radii:
        if time.time() > deadline - 10.0:
            break
        s = _mackay_icosahedron_55(r1, r2, r3)
        p, e = _relax(s, maxiter=10000, gtol=1e-12, maxls=100)
        if _is_valid(p) and e < best_e:
            best_e = e
            best_pos = p

    # Final polish if we improved
    if time.time() < deadline - 4.0:
        best_pos, best_e = _polish(best_pos, deadline=deadline)

    return best_pos
