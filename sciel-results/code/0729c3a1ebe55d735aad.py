"""
Sutton-Chen transition-metal cluster global minima via multi-start basin-hopping
with diverse structural templates, adaptive perturbation, and high-precision polish.
"""

import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_GLOBAL_START = time.time()
_HARD_LIMIT = 325.0

def _remaining():
    return max(0.0, _HARD_LIMIT - (time.time() - _GLOBAL_START))

def _fcc_template(n, scale=1.0):
    a = scale * np.sqrt(2.0)
    r_est = (3.0 * n * a**3 / (16.0 * np.pi)) ** (1.0/3.0) * 1.25
    nside = int(np.ceil(r_est / a)) + 3
    cands = []
    for i in range(-nside, nside + 1):
        for j in range(-nside, nside + 1):
            for k in range(-nside, nside + 1):
                base = np.array([i, j, k], dtype=float) * a
                for off in [(0.0,0.0,0.0),(0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5)]:
                    pt = base + np.array(off, dtype=float) * a
                    if np.dot(pt, pt) <= r_est * r_est:
                        cands.append(pt)
    if len(cands) < n:
        r_est *= 1.5
        nside = int(np.ceil(r_est / a)) + 3
        cands = []
        for i in range(-nside, nside + 1):
            for j in range(-nside, nside + 1):
                for k in range(-nside, nside + 1):
                    base = np.array([i, j, k], dtype=float) * a
                    for off in [(0.0,0.0,0.0),(0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5)]:
                        pt = base + np.array(off, dtype=float) * a
                        if np.dot(pt, pt) <= r_est * r_est:
                            cands.append(pt)
    cands = np.array(cands)
    d = np.linalg.norm(cands, axis=1)
    order = np.argsort(d)
    cands = cands[order[:n]]
    return cands - cands.mean(axis=0)

def _hcp_template(n, scale=1.0):
    a = scale
    c = a * np.sqrt(8.0 / 3.0)
    nlay = int(np.ceil(np.sqrt(n))) + 4
    nside = int(np.ceil(np.sqrt(n))) + 4
    r2 = (n / 4.0) ** (2.0/3.0) * a**2 * 3.5
    cands = []
    for layer in range(-nlay, nlay + 1):
        z = layer * c
        sx = 0.0 if layer % 2 == 0 else a / 2.0
        sy = 0.0 if layer % 2 == 0 else a * np.sqrt(3.0) / 6.0
        for i in range(-nside, nside + 1):
            for j in range(-nside, nside + 1):
                x = i * a + j * a * 0.5 + sx
                y = j * a * np.sqrt(3.0) / 2.0 + sy
                if x*x + y*y + z*z <= r2:
                    cands.append([x, y, z])
    cands = np.array(cands)
    if len(cands) < n:
        rng = np.random.default_rng(7)
        while len(cands) < n:
            cands = np.vstack([cands, rng.uniform(-2*a, 2*a, 3)])
    d = np.linalg.norm(cands, axis=1)
    order = np.argsort(d)
    cands = cands[order[:n]]
    return cands - cands.mean(axis=0)

def _bcc_template(n, scale=1.0):
    a = scale * 2.0 / np.sqrt(3.0)
    r_est = (3.0 * n * a**3 / (8.0 * np.pi)) ** (1.0/3.0) * 1.25
    nside = int(np.ceil(r_est / a)) + 3
    cands = []
    for i in range(-nside, nside + 1):
        for j in range(-nside, nside + 1):
            for k in range(-nside, nside + 1):
                base = np.array([i, j, k], dtype=float) * a
                for off in [(0.0,0.0,0.0),(0.5,0.5,0.5)]:
                    pt = base + np.array(off, dtype=float) * a
                    if np.dot(pt, pt) <= r_est * r_est:
                        cands.append(pt)
    if len(cands) < n:
        r_est *= 1.5
        nside = int(np.ceil(r_est / a)) + 3
        cands = []
        for i in range(-nside, nside + 1):
            for j in range(-nside, nside + 1):
                for k in range(-nside, nside + 1):
                    base = np.array([i, j, k], dtype=float) * a
                    for off in [(0.0,0.0,0.0),(0.5,0.5,0.5)]:
                        pt = base + np.array(off, dtype=float) * a
                        if np.dot(pt, pt) <= r_est * r_est:
                            cands.append(pt)
    cands = np.array(cands)
    d = np.linalg.norm(cands, axis=1)
    order = np.argsort(d)
    cands = cands[order[:n]]
    return cands - cands.mean(axis=0)

def _icosahedral_template(n, scale=1.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
        [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
        [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    pts = [np.zeros(3)]
    if n > 1:
        for v in verts:
            if len(pts) < n:
                pts.append(v * scale)
    a = scale * np.sqrt(2.0)
    if len(pts) < n:
        for i in range(-5, 6):
            for j in range(-5, 6):
                for k in range(-5, 6):
                    base = np.array([i, j, k], dtype=float) * a
                    for off in [(0.0,0.0,0.0),(0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5)]:
                        p = base + np.array(off, dtype=float) * a
                        r = np.linalg.norm(p)
                        if 1.4 * scale < r < 3.2 * scale and len(pts) < n:
                            pts.append(p / r * 2.0 * scale)
    rng = np.random.default_rng(3)
    while len(pts) < n:
        p = rng.uniform(-3*scale, 3*scale, 3)
        r = np.linalg.norm(p)
        if r > 0.1:
            pts.append(p / r * (1.8 + rng.random() * 1.2) * scale)
    pts = np.array(pts[:n])
    return pts - pts.mean(axis=0)

def _random_packed(n, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    radius = scale * (n / 4.0) ** (1.0/3.0) * 2.8
    pts = []
    min_d = 0.55 * scale
    for _ in range(n * 200):
        if len(pts) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.dot(p, p) > radius * radius:
            continue
        if not pts:
            pts.append(p)
            continue
        arr = np.array(pts)
        diffs = arr - p
        if np.min(np.linalg.norm(diffs, axis=1)) >= min_d:
            pts.append(p)
    while len(pts) < n:
        pts.append(rng.uniform(-radius, radius, 3))
    pts = np.array(pts[:n])
    return pts - pts.mean(axis=0)

def _enforce_min_dist(pos, thresh=0.36):
    pos = pos.copy()
    n = len(pos)
    for _ in range(80):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < thresh:
                    moved = True
                    if d > 1e-12:
                        dirn = (pos[i] - pos[j]) / d
                    else:
                        dirn = np.array([1.0, 0.0, 0.0])
                    shift = (thresh - d) / 2.0 + 0.008
                    pos[i] += dirn * shift
                    pos[j] -= dirn * shift
        if not moved:
            break
    return pos

def _bh_loop(start_pos, potential, kind, rng, n_iter, scale0, scale_min, temp):
    best_e, best_p = relax(start_pos, potential, kind)
    cur_e, cur_p = best_e, best_p.copy()
    for it in range(n_iter):
        if _remaining() < 12.0:
            break
        sc = max(scale_min, scale0 * (0.96 ** (it // 8)))
        noise = rng.normal(0.0, sc, cur_p.shape)
        new_p = cur_p + noise
        try:
            e, p = relax(new_p, potential, kind)
        except Exception:
            continue
        if not np.isfinite(e):
            continue
        accept = False
        if e < cur_e:
            accept = True
        elif temp > 0:
            T = temp * (1.0 + it / 60.0)
            if rng.random() < np.exp(-(e - cur_e) / T):
                accept = True
        if accept:
            cur_e, cur_p = e, p.copy()
        if e < best_e:
            best_e, best_p = e, p.copy()
    return best_e, best_p

def _polish(pos, kind):
    def f(x):
        pp = x.reshape(-1, 3)
        e, _ = sutton_chen_energy_gradient(pp, kind)
        return e if np.isfinite(e) else 1e12

    def g(x):
        pp = x.reshape(-1, 3)
        _, gg = sutton_chen_energy_gradient(pp, kind)
        if not np.all(np.isfinite(gg)):
            return np.zeros_like(gg).ravel()
        return gg.ravel()

    x0 = pos.ravel()
    try:
        res = minimize(f, x0, jac=g, method='L-BFGS-B',
                       options={'maxiter': 20000, 'ftol': 1e-25, 'gtol': 1e-20, 'maxls': 2000})
        if np.isfinite(res.fun) and np.all(np.isfinite(res.x)):
            fp = res.x.reshape(-1, 3)
            if np.min(np.linalg.norm(fp[:, None, :] - fp[None, :, :], axis=2)[np.triu_indices(len(fp), 1)]) > 0.35:
                return res.fun, fp
    except Exception:
        pass
    return relax(pos, 'sutton_chen', kind)

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(42 + n * 7)

    best_e = np.inf
    best_p = None

    # Generate templates
    templates = []
    try:
        templates.append(_fcc_template(n))
    except Exception:
        pass
    try:
        templates.append(_hcp_template(n))
    except Exception:
        pass
    try:
        templates.append(_bcc_template(n))
    except Exception:
        pass
    try:
        templates.append(_icosahedral_template(n))
    except Exception:
        pass
    for s in range(25):
        try:
            templates.append(_random_packed(n, seed=s))
        except Exception:
            pass

    # Relax all templates
    relaxed = []
    for t in templates:
        if _remaining() < 25.0:
            break
        try:
            e, p = relax(t, potential, kind)
            if np.isfinite(e):
                relaxed.append((e, p))
                if e < best_e:
                    best_e, best_p = e, p.copy()
        except Exception:
            pass

    if best_p is None:
        best_p = _random_packed(n, seed=999)
        best_e, best_p = relax(best_p, potential, kind)

    # Sort relaxed by energy
    relaxed.sort(key=lambda x: x[0])

    # Basin hopping from best
    if _remaining() > 35.0:
        n_iters = max(40, int((_remaining() - 25.0) * 2.5))
        e, p = _bh_loop(best_p, potential, kind, rng, n_iters, 0.55, 0.15, 0.03)
        if e < best_e:
            best_e, best_p = e, p.copy()

    # Basin hopping from other good starts
    for e_t, p_t in relaxed[:12]:
        if _remaining() < 30.0:
            break
        if e_t > best_e + 10.0:
            continue
        try:
            e, p = _bh_loop(p_t, potential, kind, rng, 35, 0.5, 0.15, 0.025)
            if e < best_e:
                best_e, best_p = e, p.copy()
        except Exception:
            pass

    # High-precision polish
    if _remaining() > 18.0:
        try:
            e_pol, p_pol = _polish(best_p, kind)
            if np.isfinite(e_pol) and e_pol < best_e:
                best_e, best_p = e_pol, p_pol.copy()
        except Exception:
            pass

    # Second round of basin hopping from polished
    if _remaining() > 45.0:
        try:
            e, p = _bh_loop(best_p, potential, kind, rng, 45, 0.4, 0.1, 0.018)
            if e < best_e:
                best_e, best_p = e, p.copy()
        except Exception:
            pass

        # Final polish
        if _remaining() > 15.0:
            try:
                e_pol, p_pol = _polish(best_p, kind)
                if np.isfinite(e_pol) and e_pol < best_e:
                    best_e, best_p = e_pol, p_pol.copy()
            except Exception:
                pass

    # Enforce minimum distance
    if best_p is not None:
        best_p = _enforce_min_dist(best_p, 0.36)

    return best_p
