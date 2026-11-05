"""Morse cluster global minima: structured templates + local polish + bounded global search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_DEADLINE = _START + 285.0  # safety margin before the 330s kill


def _remaining():
    return _DEADLINE - time.time()


def _morse_energy(x, rho):
    d = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    np.fill_diagonal(r, 0.0)
    e = np.exp(rho * (1.0 - r))
    return float(np.sum(e * (e - 2.0)))


def _local_polish(x, rho, maxiter=400, tol=1e-8):
    """Fast L-BFGS-B local minimization with a distance lower bound."""
    from scipy.optimize import minimize
    n = len(x)
    x = np.asarray(x, dtype=float)

    def fun(v):
        return _morse_energy(v.reshape(n, 3), rho)

    def jac(v):
        g = morse_energy_gradient(v.reshape(n, 3), rho)
        return g.reshape(-1)

    cons = []
    idx = np.triu_indices(n, k=1)
    for i, j in zip(*idx):
        cons.append({
            "type": "ineq",
            "fun": (lambda v, a=i, b=j: np.linalg.norm(v[3*a:3*a+3] - v[3*b:3*b+3]) - 0.5),
        })
    try:
        res = minimize(fun, x.reshape(-1), jac=jac, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-6},
                       constraints=cons)
        if res.success or res.fun < fun(x.reshape(-1)):
            return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        pass
    return x, _morse_energy(x, rho)


def _icosahedron():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, t], [0, 1, -t], [0, -1, t], [0, -1, -t],
        [1, t, 0], [1, -t, 0], [-1, t, 0], [-1, -t, 0],
        [t, 0, 1], [t, 0, -1], [-t, 0, 1], [-t, 0, -1],
    ])
    v /= np.linalg.norm(v[0])
    return v  # 12 vertices, unit radius


def _decahedral_cluster(n, rng):
    """Concentric decahedral shells (Mackay-like) for high-rho clusters."""
    verts = _icosahedron()
    pts = [np.zeros(3)]
    # shell radii and counts to reach n
    shells = [(1.0, 12), (1.63, 30), (2.3, 56), (2.9, 92)]
    for radius, count in shells:
        if len(pts) >= n:
            break
        if count == 12:
            cand = verts * radius
        elif count == 30:
            # 30 = 12 vertices + 18 edge midpoints
            cand = list(verts * radius)
            # add face-centre-like points
            face_dirs = []
            # generate 20 face normals via cross of adjacent vertices
            from itertools import combinations
            seen = set()
            for a, b in combinations(range(12), 2):
                f = np.cross(verts[a], verts[b])
                fn = f / (np.linalg.norm(f) + 1e-12)
                key = tuple(np.round(fn, 3))
                if key not in seen:
                    seen.add(key)
                    face_dirs.append(fn)
            # take first 18
            for fd in face_dirs[:18]:
                cand.append(fd * radius)
            cand = np.array(cand)
        else:
            # approximate spherical shell
            cand = rng.normal(size=(count, 3))
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12)
            cand *= radius
        pts.extend(cand.tolist())
    pts = np.array(pts[:n])
    # small random jitter to break exact symmetry (helps escape saddle)
    pts = pts + rng.normal(scale=0.01, size=pts.shape)
    return pts


def _fcc_cluster(n, rng):
    """FCC fragment for low-rho clusters."""
    pts = [np.zeros(3)]
    a = 1.0
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    for shell in range(0, 20):
        for i in range(-shell, shell+1):
            for j in range(-shell, shell+1):
                for k in range(-shell, shell+1):
                    for b in basis:
                        p = np.array([i,j,k]) * a * 0.577 + b * a * 0.577
                        if len(pts) >= n:
                            break
                        pts.append(p)
    pts = np.array(pts[:n])
    # center of mass to origin
    pts -= pts.mean(axis=0)
    pts = pts + rng.normal(scale=0.01, size=pts.shape)
    return pts


def _hcp_cluster(n, rng):
    pts = [np.zeros(3)]
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    for layer in range(0, 20):
        z = layer * c / 2.0
        offset = (layer % 2) * 0.5
        for i in range(-layer-2, layer+3):
            for j in range(-layer-2, layer+3):
                x = (i + offset) * a
                y = j * a * np.sqrt(3.0)/2.0
                p = np.array([x, y, z])
                if len(pts) >= n:
                    break
                pts.append(p)
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    pts = pts + rng.normal(scale=0.01, size=pts.shape)
    return pts


def _random_compact(n, rng, scale=1.0):
    pts = [np.zeros(3)]
    r = scale * n ** (1.0/3.0) * 0.8
    while len(pts) < n:
        p = rng.uniform(-r, r, 3)
        if np.linalg.norm(p) <= r and all(np.linalg.norm(p - q) >= 0.55 for q in pts):
            pts.append(p)
    return np.array(pts)


def _de_global(x0, rho, maxiter=200, popsize=8, tol=1e-6):
    """Bounded Differential Evolution global search."""
    from scipy.optimize import differential_evolution
    n = len(x0)
    r = 1.0 * n ** (1.0/3.0) * 1.2
    bounds = [(-r, r)] * (3 * n)

    def fun(v):
        return _morse_energy(v.reshape(n, 3), rho)

    try:
        res = differential_evolution(fun, bounds, maxiter=maxiter, popsize=popsize,
                                     tol=tol, seed=42, polish=True,
                                     init="sobol", x0=x0.reshape(-1))
        if res.fun < _morse_energy(x0, rho):
            return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        pass
    return x0, _morse_energy(x0, rho)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size)

    # Build a pool of diverse initial guesses
    templates = []
    if rho >= 6:
        templates.append(_decahedral_cluster(size, rng))
        templates.append(_icosahedron_based(size, rng))
        templates.append(_random_compact(size, rng, 0.9))
        templates.append(_fcc_cluster(size, rng))
    else:
        templates.append(_fcc_cluster(size, rng))
        templates.append(_hcp_cluster(size, rng))
        templates.append(_random_compact(size, rng, 1.0))
        templates.append(_decahedral_cluster(size, rng))

    best_x = None
    best_e = np.inf

    # Relax each template locally
    for t in templates:
        if _remaining() < 20:
            break
        try:
            x, e = _local_polish(t, rho)
        except Exception:
            continue
        if e < best_e:
            best_e = e
            best_x = x

    # For small/medium clusters, attempt a bounded global DE search from the best local result
    if size <= 50 and best_x is not None and _remaining() > 40:
        try:
            gx, ge = _de_global(best_x, rho, maxiter=120, popsize=6)
            if ge < best_e:
                best_e = ge
                best_x = gx
        except Exception:
            pass

    # Final polish on the best
    if best_x is not None and _remaining() > 15:
        try:
            fx, fe = _local_polish(best_x, rho, maxiter=600, tol=1e-9)
            if fe < best_e:
                best_e = fe
                best_x = fx
        except Exception:
            pass

    if best_x is None:
        # fallback: random compact + single relax
        t = _random_compact(size, rng)
        try:
            best_x, best_e = _local_polish(t, rho)
        except Exception:
            best_x = t

    # Enforce minimum distance 0.5 by a final projection (simple)
    best_x = _enforce_min_dist(best_x, rho)
    return best_x


def _icosahedron_based(n, rng):
    """Icosahedral cluster: 1 + 12 + 42 + ... shells."""
    verts = _icosahedron()
    pts = [np.zeros(3)]
    # 12 vertices
    pts.extend((verts * 1.0).tolist())
    # 42 edge midpoints
    from itertools import combinations
    edge_pts = []
    for a, b in combinations(range(12), 2):
        d = verts[a] - verts[b]
        if 1.0 < np.linalg.norm(d) < 1.6:
            edge_pts.append((verts[a] + verts[b]) / 2.0 * 1.0)
    pts.extend(edge_pts[:42])
    # 60 face centres
    if len(pts) < n:
        face_dirs = []
        seen = set()
        for a, b in combinations(range(12), 2):
            f = np.cross(verts[a], verts[b])
            fn = f / (np.linalg.norm(f) + 1e-12)
            key = tuple(np.round(fn, 3))
            if key not in seen:
                seen.add(key)
                face_dirs.append(fn * 1.0)
        pts.extend(face_dirs[:60])
    # fill rest on a spherical shell
    if len(pts) < n:
        need = n - len(pts)
        extra = rng.normal(size=(need, 3))
        extra /= (np.linalg.norm(extra, axis=1, keepdims=True) + 1e-12)
        extra *= 1.8
        pts.extend(extra.tolist())
    pts = np.array(pts[:n])
    pts = pts + rng.normal(scale=0.01, size=pts.shape)
    return pts


def _enforce_min_dist(x, rho):
    """Nudge apart any pair closer than 0.5 (light correction)."""
    x = np.array(x, dtype=float)
    n = len(x)
    for it in range(50):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = x[i] - x[j]
                r = np.linalg.norm(d)
                if r < 0.5:
                    if r < 1e-8:
                        d = np.array([0.001, 0.002, 0.003])
                    push = (0.5 - r) / r * d * 0.5
                    x[i] += push
                    x[j] -= push
                    moved = True
        if not moved:
            break
    return x
# EVOLVE-BLOCK-END
