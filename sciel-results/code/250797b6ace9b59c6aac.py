"""Sutton-Chen transition-metal cluster global minima - growth/template + basin hopping."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Hard wall-clock deadline for all six instances combined.
_DEADLINE = time.time() + 300.0


def _energy(positions, n_exp, m_exp, c):
    """Pairwise Sutton-Chen energy (all pairs, no cutoff)."""
    n = positions.shape[0]
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    d2 = d * d
    s1 = np.sum(d2 ** (-n_exp / 2.0), axis=1)
    s2 = np.sum(d2 ** (-m_exp / 2.0), axis=1)
    return 0.5 * np.sum(s1 - c * np.sqrt(s2))


def _gradient(positions, n_exp, m_exp, c):
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    inv = 1.0 / d
    s1 = np.sum((d2 ** (-n_exp / 2.0)) * inv, axis=1)
    s2 = np.sum((d2 ** (-m_exp / 2.0)) * inv, axis=1)
    pref = 0.5 * (n_exp * s1 - 0.5 * c * m_exp * s2 / np.sqrt(np.maximum(s2, 1e-12)))
    f = pref[:, None] * inv * diff
    return np.sum(f, axis=1)


def _valid(positions, min_d=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    if positions.shape[1] != 3:
        return False
    tree = cKDTree(positions)
    dd = tree.query_pairs(min_d, output_type='ndarray')
    return len(dd) == 0


def _fcc_positions(n):
    """Truncated FCC cluster: place atoms at FCC lattice points nearest to the origin."""
    a = 1.0
    pts = []
    L = int(np.ceil(n ** (1.0 / 3.0)) + 3)
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                if (i + j) % 2 == 0 and (j + k) % 2 == 0 and (i + k) % 2 == 0:
                    pts.append(np.array([i, j, k]) * (a / np.sqrt(2.0)))
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order]
    if len(pts) >= n:
        return pts[:n].copy()
    # pad with random if too few
    rng = np.random.default_rng(0)
    extra = rng.normal(0, 0.5, (n - len(pts), 3))
    return np.vstack([pts, extra])


def _hcp_positions(n):
    """Truncated HCP cluster (ABAB stacking)."""
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a
    pts = []
    L = int(np.ceil(n ** (1.0 / 3.0)) + 3)
    for layer in range(-L, L + 1):
        z = layer * h
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                if layer % 2 == 0:
                    pts.append([x, y, z])
                else:
                    pts.append([x + a / 2.0, y + a * np.sqrt(3.0) / 6.0, z])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order]
    if len(pts) >= n:
        return pts[:n].copy()
    rng = np.random.default_rng(1)
    extra = rng.normal(0, 0.5, (n - len(pts), 3))
    return np.vstack([pts, extra])


def _icosahedral_positions(n):
    """Mackay-type icosahedral shells (1, 12, 42, 92, 140, ...)."""
    shells = [1, 12, 42, 92, 140, 202]
    # build shell point sets
    def shell_points(m):
        # vertices of an icosahedron scaled to shell m
        t = (1.0 + np.sqrt(5.0)) / 2.0
        base = np.array([
            [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
            [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
            [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
        ])
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        r = m * 1.0
        return base * r
    pts = [np.zeros(3)]  # center
    for m in range(1, 10):
        sp = shell_points(m)
        pts.append(sp)
    allp = np.vstack(pts)
    order = np.argsort(np.linalg.norm(allp, axis=1))
    allp = allp[order]
    if len(allp) >= n:
        return allp[:n].copy()
    rng = np.random.default_rng(2)
    extra = rng.normal(0, 0.5, (n - len(allp), 3))
    return np.vstack([allp, extra])


def _growth_positions(n, seed=0):
    """Growth: start from a small dense core and add atoms at low-energy interstitial sites."""
    rng = np.random.default_rng(seed)
    a = 1.0
    pts = [np.zeros(3)]
    # add a first icosahedral-ish shell
    for p in _icosahedral_positions(13)[:12]:
        pts.append(p)
    while len(pts) < n:
        arr = np.array(pts)
        tree = cKDTree(arr)
        # candidate interstitial sites: midpoints between close pairs
        pairs = tree.query_pairs(a * 1.2, output_type='ndarray')
        cands = []
        for (i, j) in pairs:
            mid = 0.5 * (arr[i] + arr[j])
            # push out from the two atoms
            for atom in (arr[i], arr[j]):
                v = mid - atom
                nv = np.linalg.norm(v)
                if nv > 1e-8:
                    mid = atom + (v / nv) * 1.0
            cands.append(mid)
        if not cands:
            cands = [arr[0] + rng.normal(0, 0.6, 3)]
        cands = np.array(cands)
        # score each candidate by local energy contribution
        best_p = None
        best_e = np.inf
        for c in cands:
            d2 = np.sum((arr - c) ** 2, axis=1)
            d2 = np.maximum(d2, 1e-8)
            s1 = np.sum(d2 ** (-6.0))
            s2 = np.sum(d2 ** (-3.0))
            e = s1 - np.sqrt(s2)
            if e < best_e:
                best_e = e
                best_p = c
        pts.append(best_p)
    return np.array(pts)


def _random_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in pts):
            pts.append(p)
    if len(pts) < n:
        pts = list(pts) + [rng.uniform(-radius, radius, 3) for _ in range(n - len(pts))]
    return np.array(pts[:n])


def _basin_hop(positions, n_exp, m_exp, c, max_iter, T, rng, deadline):
    """Basin hopping: perturb, relax, Metropolis accept."""
    best_x = positions.copy()
    best_e = _energy(best_x, n_exp, m_exp, c)
    cur_x = best_x.copy()
    cur_e = best_e
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        step = 0.15 * np.sqrt(len(cur_x))
        pert = cur_x + rng.normal(0, step / np.sqrt(3.0), cur_x.shape)
        res = relax(pert, 'sutton_chen', '12-6')
        new_x = res[1]
        new_e = _energy(new_x, n_exp, m_exp, c)
        if not np.all(np.isfinite(new_e)):
            continue
        if new_e < cur_e or rng.random() < np.exp(-(new_e - cur_e) / max(T, 1e-6)):
            cur_x, cur_e = new_x, new_e
            if cur_e < best_e:
                best_x, best_e = cur_x.copy(), cur_e
    return best_x, best_e


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    deadline = _DEADLINE - 5.0  # safety margin

    rng = np.random.default_rng(n * 7 + 13)
    T = 0.05

    # Build a diverse pool of starting templates.
    templates = []
    for seed in range(3):
        templates.append(_growth_positions(n, seed=seed))
    templates.append(_fcc_positions(n))
    templates.append(_hcp_positions(n))
    templates.append(_icosahedral_positions(n))
    for seed in range(3):
        templates.append(_random_positions(n, seed=seed))

    # Relax each template; keep valid ones.
    candidates = []
    for t in templates:
        if time.time() > deadline:
            break
        res = relax(t, 'sutton_chen', kind)
        e = _energy(res[1], n_exp, m_exp, c)
        if np.all(np.isfinite(e)) and _valid(res[1]):
            candidates.append((e, res[1]))

    if not candidates:
        # fallback: return a valid relaxed random
        res = relax(_random_positions(n), 'sutton_chen', kind)
        return res[1]

    candidates.sort(key=lambda x: x[0])
    best_x = candidates[0][1].copy()
    best_e = candidates[0][0]

    # Basin hopping from the best few templates while time remains.
    for e0, x0 in candidates[:4]:
        if time.time() > deadline:
            break
        bx, be = _basin_hop(x0, n_exp, m_exp, c, max_iter=40, T=T, rng=rng, deadline=deadline)
        if np.all(np.isfinite(be)) and _valid(bx) and be < best_e:
            best_e, best_x = be, bx

    # Final safety: ensure validity.
    if not _valid(best_x):
        for e0, x0 in candidates:
            if _valid(x0):
                best_x = x0
                break

    return best_x
