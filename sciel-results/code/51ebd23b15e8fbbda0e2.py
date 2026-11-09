"""Sutton-Chen transition-metal cluster global minima: time-budgeted multi-start structured seeds + nested L-BFGS-B relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _deadline_ok(t0, margin=30.0, limit=330.0):
    return (time.time() - t0) < (limit - margin)


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = spacing * np.array([i, j, k], dtype=float)
                for b in basis:
                    sites.append(base + spacing * b)
    sites = np.array(sites)
    d = np.linalg.norm(sites, axis=1)
    idx = np.argsort(d)[:n]
    return sites[idx]


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    d = np.linalg.norm(sites, axis=1)
    idx = np.argsort(d)[:n]
    return sites[idx]


def _icosahedral_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    # shell radii (icosahedral)
    shell_r = [0.0, 1.0, np.sqrt(2.0), np.sqrt(2.0 + 2.0 / phi), np.sqrt(2.0 + 2.0 * phi)]
    # candidate points on each shell: vertices and midpoints of edges
    cand = [np.array([0.0, 0.0, 0.0])]
    for r in shell_r[1:]:
        for a in verts:
            cand.append(r * a)
        # edge midpoints (icosahedral edges connect verts at distance 2)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                dv = verts[i] - verts[j]
                if abs(np.linalg.norm(dv) - 2.0) < 1e-6:
                    mid = (verts[i] + verts[j]) / 2.0
                    cand.append(r * (mid / np.linalg.norm(mid)))
    cand = np.array(cand)
    # dedupe
    uniq = []
    for p in cand:
        if not any(np.allclose(p, u, atol=1e-6) for u in uniq):
            uniq.append(p)
    cand = np.array(uniq)
    d = np.linalg.norm(cand, axis=1)
    idx = np.argsort(d)[:n]
    return spacing * cand[idx]


def _compressed_icosahedral(n, spacing=0.72):
    """Icosahedral with a compressed inner region (mimics real TM clusters)."""
    base = _icosahedral_cluster(n, spacing)
    # pull inner atoms toward origin slightly to model the denser core
    r = np.linalg.norm(base, axis=1)
    scale = np.where(r < spacing * 1.5, 0.85, 1.0)
    return base * scale[:, None]


def _random_compact(n, spacing, seed):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in pts):
            pts.append(p)
    # fallback: fill remaining on a sphere shell
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _perturb(pts, sigma, seed):
    rng = np.random.default_rng(seed)
    return pts + sigma * rng.standard_normal(pts.shape)


def _nested_relax(pts, potential, parameter, t0):
    """Relax, then re-relax from the result to escape shallow local minima."""
    try:
        e1, p1 = relax(pts, potential, parameter)
    except Exception:
        return None, None
    if not np.all(np.isfinite(p1)):
        return None, None
    if not _deadline_ok(t0):
        return e1, p1
    # second pass from a tiny perturbation of the first result
    try:
        p2 = p1 + 0.02 * np.random.default_rng(int(e1 * 1e4) if e1 else 0).standard_normal(p1.shape)
        e2, p2 = relax(p2, potential, parameter)
        if e2 < e1:
            return e2, p2
    except Exception:
        pass
    return e1, p1


def construct(instance):
    t0 = time.time()
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)

    # Build the ordered list of starting configurations
    starts = []
    for gen in (_fcc_cluster, _hcp_cluster, _icosahedral_cluster, _compressed_icosahedral):
        starts.append(gen(size, spacing))
    # random compact with a few seeds
    for s in (size, size + 101, size + 202):
        starts.append(_random_compact(size, spacing, s))
    # perturbations of the structured seeds
    for base in starts[:4]:
        for sigma in (0.08, 0.15):
            starts.append(_perturb(base, sigma, size + int(sigma * 100)))

    best_energy = None
    best_pos = None
    for k, start in enumerate(starts):
        if not _deadline_ok(t0):
            break
        if best_pos is None:
            best_pos = _ensure_min_dist(start)  # always have a valid fallback
        e, p = _nested_relax(start, potential, parameter, t0)
        if p is not None and np.all(np.isfinite(p)):
            if best_energy is None or e < best_energy:
                best_energy = e
                best_pos = p
    # final validity guarantee
    return _ensure_min_dist(best_pos)
# EVOLVE-BLOCK-END
