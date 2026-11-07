"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate FCC-like cluster by taking nearest neighbors from FCC lattice."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    pts.append(np.array([i+dx, j+dy, k+dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_bcc(n):
    """Generate BCC-like cluster."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i+0.5, j+0.5, k+0.5]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_icosahedral(n):
    """Generate Mackay icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1,0,phi],[-1,0,phi],[1,0,-phi],[-1,0,-phi],
        [0,phi,1],[0,-phi,1],[0,phi,-1],[0,-phi,-1],
        [phi,1,0],[-phi,1,0],[phi,-1,0],[-phi,-1,0]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    radii = [1.0, 1.618, 2.236, 2.828, 3.317, 3.804, 4.236]
    for r in radii:
        for v in verts:
            pts.append(r * v)
        for i in range(12):
            for j in range(i+1, 12):
                mid = (verts[i] + verts[j]) / 2
                mid /= np.linalg.norm(mid)
                pts.append(r * mid)
    pts = np.array(pts)
    # Remove duplicates
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.01 for q in unique_pts):
            unique_pts.append(p)
    pts = np.array(unique_pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_decahedral(n):
    """Generate decahedral cluster (twinned FCC)."""
    pts = []
    # Simple decahedron: 5-fold symmetric
    for layer in range(5):
        r = 0.5 * (layer + 1)
        if layer == 0:
            pts.append(np.array([0.0, 0.0, 0.0]))
        else:
            n_per_layer = 5 * layer
            for i in range(n_per_layer):
                angle = 2 * np.pi * i / n_per_layer
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = 0.0
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_random_compact(n, rng, spacing=0.72):
    """Generate random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    if len(points) == n:
        return np.array(points)
    # Fallback: uniform random
    return rng.uniform(-radius, radius, (n, 3))


def _perturb(positions, rng, scale=0.1):
    return positions + rng.normal(0, scale, positions.shape)


def _basin_hop(positions, potential, kind, rng, max_iter=5, step_scale=0.15):
    """Simple basin hopping: perturb, relax, accept if lower energy."""
    energy, pos = sutton_chen_energy_gradient(positions, kind)
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        pert = _perturb(pos, rng, step_scale)
        if not _is_valid(pert):
            continue
        try:
            new_energy, new_pos = relax(pert, potential, kind)
            if np.isfinite(new_energy) and _is_valid(new_pos) and new_energy < energy:
                energy = new_energy
                pos = new_pos
        except Exception:
            pass
    return energy, pos


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345)
    best_energy = np.inf
    best_positions = None

    def _try_relax(start, use_bh=False):
        nonlocal best_energy, best_positions
        if _time_left() < 15.0:
            return
        try:
            if use_bh:
                energy, positions = _basin_hop(start, potential, kind, rng)
            else:
                energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Generate diverse starting points
    starts = []

    # Structured templates
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            starts.append(('fcc', fcc))
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            starts.append(('bcc', bcc))
    except Exception:
        pass

    try:
        ico = _make_icosahedral(n)
        if _is_valid(ico):
            starts.append(('ico', ico))
    except Exception:
        pass

    try:
        deca = _make_decahedral(n)
        if _is_valid(deca):
            starts.append(('deca', deca))
    except Exception:
        pass

    # Perturbed versions of structured templates
    for name, template in list(starts):
        for scale in [0.05, 0.1, 0.2]:
            pert = _perturb(template, rng, scale)
            if _is_valid(pert):
                starts.append((f'{name}_pert_{scale}', pert))

    # Random compact starts
    for _ in range(5):
        rc = _make_random_compact(n, rng)
        if _is_valid(rc):
            starts.append(('random', rc))

    # Phase 1: Simple relaxation of all starts
    for name, start in starts:
        if _time_left() < 20.0:
            break
        _try_relax(start, use_bh=False)

    # Phase 2: Basin hopping on best results and diverse starts
    if best_positions is not None and _time_left() > 30.0:
        _try_relax(best_positions, use_bh=True)

    # Phase 3: More basin hopping on diverse starts if time permits
    if _time_left() > 60.0:
        for name, start in starts[:10]:
            if _time_left() < 30.0:
                break
            _try_relax(start, use_bh=True)

    # Fallback
    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
