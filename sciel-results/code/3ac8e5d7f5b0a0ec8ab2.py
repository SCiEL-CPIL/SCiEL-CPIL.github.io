"""Sutton-Chen cluster global minima via rich multi-start local minimization
with domain-informed structural seeds (no stochastic basin hopping)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True


def _truncate_by_radius(pts, n, center=None):
    """Take n points closest to center (default centroid)."""
    if len(pts) < n:
        # pad if needed
        while len(pts) < n:
            p = pts[np.random.default_rng(len(pts)).integers(0, len(pts))]
            p = p + np.random.default_rng(len(pts) * 7).normal(0, 0.1, 3)
            pts = np.vstack([pts, p])
    if center is None:
        center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _fcc_cluster(n, lattice=0.8):
    """Compact FCC-like cluster: take n nearest FCC lattice points to origin."""
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    return _truncate_by_radius(pts, n)


def _hcp_cluster(n, lattice=0.8):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    return _truncate_by_radius(pts, n)


def _icosahedral_shells(n):
    """Icosahedral cluster built from concentric shells at physically consistent radii."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    # shell 1: 12 vertices at radius 1.0
    for v in verts:
        pts.append(v.copy())
    # shell 2: 30 edge midpoints, pushed to radius ~1.7 (consistent with SC spacing)
    edge_mids = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.7
                edge_mids.append(mid)
    pts.extend(edge_mids)
    # shell 3: 20 face centers at radius ~2.3
    face_centers = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 2.3
                    face_centers.append(fc)
    pts.extend(face_centers)
    pts = np.array(pts)
    if len(pts) >= n:
        return _truncate_by_radius(pts, n)
    # pad with points on a larger shell
    rng = np.random.default_rng(44)
    while len(pts) < n:
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p) * 2.8
        pts = np.vstack([pts, p])
    return _truncate_by_radius(pts, n)


def _decahedral_cluster(n):
    """Truncated icosahedron (decahedral) approximation via FCC 5-fold twinning."""
    # Approximate by taking FCC points and keeping only those in a 5-twin geometry
    # Simplified: use FCC but with slightly larger lattice to encourage 5-fold symmetry
    return _fcc_cluster(n, lattice=0.85)


def _truncated_fcc(n, lattice=0.8):
    """FCC with a slightly different truncation to give a distinct surface."""
    pts = _fcc_cluster(n + 5, lattice=lattice)
    return _truncate_by_radius(pts, n)


def _random_compact(n, seed, min_dist=0.8):
    rng = np.random.default_rng(seed)
    max_radius = 0.9 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 400:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            if pts:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) < min_dist:
                    ok = False
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        pts.append(p)
    return np.array(pts)


def _random_fcc_perturbed(n, seed, lattice=0.8, amp=0.15):
    base = _fcc_cluster(n, lattice=lattice)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, amp, size=(n, 3))


def _random_hcp_perturbed(n, seed, lattice=0.8, amp=0.15):
    base = _hcp_cluster(n, lattice=lattice)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, amp, size=(n, 3))


def construct(instance):
    start_time = time.time()
    deadline = start_time + 315.0  # safety margin vs 330s kill

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))

    # Build a diverse library of physically-motivated prototypes
    seeds = [
        _fcc_cluster(n, lattice=0.8),
        _fcc_cluster(n, lattice=0.75),
        _hcp_cluster(n, lattice=0.8),
        _hcp_cluster(n, lattice=0.75),
        _icosahedral_shells(n),
        _decahedral_cluster(n),
        _truncated_fcc(n, lattice=0.8),
        _random_compact(n, seed=7, min_dist=0.8),
        _random_compact(n, seed=13, min_dist=0.9),
        _random_compact(n, seed=23, min_dist=0.85),
        _random_fcc_perturbed(n, seed=101, lattice=0.8, amp=0.15),
        _random_fcc_perturbed(n, seed=202, lattice=0.78, amp=0.12),
        _random_hcp_perturbed(n, seed=303, lattice=0.8, amp=0.15),
        _random_hcp_perturbed(n, seed=404, lattice=0.78, amp=0.12),
    ]

    best_e = np.inf
    best_pos = None

    # Phase 1: relax each seed independently; keep the global best
    for s in seeds:
        if time.time() > deadline:
            break
        if not _validate_positions(s):
            continue
        try:
            e, p = relax(s, potential, kind)
            if np.isfinite(e) and _validate_positions(p) and e < best_e:
                best_e = e
                best_pos = p.copy()
        except Exception:
            pass

    if best_pos is None:
        # fallback
        pos = _random_compact(n, seed=99, min_dist=0.8)
        try:
            best_e, best_pos = relax(pos, potential, kind)
        except Exception:
            best_e = np.inf
        if best_e == np.inf:
            return pos

    # Phase 2: a few extra perturbed restarts from the current best to nudge it lower
    rng = np.random.default_rng(777)
    for amp in [0.2, 0.1, 0.05]:
        if time.time() > deadline - 20.0:
            break
        for _ in range(3):
            if time.time() > deadline - 15.0:
                break
            perturbed = best_pos + rng.normal(0, amp, size=(n, 3))
            if not _validate_positions(perturbed):
                continue
            try:
                e_new, p_new = relax(perturbed, potential, kind)
                if np.isfinite(e_new) and _validate_positions(p_new) and e_new < best_e:
                    best_e = e_new
                    best_pos = p_new.copy()
            except Exception:
                pass

    # Phase 3: final full polish on the best configuration
    if time.time() < deadline - 5.0:
        try:
            e_final, p_final = relax(best_pos, potential, kind)
            if np.isfinite(e_final) and _validate_positions(p_final) and e_final < best_e:
                best_e = e_final
                best_pos = p_final
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
