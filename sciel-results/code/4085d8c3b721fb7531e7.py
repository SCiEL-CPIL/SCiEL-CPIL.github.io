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
    if len(pts) <= n:
        # pad if needed
        while len(pts) < n:
            p = pts[np.random.default_rng(len(pts)).integers(0, len(pts))]
            p = p + np.random.default_rng(len(pts) * 7).normal(0, 0.1, 3)
            pts.append(p)
        return np.array(pts[:n])
    if center is None:
        center = np.mean(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _fcc_cluster(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
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
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _icosahedral_shells(n, r1=1.0, r2=1.27, r3=1.51):
    """Mackay-like icosahedral shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    # shell 1: 12 vertices
    for v in verts:
        pts.append(v * r1)
    # shell 2: 30 edge midpoints
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * r2
                pts.append(mid)
    # shell 3: 20 face centers
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * r3
                    pts.append(fc)
    # shell 4: more points on faces if needed
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            # random point on icosahedron surface, pushed to r3.5
            v = verts[rng.integers(0, len(verts))]
            v2 = verts[rng.integers(0, len(verts))]
            p = (v + v2) / 2
            p = p / np.linalg.norm(p) * (r3 + 0.3)
            pts.append(p)
    return _truncate_by_radius(np.array(pts), n)


def _random_compact(n, seed, min_dist=0.75):
    rng = np.random.default_rng(seed)
    max_radius = 0.85 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 500:
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


def _decahedral_cluster(n):
    """Bi-icosahedral / decahedral structure: two icosahedra sharing a face."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    # Two icosahedra offset along z
    r = 1.0
    pts = []
    # First icosahedron centered at (0,0,-0.6)
    for v in verts:
        pts.append(np.array([v[0], v[1], v[2] - 0.6]) * r)
    pts.append(np.array([0, 0, -0.6]))
    # Second icosahedron centered at (0,0,0.6)
    for v in verts:
        pts.append(np.array([v[0], v[1], v[2] + 0.6]) * r)
    pts.append(np.array([0, 0, 0.6]))
    # Add interstitial points
    rng = np.random.default_rng(11)
    while len(pts) < n:
        p = rng.normal(0, 1.5, 3)
        pts.append(p)
    return _truncate_by_radius(np.array(pts), n)


def _truncated_fcc(n, lattice=0.72):
    """FCC with corners cut (more spherical)."""
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    p = base + np.array(off)
                    # Spherical cutoff
                    if np.linalg.norm(p) < 0.85 * n ** (1.0 / 3.0) * 0.9:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        # fill with more FCC points
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                for k in range(-size, size + 1):
                    base = np.array([i, j, k]) * lattice
                    for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                                [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                        pts = np.vstack([pts, base + np.array(off)])
                        if len(pts) >= n:
                            break
                    if len(pts) >= n:
                        break
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
    return _truncate_by_radius(pts, n)


def _random_fcc_perturbed(n, seed, lattice=0.72, pert=0.15):
    """FCC with random perturbations."""
    base = _fcc_cluster(n, lattice)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, pert, size=(n, 3))


def _random_hcp_perturbed(n, seed, lattice=0.72, pert=0.15):
    """HCP with random perturbations."""
    base = _hcp_cluster(n, lattice)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, pert, size=(n, 3))


def construct(instance):
    start_time = time.time()
    deadline = start_time + 310.0  # safety margin vs 330s kill

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))

    # --- Generate diverse structural seeds ---
    seeds = []
    
    # Structured seeds
    seeds.append(_fcc_cluster(n, lattice=0.72))
    seeds.append(_fcc_cluster(n, lattice=0.70))
    seeds.append(_fcc_cluster(n, lattice=0.74))
    seeds.append(_hcp_cluster(n, lattice=0.72))
    seeds.append(_hcp_cluster(n, lattice=0.70))
    seeds.append(_icosahedral_shells(n))
    seeds.append(_icosahedral_shells(n, r1=0.95, r2=1.22, r3=1.45))
    seeds.append(_decahedral_cluster(n))
    seeds.append(_truncated_fcc(n, lattice=0.72))
    seeds.append(_truncated_fcc(n, lattice=0.70))
    
    # Perturbed structured seeds
    for s in range(5):
        seeds.append(_random_fcc_perturbed(n, seed=100 + s, pert=0.1 + s * 0.05))
        seeds.append(_random_hcp_perturbed(n, seed=200 + s, pert=0.1 + s * 0.05))
    
    # Random compact seeds
    for s in range(8):
        seeds.append(_random_compact(n, seed=500 + s, min_dist=0.7 + s * 0.05))
    
    # --- Relax all seeds, keep best ---
    best_e = np.inf
    best_pos = None
    
    for idx, s in enumerate(seeds):
        if time.time() > deadline:
            break
        if not _validate_positions(s):
            continue
        try:
            e, p = relax(s, potential, kind)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_e:
                    best_e = e
                    best_pos = p.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback
        pos = _random_compact(n, seed=999, min_dist=0.8)
        try:
            best_e, best_pos = relax(pos, potential, kind)
        except Exception:
            best_pos = pos
        if not _validate_positions(best_pos):
            return _random_compact(n, seed=12345, min_dist=0.9)
    
    # --- Use remaining time for additional random restarts ---
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Estimate time per relaxation from elapsed
        elapsed = time.time() - start_time
        n_done = len(seeds)
        if n_done > 0:
            t_per = elapsed / n_done
        else:
            t_per = 5.0
        
        # How many more can we do?
        n_more = int(remaining / max(t_per, 1.0)) - 2  # leave 2s buffer
        
        rng = np.random.default_rng(777)
        for i in range(max(n_more, 0)):
            if time.time() > deadline - 2.0:
                break
            
            # Mix of random compact and perturbed best
            if i % 3 == 0:
                s = _random_compact(n, seed=2000 + i, min_dist=0.75)
            elif i % 3 == 1:
                s = best_pos + rng.normal(0, 0.3, size=(n, 3))
            else:
                s = best_pos + rng.normal(0, 0.5, size=(n, 3))
            
            if not _validate_positions(s):
                continue
            try:
                e, p = relax(s, potential, kind)
                if np.isfinite(e) and _validate_positions(p):
                    if e < best_e:
                        best_e = e
                        best_pos = p.copy()
            except Exception:
                pass
    
    # --- Final polish ---
    if time.time() < deadline - 5.0:
        try:
            e_final, p_final = relax(best_pos, potential, kind)
            if np.isfinite(e_final) and _validate_positions(p_final):
                best_e = e_final
                best_pos = p_final
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
