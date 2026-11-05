"""Dzugutov cluster global minima: parallel basin-hopping with polytetrahedral seeds."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# ---- Dzugutov potential (exact, vectorized) ----
_A, _B, _a, _b, _c, _d, _m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
_RMIN = (1.0 / _B) ** (1.0 / _m)  # equilibrium bond length ≈ 1.047

def _dzugutov_energy(pos):
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    rp = r[iu]
    e = np.zeros_like(rp)
    mask_a = rp < _a
    mask_b = (rp < _b) & ~mask_a
    e[mask_a] = _A * (rp[mask_a] ** (-_m) - _B) * np.exp(_c / (rp[mask_a] - _a))
    e[mask_b] = _B * np.exp(_d / (rp[mask_b] - _b))
    return float(np.sum(e))

def _valid(pos, n):
    if pos.shape != (n, 3) or not np.all(np.isfinite(pos)):
        return False
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(n, 1)
    return bool(np.all(r[iu] >= 0.5 - 1e-9))

# ---- Icosahedral / polytetrahedral geometry ----
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _icosahedron_vertices():
    v = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1],
    ], dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _icosahedron_edges(v):
    d = v[:, None, :] - v[None, :, :]
    r = np.linalg.norm(d, axis=2)
    iu = np.triu_indices(len(v), 1)
    edges = set()
    for i, j in zip(iu[0], iu[1]):
        if r[i, j] < 1.1 * r[iu[0], iu[1]].min():
            edges.add((min(i, j), max(i, j)))
    return edges

def _icosahedron_face_centers(v):
    edges = _icosahedron_edges(v)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in edges:
                for k in range(j + 1, 12):
                    if (j, k) in edges and (i, k) in edges:
                        faces.append((i, j, k))
    centers = []
    for (i, j, k) in faces:
        nrm = np.cross(v[j] - v[i], v[k] - v[i])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-15)
        centers.append(nrm)
    return np.array(centers)

def _build_mackay(n_target, rng, seed_scale=None):
    """Mackay icosahedral cluster: center + 12 (1st shell) + up to 42 (2nd shell)."""
    if seed_scale is None:
        seed_scale = _RMIN
    v1_unit = _icosahedron_vertices()
    # Scale 1st shell so edge = seed_scale
    d12 = np.linalg.norm(v1_unit[0] - v1_unit[1])
    v1 = v1_unit * (seed_scale / d12)
    center = np.zeros(3)
    # 2nd shell radius: for Mackay, r2 = r1 * (1 + 1/phi)
    a2 = seed_scale * (1.0 + 1.0 / _PHI)
    fc = _icosahedron_face_centers(v1_unit)  # 12 face normals (unit)
    edge_pts = []
    edges = _icosahedron_edges(v1_unit)
    for (i, j) in edges:
        mid = (v1_unit[i] + v1_unit[j])
        mid = mid / (np.linalg.norm(mid) + 1e-15)
        edge_pts.append(mid)
    shell2_unit = np.vstack([fc, np.array(edge_pts)])
    # dedupe
    shell2_unit = np.unique(np.round(shell2_unit, 8), axis=0)
    shell2 = shell2_unit * a2
    # Assemble
    parts = [center[None, :], v1, shell2]
    positions = np.vstack(parts)
    if positions.shape[0] > n_target:
        keep = n_target - 13
        if keep < 0:
            keep = 0
        positions = np.vstack([center[None, :], v1, shell2[:max(0, keep)]])
    # Add jitter for diversity
    if rng is not None and positions.shape[0] == n_target:
        positions = positions + rng.normal(0, 0.02, positions.shape)
    return positions

def _random_compact(n, rng, scale):
    radius = scale * n ** (1.0 / 3.0) * 1.0
    pts = []
    tries = 0
    while len(pts) < n and tries < 300 * n:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= 0.55 * scale for q in pts):
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)

def _relax_safe(pos, potential, parameter):
    """Wrapper around relax that returns (energy, positions) or (inf, pos) on failure."""
    try:
        e, p = relax(np.array(pos, dtype=float), potential, parameter)
        if np.all(np.isfinite(p)) and p.shape == pos.shape:
            return float(e), np.array(p, dtype=float)
    except Exception:
        pass
    return np.inf, np.array(pos, dtype=float)

def _basin_hopping(pos, n, potential, parameter, rng, max_iter, deadline):
    """Softness-annealed basin hopping: perturb, relax, accept/reject with Metropolis-like rule."""
    best_pos = np.array(pos, dtype=float)
    best_e = _dzugutov_energy(best_pos)
    cur_pos = np.array(best_pos, dtype=float)
    cur_e = best_e
    t0 = time.time()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Annealed perturbation: start large, shrink
        frac = it / max(1, max_iter)
        sigma = 0.15 * (1.0 - 0.5 * frac)
        pert = cur_pos + rng.normal(0, sigma, cur_pos.shape)
        if not _valid(pert, n):
            # Try smaller perturbation
            pert = cur_pos + rng.normal(0, sigma * 0.3, cur_pos.shape)
            if not _valid(pert, n):
                continue
        e_new, p_new = _relax_safe(pert, potential, parameter)
        if not np.isfinite(e_new):
            continue
        # Accept if better, or with probability exp(-(e_new - cur_e)/T) where T anneals
        T = 0.5 * (1.0 - frac) + 0.01
        if e_new < cur_e or rng.random() < np.exp(-(e_new - cur_e) / T):
            cur_pos, cur_e = p_new, e_new
            if e_new < best_e:
                best_e, best_pos = e_new, p_new.copy()
    return best_pos, best_e

def construct(instance):
    size = int(instance["n"])
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    t_start = time.time()
    DEADLINE = 320.0  # safety margin under 330s kill
    rng = np.random.default_rng(size * 7919 + 13)

    # ---- Generate diverse seeds ----
    seeds = []
    # Mackay icosahedral (polytetrahedral) — primary topology
    for trial in range(3):
        try:
            s = _build_mackay(size, rng, seed_scale=_RMIN * (0.95 + 0.1 * trial))
            if s.shape[0] == size:
                seeds.append(s)
        except Exception:
            pass
    # Random compact starts
    for trial in range(8):
        s = _random_compact(size, rng, _RMIN * (0.9 + 0.2 * rng.random()))
        seeds.append(s)
    # Spherical shell starts (quasicrystal-like)
    for trial in range(3):
        theta = rng.uniform(0, np.pi, size)
        phi = rng.uniform(0, 2 * np.pi, size)
        R = _RMIN * size ** (1.0 / 3.0) * (0.7 + 0.3 * trial)
        s = np.column_stack([
            R * np.sin(theta) * np.cos(phi),
            R * np.sin(theta) * np.sin(phi),
            R * np.cos(theta),
        ])
        seeds.append(s)

    # ---- Parallel relaxation of all seeds ----
    N_WORKERS = min(4, len(seeds))
    results = []  # list of (energy, positions)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_relax_safe, s, potential, parameter): i for i, s in enumerate(seeds)}
        for fut in as_completed(futures):
            if time.time() > DEADLINE - 10.0:
                break
            e, p = fut.result()
            if np.isfinite(e) and _valid(p, size):
                results.append((e, p))

    if not results:
        # Fallback
        ang = 2 * np.pi * np.arange(size) / size
        R = max(1.0, 0.5 * size / (2 * np.pi) * 1.1)
        ring = np.column_stack([R * np.cos(ang), R * np.sin(ang), np.zeros(size)])
        return ring

    results.sort(key=lambda x: x[0])
    best_pos = results[0][1].copy()
    best_e = results[0][0]

    # ---- Basin hopping on top basins (sequential, deadline-guarded) ----
    top_k = min(5, len(results))
    for idx in range(top_k):
        if time.time() > DEADLINE - 15.0:
            break
        remaining = DEADLINE - time.time()
        # Allocate time: more iterations for earlier basins
        max_iter = max(5, int(remaining * 0.6 / (top_k - idx)))
        bh_pos, bh_e = _basin_hopping(results[idx][1], size, potential, parameter, rng, max_iter, DEADLINE - 10.0)
        if bh_e < best_e:
            best_e, best_pos = bh_e, bh_pos.copy()

    # ---- Final polish: relax the best once more ----
    if time.time() < DEADLINE - 5.0:
        e_final, p_final = _relax_safe(best_pos, potential, parameter)
        if np.isfinite(e_final) and _valid(p_final, size) and e_final < best_e:
            best_e, best_pos = e_final, p_final

    # ---- Fallback guarantee ----
    if not _valid(best_pos, size):
        ang = 2 * np.pi * np.arange(size) / size
        R = max(1.0, 0.5 * size / (2 * np.pi) * 1.1)
        best_pos = np.column_stack([R * np.cos(ang), R * np.sin(ang), np.zeros(size)])

    return best_pos
# EVOLVE-BLOCK-END
