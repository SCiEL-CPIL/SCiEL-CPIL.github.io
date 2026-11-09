"""Structure-based global search: exact symmetric prototypes + L-BFGS-B relaxation with basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}

_DEADLINE_FRAC = 0.90
_NTHREADS = 4


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites) * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                sites.append(base + np.array([0, 0, z]))
                sites.append(base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2]))
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _bcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    """Mackay icosahedral cluster: 1 + 12 + 30 + 20 + ... shells."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    # Shell 1: 12 vertices at distance r1
    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    # Shell 2: 30 edge midpoints (face centers of icosahedron)
    # The icosahedron has 30 edges; midpoints of edges give 30 points
    r2 = spacing * np.sqrt(3.0)
    edges_found = set()
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 1.0 < d < 2.0:  # adjacent vertices
                key = tuple(sorted([i, j]))
                if key not in edges_found:
                    edges_found.add(key)
                    mid = (ico_verts[i] + ico_verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    # Shell 3: 20 face centers
    r3 = spacing * np.sqrt(5.0)
    # Find faces (triangles) of the icosahedron
    faces = []
    adj = {}
    for i in range(12):
        adj[i] = []
        for j in range(12):
            if i != j:
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if 1.0 < d < 2.0:
                    adj[i].append(j)
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k > i and k != j:
                    d_ik = np.linalg.norm(ico_verts[i] - ico_verts[k])
                    if 1.0 < d_ik < 2.0:
                        faces.append((i, j, k))
    # Deduplicate faces
    unique_faces = set()
    for f in faces:
        unique_faces.add(tuple(sorted(f)))
    for f in unique_faces:
        center = (ico_verts[f[0]] + ico_verts[f[1]] + ico_verts[f[2]]) / 3.0
        center = center / np.linalg.norm(center)
        points.append(center * r3)
    if len(points) >= n:
        return np.array(points[:n])

    # Additional shells: add more points on a sphere
    remaining = n - len(points)
    if remaining > 0:
        r4 = spacing * np.sqrt(7.0)
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(max(0, 1 - z * z))
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r4)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42, min_dist_frac=0.7):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_d = min_dist_frac * spacing
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempt += 1
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
        else:
            base = np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _jitter(pos, rng, sigma):
    return pos + rng.normal(0, sigma, pos.shape)


def _relax_sc(pos, kind, timeout=10.0, maxiter=500):
    """Relax positions using L-BFGS-B on Sutton-Chen potential."""
    start = time.time()
    try:
        if not _validate_positions(pos):
            return np.inf, None

        n = pos.shape[0]

        def obj(x):
            if time.time() - start > timeout:
                raise TimeoutError()
            e, g = sutton_chen_energy_gradient(x.reshape(n, 3), kind)
            return e, g.flatten()

        res = minimize(
            fun=obj,
            x0=pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6,
                     'maxfun': 1000, 'maxls': 30, 'maxcor': 15}
        )

        if np.isfinite(res.fun):
            relaxed = res.x.reshape(n, 3)
            if _validate_positions(relaxed):
                return float(res.fun), relaxed
    except TimeoutError:
        pass
    except Exception:
        pass
    return np.inf, None


def _basin_hopping(pos, kind, rng, n_iter=30, timeout_per=3.0, maxiter=200):
    """Simple basin hopping: jitter + relax, accept if lower energy."""
    best_e, best_p = _relax_sc(pos, kind, timeout=timeout_per, maxiter=maxiter)
    if best_p is None:
        return best_e, best_p

    for _ in range(n_iter):
        if time.time() > time.time():  # placeholder
            break
        sigma = 0.1 + 0.2 * rng.random()
        jittered = _jitter(best_p, rng, sigma)
        if not _validate_positions(jittered):
            continue
        e, p = _relax_sc(jittered, kind, timeout=timeout_per, maxiter=maxiter)
        if p is not None and e < best_e:
            best_e, best_p = e, p
    return best_e, best_p


def _generate_prototypes(n, kind):
    """Generate a diverse set of physically-motivated structural prototypes."""
    spacing = _SPACING.get(kind, 0.72)
    prototypes = []

    # Multiple spacings for each crystal structure
    for sp_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        sp = spacing * sp_factor
        prototypes.append(_fcc_cluster(n, sp))
        prototypes.append(_hcp_cluster(n, sp))
        prototypes.append(_bcc_cluster(n, sp))
        prototypes.append(_icosahedral_cluster(n, sp))

    # Random compact seeds with various parameters
    for seed in range(1, 16):
        for md in [0.6, 0.7, 0.8, 0.9]:
            pos = _random_compact(n, spacing, seed=seed * 1000 + int(md * 100), min_dist_frac=md)
            if _validate_positions(pos):
                prototypes.append(pos)

    # Jittered versions of key prototypes
    rng = np.random.default_rng(12345)
    for base in prototypes[:12]:
        for sigma in [0.05, 0.1, 0.15, 0.25]:
            jittered = _jitter(base, rng, sigma)
            if _validate_positions(jittered):
                prototypes.append(jittered)

    return prototypes


def construct(instance):
    start_time = time.time()
    deadline = start_time + 330.0 * _DEADLINE_FRAC

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))

    # Generate diverse prototypes
    prototypes = _generate_prototypes(n, kind)

    # Filter valid prototypes
    valid_protos = [p for p in prototypes if _validate_positions(p)]

    # Shuffle for diversity in parallel execution
    rng = np.random.default_rng(42)
    rng.shuffle(valid_protos)

    best_energy = np.inf
    best_positions = None

    n_protos = len(valid_protos)
    if n_protos == 0:
        # Fallback: single random seed
        pos = _random_compact(n, _SPACING.get(kind, 0.72), seed=999)
        if _validate_positions(pos):
            best_positions = pos
        return best_positions

    # Phase 1: Parallel relaxation of all prototypes
    # Budget: use ~70% of time for parallel relaxation, 30% for basin hopping on best
    phase1_budget = (deadline - start_time) * 0.70
    per_relax = max(1.0, phase1_budget / max(n_protos, 1))
    per_relax = min(per_relax, 5.0)  # cap individual relaxation

    def worker(proto):
        e, p = _relax_sc(proto, kind, timeout=per_relax, maxiter=400)
        return e, p

    results = []
    with ThreadPoolExecutor(max_workers=_NTHREADS) as executor:
        futures = {executor.submit(worker, p): i for i, p in enumerate(valid_protos)}
        for future in as_completed(futures):
            if time.time() > deadline - 30:  # leave time for phase 2
                break
            try:
                e, p = future.result(timeout=1)
                if p is not None and np.isfinite(e) and _validate_positions(p):
                    results.append((e, p))
                    if e < best_energy:
                        best_energy = e
                        best_positions = p.copy()
            except Exception:
                pass

    # Sort results by energy
    results.sort(key=lambda x: x[0])

    # Phase 2: Basin hopping on the best results
    phase2_budget = deadline - time.time() - 10  # leave 10s safety margin
    if best_positions is not None and phase2_budget > 5:
        # Take top K results for basin hopping
        top_k = min(5, len(results))
        bh_budget_each = phase2_budget / max(top_k, 1)
        bh_budget_each = max(2.0, min(bh_budget_each, 20.0))

        for e0, p0 in results[:top_k]:
            if time.time() > deadline - 10:
                break
            e_bh, p_bh = _basin_hopping(
                p0, kind, rng,
                n_iter=int(bh_budget_each / 3.0),
                timeout_per=min(3.0, bh_budget_each / 3.0),
                maxiter=150
            )
            if p_bh is not None and e_bh < best_energy:
                best_energy = e_bh
                best_positions = p_bh.copy()

    # Phase 3: Final polish on the best
    remaining = deadline - time.time() - 5
    if best_positions is not None and remaining > 5:
        e_final, p_final = _relax_sc(best_positions, kind, timeout=min(remaining, 15), maxiter=800)
        if p_final is not None and np.isfinite(e_final) and _validate_positions(p_final) and e_final < best_energy:
            best_energy = e_final
            best_positions = p_final.copy()

    # Fallback
    if best_positions is None:
        pos = _random_compact(n, _SPACING.get(kind, 0.72), seed=12345)
        e, best_positions = _relax_sc(pos, kind, timeout=10, maxiter=300)
        if best_positions is None:
            best_positions = pos

    # Final safety check
    if not _validate_positions(best_positions):
        best_positions = _random_compact(n, _SPACING.get(kind, 0.72), seed=98765)

    return best_positions
# EVOLVE-BLOCK-END
