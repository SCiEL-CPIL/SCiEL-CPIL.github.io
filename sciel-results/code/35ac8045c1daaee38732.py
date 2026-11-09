"""Parallel multi-start L-BFGS-B on true Sutton-Chen potential with diverse seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_MAX_N = 80
_DEADLINE_FRAC = 0.92
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

    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 1.0 < d < 2.0:
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

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


def _relax_with_timeout(pos, potential, kind, timeout):
    """Run L-BFGS-B on the true potential with a wall-clock timeout."""
    start = time.time()
    try:
        if not _validate_positions(pos):
            return np.inf, None

        n = pos.shape[0]

        def obj(x):
            if time.time() - start > timeout:
                raise TimeoutError()
            return sutton_chen_energy_gradient(x.reshape(n, 3), kind)

        from scipy.optimize import minimize
        res = minimize(
            fun=obj,
            x0=pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 600,
                     'maxls': 20, 'maxcor': 10}
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


def _generate_seed_list(n, kind, rng):
    """Generate a diverse list of seed positions."""
    spacing = _SPACING.get(kind, 0.72)
    seeds = []

    # Crystallographic seeds at multiple spacings
    for sp in [spacing * 0.85, spacing, spacing * 1.1, spacing * 1.25]:
        seeds.append(_fcc_cluster(n, sp))
        seeds.append(_hcp_cluster(n, sp))
        seeds.append(_bcc_cluster(n, sp))
        seeds.append(_icosahedral_cluster(n, sp))

    # Random compact seeds
    for seed in range(1, 13):
        for md in [0.65, 0.75, 0.85]:
            pos = _random_compact(n, spacing, seed=seed * 100 + int(md * 100), min_dist_frac=md)
            if _validate_positions(pos):
                seeds.append(pos)

    return seeds


def construct(instance):
    start_time = time.time()
    deadline = start_time + _MAX_N * _DEADLINE_FRAC * 0.0  # placeholder, set below
    # Compute deadline based on n
    # We have 330s total; use 92% as safety
    deadline = start_time + 330.0 * _DEADLINE_FRAC

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)

    rng = np.random.default_rng(42)

    # Generate seeds
    seeds = _generate_seed_list(n, kind, rng)

    # Filter valid seeds
    valid_seeds = [s for s in seeds if _validate_positions(s)]

    # Add jittered copies of the best crystallographic seeds for diversity
    for base in valid_seeds[:8]:
        for sigma in [0.1, 0.2, 0.3]:
            jittered = _jitter(base, rng, sigma)
            if _validate_positions(jittered):
                valid_seeds.append(jittered)

    # Shuffle to mix seed types
    rng.shuffle(valid_seeds)

    # Parallel multi-start relaxation
    best_energy = np.inf
    best_positions = None

    # Time per relaxation: budget / n_seeds, but cap at reasonable values
    n_seeds = len(valid_seeds)
    total_budget = deadline - time.time()
    per_relax = max(1.0, total_budget / max(n_seeds, 1))
    per_relax = min(per_relax, 8.0)  # cap individual relaxation time

    # Use thread pool
    results = []

    def worker(seed_pos):
        e, p = _relax_with_timeout(seed_pos, potential, kind, per_relax)
        return e, p

    with ThreadPoolExecutor(max_workers=_NTHREADS) as executor:
        futures = {executor.submit(worker, s): i for i, s in enumerate(valid_seeds)}
        for future in as_completed(futures):
            if time.time() > deadline - 5:
                break
            try:
                e, p = future.result(timeout=1)
                if p is not None and np.isfinite(e) and _validate_positions(p):
                    if e < best_energy:
                        best_energy = e
                        best_positions = p.copy()
            except Exception:
                pass

    # If we have time left, do a second pass with tighter relaxation on the best found
    remaining = deadline - time.time()
    if best_positions is not None and remaining > 10:
        # Polish the best with a longer relaxation
        e2, p2 = _relax_with_timeout(best_positions, potential, kind, min(remaining - 5, 15))
        if p2 is not None and np.isfinite(e2) and _validate_positions(p2) and e2 < best_energy:
            best_energy = e2
            best_positions = p2.copy()

        # Also try jittering the best and relaxing
        for sigma in [0.05, 0.1, 0.15]:
            if time.time() > deadline - 5:
                break
            jittered = _jitter(best_positions, rng, sigma)
            if _validate_positions(jittered):
                e3, p3 = _relax_with_timeout(jittered, potential, kind, min(remaining - 5, 10))
                if p3 is not None and np.isfinite(e3) and _validate_positions(p3) and e3 < best_energy:
                    best_energy = e3
                    best_positions = p3.copy()

    # Fallback
    if best_positions is None:
        pos = _random_compact(n, spacing, seed=999)
        e, best_positions = _relax_with_timeout(pos, potential, kind, 10)
        if best_positions is None:
            best_positions = pos

    # Final safety check
    if not _validate_positions(best_positions):
        best_positions = _random_compact(n, spacing, seed=12345)

    return best_positions
# EVOLVE-BLOCK-END
