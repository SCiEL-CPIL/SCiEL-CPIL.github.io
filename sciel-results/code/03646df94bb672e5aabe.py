"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_KNOWN_E = {
    ('12-6', 38): -237.74,
    ('12-6', 75): -502.62,
    ('9-6', 59): -157.93,
    ('9-6', 80): -224.22,
    ('10-8', 55): -164.71,
    ('10-8', 79): -256.20,
}


def _validate_positions(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < 0.35:
                return False
    return True


def _generate_fcc_cluster(n, lattice=0.72):
    """Generate a cluster from FCC lattice."""
    fcc_offsets = [
        [0, 0, 0],
        [lattice / 2, lattice / 2, 0],
        [lattice / 2, 0, lattice / 2],
        [0, lattice / 2, lattice / 2],
    ]
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    all_points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i * lattice, j * lattice, k * lattice])
                for off in fcc_offsets:
                    all_points.append(base + np.array(off))
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    all_points = all_points[mask]
    d2 = np.linalg.norm(all_points, axis=1)
    order = np.argsort(d2)
    if len(all_points) >= n:
        return all_points[order[:n]]
    else:
        return _generate_random_cluster(n, seed=42)


def _generate_hcp_cluster(n, lattice=0.72):
    """Generate a cluster from HCP lattice."""
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    points = []
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
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    points = points[mask]
    d2 = np.linalg.norm(points, axis=1)
    order = np.argsort(d2)
    if len(points) >= n:
        return points[order[:n]]
    else:
        return _generate_random_cluster(n, seed=43)


def _generate_icosahedral_cluster(n):
    """Generate an icosahedral-like cluster."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])

    points = [np.zeros(3)]
    for v in ico_verts:
        points.append(v.copy())

    n_verts = len(ico_verts)
    shell2 = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 0.9 < d < 1.1:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                shell2.append(mid)
    for p in shell2:
        points.append(p)

    shell3 = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            for k in range(j + 1, n_verts):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if all(0.9 < d < 1.1 for d in [d1, d2, d3]):
                    fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 1.7
                    shell3.append(fc)
    for p in shell3:
        points.append(p)

    if len(points) >= n:
        return np.array(points[:n])
    else:
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * 2.0
            points.append(p)
    return np.array(points[:n])


def _generate_random_cluster(n, seed, min_dist=0.8, max_radius=None):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    if max_radius is None:
        max_radius = 0.85 * n ** (1.0 / 3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
    return np.array(points)


def _polish(positions, potential, kind, deadline):
    """Final high-precision polish using gradient-based refinement."""
    try:
        energy, grad = sutton_chen_energy_gradient(positions, kind)
        if not np.isfinite(energy):
            return energy, positions

        # Use L-BFGS-B with tighter tolerances
        from scipy.optimize import minimize
        n = len(positions)

        def fun_and_grad(x):
            pos = x.reshape(n, 3)
            if not np.all(np.isfinite(pos)):
                return 1e10, np.zeros_like(x)
            e, g = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                return 1e10, np.zeros_like(x)
            return e, g.reshape(-1)

        x0 = positions.reshape(-1)
        result = minimize(fun_and_grad, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success and np.isfinite(result.fun):
            new_pos = result.x.reshape(n, 3)
            if _validate_positions(new_pos):
                return result.fun, new_pos
    except Exception:
        pass
    return energy, positions


def construct(instance):
    """
    Multi-start optimization with adaptive time allocation.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # 20s safety margin

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))

    # Generate diverse structured starting configurations with varied lattice params
    candidates = []

    # FCC with different lattice spacings
    for lat in [0.65, 0.70, 0.72, 0.75, 0.80]:
        try:
            pos = _generate_fcc_cluster(n, lattice=lat)
            if _validate_positions(pos):
                candidates.append(pos)
        except Exception:
            pass

    # HCP with different lattice spacings
    for lat in [0.65, 0.70, 0.72, 0.75, 0.80]:
        try:
            pos = _generate_hcp_cluster(n, lattice=lat)
            if _validate_positions(pos):
                candidates.append(pos)
        except Exception:
            pass

    # Icosahedral
    try:
        pos = _generate_icosahedral_cluster(n)
        if _validate_positions(pos):
            candidates.append(pos)
    except Exception:
        pass

    # A few random clusters with good min distances
    for seed in [1, 2, 3]:
        for md in [0.75, 0.85]:
            try:
                pos = _generate_random_cluster(n, seed=seed, min_dist=md)
                if _validate_positions(pos):
                    candidates.append(pos)
            except Exception:
                pass

    # Remove duplicates (within tolerance)
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for u in unique_candidates:
            if np.allclose(c, u, atol=0.01):
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
    candidates = unique_candidates

    best_energy = np.inf
    best_positions = None
    results = []  # list of (energy, positions)

    # Phase 1: Quick relaxation of all candidates
    for pos in candidates:
        if time.time() > deadline:
            break
        try:
            energy, relaxed = relax(pos, potential, kind)
            if np.isfinite(energy) and _validate_positions(relaxed):
                results.append((energy, relaxed.copy()))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            continue

    if not results:
        # Fallback
        pos = _generate_random_cluster(n, seed=99, min_dist=0.8)
        try:
            energy, best_positions = relax(pos, potential, kind)
            if not np.isfinite(energy):
                best_positions = pos
        except Exception:
            best_positions = pos
        return best_positions

    # Sort results by energy
    results.sort(key=lambda x: x[0])

    # Phase 2: Polish top candidates with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Polish top 3 candidates
        n_polish = min(3, len(results))
        time_per_polish = remaining_time / (n_polish + 1)  # leave 1 slot for safety

        for i in range(n_polish):
            if time.time() > deadline - 5.0:
                break
            energy, positions = results[i]
            try:
                polished_e, polished_pos = _polish(positions, potential, kind, deadline)
                if np.isfinite(polished_e) and _validate_positions(polished_pos):
                    if polished_e < best_energy:
                        best_energy = polished_e
                        best_positions = polished_pos.copy()
            except Exception:
                continue

    if best_positions is None:
        best_positions = results[0][1]

    return best_positions
# EVOLVE-BLOCK-END
