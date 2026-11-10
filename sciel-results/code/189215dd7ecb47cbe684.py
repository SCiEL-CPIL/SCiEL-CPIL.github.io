"""Morse cluster global minima via basin hopping over structured crystalline seeds."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _morse_energy(positions, rho):
    """Vectorized Morse total energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    pe = ex * (ex - 2.0)
    return float(np.sum(pe) / 2.0)


def _morse_energy_and_grad(positions, rho):
    """Vectorized Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    pe = ex * (ex - 2.0)
    energy = float(np.sum(pe) / 2.0)
    # d(pe)/d(dist) = -rho * ex * (ex - 1)
    dpe_dd = -rho * ex * (ex - 1.0)
    # d(dist)/d(pos_i) = (pos_i - pos_j) / dist  (for i != j)
    inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
    grad = -np.sum(dpe_dd[:, :, None] * diff * inv_dist[:, :, None], axis=1)
    return energy, grad


def _is_valid(positions, min_dist=0.5):
    """Check all pairwise distances >= min_dist."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist)) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    pts = []
    # FCC lattice points in a cube, sorted by distance from center
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                p = np.array([i, j, k]) * (spacing / np.sqrt(2))
                candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    seen = set()
    for _, p in candidates:
        key = tuple(p.round(6))
        if key not in seen:
            seen.add(key)
            pts.append(p)
        if len(pts) == n:
            break
    if len(pts) < n:
        # Fallback: add random points
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-3, 3, 3)
            pts.append(p)
    return np.array(pts)


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    pts = []
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                p = np.array([i, j, k]) * (spacing / np.sqrt(3))
                candidates.append((np.linalg.norm(p), p))
                # BCC center
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * (spacing / np.sqrt(3))
                candidates.append((np.linalg.norm(pc), pc))
    candidates.sort(key=lambda x: x[0])
    seen = set()
    for _, p in candidates:
        key = tuple(p.round(6))
        if key not in seen:
            seen.add(key)
            pts.append(p)
        if len(pts) == n:
            break
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-3, 3, 3)
            pts.append(p)
    return np.array(pts)


def _generate_icosahedral(n, spacing=1.0):
    """Generate an icosahedral (Mackay-like) cluster."""
    pts = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    base /= np.linalg.norm(base[0]) * spacing
    pts.append(np.zeros(3))  # center
    pts.extend([p.copy() for p in base])  # 12 vertices
    # Add face centers and edges for larger clusters
    if n > 13:
        # Edge midpoints (30 edges)
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(base[i] - base[j])
                if d < 1.2 * spacing:  # connected
                    edges.append((base[i] + base[j]) / 2.0)
        edges.sort(key=lambda p: np.linalg.norm(p))
        pts.extend(edges[:min(len(edges), n - 13)])
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-3, 3, 3)
            pts.append(p)
    return np.array(pts[:n])


def _generate_random(n, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.8
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 for q in pts):
                pts.append(p)
    return np.array(pts)


def _local_relax(positions, rho, max_iter=500, tol=1e-8):
    """L-BFGS-B local minimization with gradient."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.ravel()

    res = minimize(fun, positions.ravel(), jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                            'maxls': 40, 'maxcor': 20})
    return res.fun, res.x.reshape(-1, 3)


def construct(instance):
    """Basin hopping over structured seeds with deadline guard."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    t_start = time.time()
    deadline = t_start + 48.0  # ~48s per instance, 6 instances = 288s, safety margin

    # Generate diverse seeds
    seeds = []
    try:
        seeds.append(_generate_fcc(n))
    except Exception:
        pass
    try:
        seeds.append(_generate_bcc(n))
    except Exception:
        pass
    try:
        seeds.append(_generate_icosahedral(n))
    except Exception:
        pass
    for s in range(6):
        try:
            seeds.append(_generate_random(n, seed=s * 1000 + n))
        except Exception:
            pass

    best_energy = np.inf
    best_positions = seeds[0].copy() if seeds else np.zeros((n, 3))

    # Phase 1: Relax each seed
    for seed_pos in seeds:
        if time.time() > deadline:
            break
        try:
            e, pos = _local_relax(seed_pos, rho, max_iter=300)
            if e < best_energy and _is_valid(pos):
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            pass

    # Phase 2: Basin hopping from best configuration
    rng = np.random.default_rng(n * 7919)
    bh_count = 0
    while time.time() < deadline - 2.0:
        # Perturb
        sigma = 0.15 + 0.1 * min(bh_count / 20.0, 1.0)
        perturbed = best_positions + rng.normal(0, sigma, best_positions.shape)
        try:
            e, pos = _local_relax(perturbed, rho, max_iter=200)
            if e < best_energy and _is_valid(pos):
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            pass
        bh_count += 1

    # Phase 3: Final polish
    if time.time() < deadline - 0.5:
        try:
            e, pos = _local_relax(best_positions, rho, max_iter=1000, tol=1e-10)
            if e < best_energy and _is_valid(pos):
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
