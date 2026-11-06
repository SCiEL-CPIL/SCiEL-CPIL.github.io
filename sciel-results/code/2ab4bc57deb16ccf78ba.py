"""Morse cluster global minima search with diverse structural starts and multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize


def _morse_energy(pos, rho):
    """Compute Morse energy with well depth 1."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(r, np.inf)
    exp_term = np.exp(rho * (1.0 - r))
    e = np.sum(exp_term * (exp_term - 2.0))
    return 0.5 * e


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 3
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx].copy()
    rng = np.random.default_rng(100)
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    return cluster


def _generate_bcc(n):
    """Generate BCC-like cluster of n atoms."""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    side = int(np.ceil((0.5 * n) ** (1.0/3.0))) + 3
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx].copy()
    rng = np.random.default_rng(101)
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    return cluster


def _generate_icosahedral(n):
    """Generate icosahedral-like cluster (Mackay shells)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]

    points = [np.zeros(3)]

    # Shell 1: 12 vertices at radius 1
    points.extend([v.copy() for v in verts])

    # Shell 2: 42 atoms at radius 2
    # These are the midpoints of the 30 edges and centroids of 12 faces
    # Simplified: generate points on sphere of radius 2 with icosahedral symmetry
    # For now, use a dense sampling and keep those closest to icosahedral directions
    rng = np.random.default_rng(102)
    min_dist = 0.55
    target_r = 2.0
    attempts = 0
    while len(points) < 55 and attempts < 5000:
        # Try adding points in shell 2
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * target_r
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1

    # If we need more, add shell 3
    if n > 55:
        target_r = 3.0
        attempts = 0
        while len(points) < n and attempts < 10000:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * target_r
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1

    # Pad with random if needed
    while len(points) < n:
        p = rng.uniform(-4, 4, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        else:
            # Force add if stuck
            points.append(p)

    return np.array(points[:n])


def _generate_hcp(n):
    """Generate HCP-like cluster."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    radius = np.sqrt(n / (np.pi * 0.866))
    num_layers = int(np.ceil(2 * radius / (c/2))) + 2

    for layer_idx in range(-num_layers//2, num_layers//2 + 1):
        z = layer_idx * (c / 2.0)
        if layer_idx % 2 == 0:
            offset = np.array([0.0, 0.0])
        else:
            offset = np.array([a/2.0, a*np.sqrt(3)/6.0])
        for i in range(-int(radius), int(radius) + 1):
            for j in range(-int(radius), int(radius) + 1):
                x = a * (i + j/2.0) + offset[0]
                y = a * (np.sqrt(3)/2.0 * j) + offset[1]
                p = np.array([x, y, z])
                if np.linalg.norm(p[:2]) <= radius:
                    pts.append(p)

    if len(pts) < n:
        return _generate_fcc(n)

    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx].copy()
    rng = np.random.default_rng(103)
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    return cluster


def _generate_random_compact(n, seed):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1.0/3.0) * 0.9
    points = []
    min_dist = 0.52
    attempts = 0
    while len(points) < n and attempts < 50000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            points.append(p)
    return np.array(points[:n])


def _optimize(pos, rho, time_budget=10.0, deadline=None):
    """Multi-stage optimization: relax, then scipy optimizers."""
    start = time.time()
    if deadline is not None and time.time() > deadline:
        return _morse_energy(pos, rho), pos

    # Stage 1: built-in relax
    try:
        e, pos = relax(pos.copy(), 'morse', rho)
    except Exception:
        e = _morse_energy(pos, rho)

    elapsed = time.time() - start
    if deadline is not None and time.time() > deadline:
        return e, pos
    if elapsed > time_budget:
        return e, pos

    # Stage 2: L-BFGS-B
    try:
        def fun(x):
            p = x.reshape(-1, 3)
            return _morse_energy(p, rho)
        def grad(x):
            p = x.reshape(-1, 3)
            try:
                e, g = morse_energy_gradient(p, rho)
                return g.flatten()
            except Exception:
                return None
        x0 = pos.flatten()
        res = minimize(fun, x0, method='L-BFGS-B', jac=grad,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        if res.success or res.fun < e:
            e = res.fun
            pos = res.x.reshape(-1, 3)
    except Exception:
        pass

    elapsed = time.time() - start
    if deadline is not None and time.time() > deadline:
        return e, pos
    if elapsed > time_budget:
        return e, pos

    # Stage 3: BFGS polish
    try:
        def fun2(x):
            p = x.reshape(-1, 3)
            return _morse_energy(p, rho)
        def grad2(x):
            p = x.reshape(-1, 3)
            try:
                e, g = morse_energy_gradient(p, rho)
                return g.flatten()
            except Exception:
                return None
        x0 = pos.flatten()
        res = minimize(fun2, x0, method='BFGS', jac=grad2,
                      options={'maxiter': 300, 'gtol': 1e-10})
        if res.success or res.fun < e:
            e = res.fun
            pos = res.x.reshape(-1, 3)
    except Exception:
        pass

    return e, pos


def construct(instance):
    """Find low-energy Morse cluster arrangement."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))

    global_start = time.time()
    total_budget = 300.0
    safety_margin = 15.0
    effective_budget = total_budget - safety_margin
    deadline = global_start + effective_budget

    # Generate diverse starting structures
    generators = [
        lambda: _generate_fcc(n),
        lambda: _generate_bcc(n),
        lambda: _generate_icosahedral(n),
        lambda: _generate_hcp(n),
        lambda: _generate_random_compact(n, seed=n*7+1),
        lambda: _generate_random_compact(n, seed=n*13+3),
        lambda: _generate_random_compact(n, seed=n*17+7),
    ]

    best_energy = np.inf
    best_pos = None

    n_structs = len(generators)
    per_struct_budget = effective_budget / (n_structs + 2)

    for i, gen in enumerate(generators):
        elapsed = time.time() - global_start
        remaining = effective_budget - elapsed
        if remaining < 5.0:
            break

        alloc = min(per_struct_budget, remaining * 0.8)
        if alloc < 2.0:
            break

        try:
            pos = gen()
            e, pos = _optimize(pos, rho, time_budget=alloc, deadline=deadline)

            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            continue

    # Final polish with remaining time
    elapsed = time.time() - global_start
    remaining = effective_budget - elapsed
    if remaining > 3.0 and best_pos is not None:
        try:
            e, pos = _optimize(best_pos, rho, time_budget=remaining, deadline=deadline)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            pass

    # Ensure minimum distance constraint
    if best_pos is None:
        rng = np.random.default_rng(n)
        radius = 1.0 * n ** (1.0/3.0)
        best_pos = []
        for _ in range(n):
            for _ in range(10000):
                p = rng.uniform(-radius, radius, 3)
                if all(np.linalg.norm(p - q) >= 0.5 for q in best_pos):
                    best_pos.append(p)
                    break
            else:
                p = rng.uniform(-radius, radius, 3)
                best_pos.append(p)
        best_pos = np.array(best_pos)

    # Final check: enforce minimum distance
    # If any pair is too close, nudge them apart
    pos = best_pos.copy()
    for _ in range(100):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d >= 0.5:
            break
        # Find the closest pair
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = idx
        d = pos[i] - pos[j]
        norm = np.linalg.norm(d)
        if norm < 1e-10:
            d = np.array([0.01, 0, 0])
            norm = 0.01
        push = (0.5 - norm + 0.01) / norm * d * 0.5
        pos[i] += push
        pos[j] -= push

    return pos
# EVOLVE-BLOCK-END
