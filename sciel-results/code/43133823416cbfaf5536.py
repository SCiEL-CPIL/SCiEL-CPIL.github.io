"""Dzugutov cluster global minima: Mackay icosahedral seeds + adaptive basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 280.0
_STOP = threading.Event()

_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, TIME_LIMIT - _elapsed())


def _energy(positions):
    """Direct Dzugutov energy (vectorised)."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    rv = r[iu]
    e = np.zeros_like(rv)
    m1 = rv < _a
    m2 = (rv >= _a) & (rv < _b)
    if m1.any():
        r1 = rv[m1]
        e[m1] = _A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a))
    if m2.any():
        r2 = rv[m2]
        e[m2] = _B * np.exp(_d / (r2 - _b))
    return float(np.sum(e))


def _relax(positions, deadline=None):
    """Relax using the provided relaxer; fall back to gradient descent."""
    pos = np.asarray(positions, dtype=float)
    try:
        energy, out = relax(pos, "dzugutov")
        if out is not None:
            out = np.asarray(out, dtype=float)
            if out.shape == pos.shape and np.all(np.isfinite(out)):
                return float(energy), out
    except Exception:
        pass
    # Fallback: adaptive gradient descent with backtracking
    p = pos.copy()
    best_e = _energy(p)
    best_p = p.copy()
    lr = 0.01
    for it in range(300):
        if deadline is not None and _elapsed() > deadline:
            break
        if _STOP.is_set():
            break
        try:
            g = dzugutov_energy_gradient(p)
        except Exception:
            break
        g = np.asarray(g, dtype=float)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        # Adaptive step size
        step = min(0.5, 0.05 * lr * gn)
        new_p = p - step * g / gn
        if not np.all(np.isfinite(new_p)):
            step *= 0.1
            new_p = p - step * g / gn
            if not np.all(np.isfinite(new_p)):
                break
        e_new = _energy(new_p)
        if e_new < best_e:
            best_e = e_new
            best_p = new_p.copy()
            p = new_p
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.7, 0.001)
            if lr < 0.002:
                break
    return best_e, best_p


def _icosahedron_vertices():
    """Standard icosahedron vertices (12), normalized to unit edge length."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    edge = np.linalg.norm(v[0] - v[1])
    return v / edge


def _shell_directions(shell_num):
    """Candidate unit directions for Mackay icosahedral shell."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = _icosahedron_vertices()
    
    if shell_num == 1:
        return ico  # 12 vertices
    
    if shell_num == 2:
        # 42 directions: 20 face centers + 22 edge midpoints
        dirs = []
        faces = [
            [0, 1, 8], [0, 8, 4], [0, 4, 11], [0, 11, 2], [0, 2, 1],
            [1, 2, 5], [2, 11, 3], [11, 4, 9], [4, 8, 10], [8, 1, 6],
            [5, 2, 3], [3, 11, 7], [7, 9, 3], [9, 4, 10], [10, 8, 6],
            [6, 1, 5], [5, 3, 7], [7, 9, 10], [10, 6, 5], [6, 10, 8],
        ]
        for f in faces:
            c = (ico[f[0]] + ico[f[1]] + ico[f[2]]) / 3.0
            c /= np.linalg.norm(c)
            dirs.append(c)
        # Edge midpoints - pick 22 from the 30 edges
        edges = [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 11],
            [1, 2], [1, 5], [1, 6], [1, 8],
            [2, 3], [2, 5], [2, 11],
            [3, 7], [3, 9], [3, 11],
            [4, 9], [4, 10], [4, 11],
            [5, 6], [5, 7],
            [6, 7], [6, 10],
        ]
        for e in edges:
            c = (ico[e[0]] + ico[e[1]]) / 2.0
            c /= np.linalg.norm(c)
            dirs.append(c)
        return np.array(dirs[:42])
    
    # Higher shells: use Fibonacci sphere for even distribution
    n_pts = 92 if shell_num == 3 else 120
    pts = []
    for i in range(n_pts):
        z = 1 - 2 * (i + 0.5) / n_pts
        r = np.sqrt(1 - z*z)
        theta = np.pi * (1 + np.sqrt(5)) * i
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _build_icosahedral(n, scale=1.0, rng=None):
    """Build an n-atom Mackay icosahedral cluster."""
    if rng is None:
        rng = np.random.default_rng(n)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    pts = [np.zeros(3) * scale]  # center
    
    # Shell definitions: (radius_multiplier, count)
    shells = [
        (1.0, 12),
        (1.0 + 1.0/phi, 42),
        (1.0 + 2.0/phi, 92),
        (1.0 + 3.0/phi, 120),
    ]
    
    for shell_idx, (radius_mult, count) in enumerate(shells):
        if len(pts) >= n:
            break
        need = min(count, n - len(pts))
        radius = radius_mult * scale
        
        if shell_idx == 0:
            dirs = _icosahedron_vertices()
        else:
            dirs = _shell_directions(shell_idx + 1)
        
        if dirs is not None and len(dirs) >= need:
            chosen = dirs[:need]
        else:
            # Fallback: scatter on sphere
            chosen = []
            for i in range(need):
                z = 1 - 2 * (i + 0.5) / need
                r = np.sqrt(1 - z*z)
                theta = np.pi * (1 + np.sqrt(5)) * i
                v = np.array([r * np.cos(theta), r * np.sin(theta), z])
                chosen.append(v)
            chosen = np.array(chosen)
        
        for d in chosen:
            pts.append(d * radius)
    
    # If we still need more atoms, add them near the surface
    while len(pts) < n:
        outer_r = (1.0 + 3.0/phi) * scale
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * outer_r * 1.02)
    
    return np.array(pts[:n])


def _build_polycrystal(n, rng=None):
    """Build a polycrystalline seed: multiple small icosahedra."""
    if rng is None:
        rng = np.random.default_rng(n)
    pts = []
    # Place 2-4 small icosahedral clusters
    n_clusters = min(4, max(2, n // 15))
    cluster_sizes = [n // n_clusters] * n_clusters
    cluster_sizes[-1] += n - sum(cluster_sizes)
    
    for i, size in enumerate(cluster_sizes):
        if size < 13:
            continue
        small_ico = _build_icosahedral(min(size, 13), scale=0.9, rng=rng)
        # Place cluster center
        if i == 0:
            center = np.zeros(3)
        else:
            angle = 2 * np.pi * i / n_clusters
            radius = 3.5
            center = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
        pts.extend([center + p for p in small_ico])
    
    # Fill remaining with random compact positions
    while len(pts) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * 2.5)
    
    return np.array(pts[:n])


def _random_compact(n, spacing=1.1, rng=None):
    """Random compact cluster with minimum separation."""
    if rng is None:
        rng = np.random.default_rng(n)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    pts = []
    min_dist = 0.6 * spacing
    tries = 0
    while len(pts) < n and tries < 5000:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:n])


def _seeds(n):
    """List of diverse candidate starting geometries for cluster size n."""
    rng = np.random.default_rng(1000 + n)
    seeds = []
    
    # Primary: proper Mackay icosahedral (favoured by Dzugutov)
    try:
        seeds.append(_build_icosahedral(n, scale=1.0, rng=rng))
    except Exception:
        pass
    
    # Multiple icosahedral scales for better coverage
    for scale in (0.92, 0.97, 1.03, 1.08):
        try:
            seeds.append(_build_icosahedral(n, scale=scale, rng=rng))
        except Exception:
            pass
    
    # Polycrystalline seed
    try:
        seeds.append(_build_polycrystal(n, rng=rng))
    except Exception:
        pass
    
    # Random compact seeds with varied spacing
    for spacing in (1.0, 1.15, 1.25):
        seeds.append(_random_compact(n, spacing=spacing, rng=rng))
    
    # Perturbed icosahedral with different noise levels
    if seeds:
        base = seeds[0]
        for noise in (0.12, 0.2, 0.3):
            perturbed = base + rng.normal(0, noise, size=base.shape)
            seeds.append(perturbed)
    
    return seeds


def _basin_hopping_worker(seed_pos, n, deadline, worker_id):
    """Run a single basin-hopping trajectory from seed_pos until deadline."""
    pos = np.asarray(seed_pos, dtype=float).copy()
    best_e = np.inf
    best_pos = pos.copy()
    
    # Initial relaxation
    try:
        e, p = _relax(pos, deadline)
        if np.all(np.isfinite(p)):
            best_e = e
            best_pos = p.copy()
    except Exception:
        best_e = _energy(pos)
        best_pos = pos.copy()
    
    if _STOP.is_set() or _elapsed() > deadline:
        return best_e, best_pos
    
    # Basin hopping with adaptive parameters
    amp = 0.2
    temp = 1.0
    rng = np.random.default_rng(42 + worker_id * 1000 + n)
    current_e = best_e
    current_pos = best_pos.copy()
    consecutive_rejects = 0
    n_iter = 0
    
    while not _STOP.is_set() and _elapsed() < deadline:
        # Adaptive perturbation: occasional large jumps
        if n_iter % 15 == 14:
            jump_amp = amp * 2.5
        else:
            jump_amp = amp
        
        noise = rng.normal(0, jump_amp, size=current_pos.shape)
        perturbed = current_pos + noise
        if not np.all(np.isfinite(perturbed)):
            amp *= 0.5
            continue
        
        # Relax perturbed structure
        try:
            e_new, p_new = _relax(perturbed, deadline)
        except Exception:
            e_new = _energy(perturbed)
            p_new = perturbed
        
        if not np.all(np.isfinite(p_new)):
            amp *= 0.5
            continue
        
        # Acceptance criterion with Metropolis
        if e_new < current_e:
            current_e = e_new
            current_pos = p_new.copy()
            amp = min(amp * 1.1, 0.45)
            consecutive_rejects = 0
        else:
            delta_e = e_new - current_e
            if delta_e > 0:
                p_accept = np.exp(-delta_e / max(temp, 1e-8))
                if rng.random() > p_accept:
                    amp = max(amp * 0.9, 0.06)
                    temp = max(temp * 0.95, 0.1)
                    consecutive_rejects += 1
                    if consecutive_rejects > 12:
                        # Reset to best and try larger jumps
                        current_pos = best_pos.copy()
                        current_e = best_e
                        amp = min(amp * 1.4, 0.5)
                        consecutive_rejects = 0
                    continue
            current_e = e_new
            current_pos = p_new.copy()
        
        # Update global best
        if e_new < best_e:
            best_e = e_new
            best_pos = p_new.copy()
            consecutive_rejects = 0
        
        n_iter += 1
        temp = max(temp * 0.985, 0.05)
    
    return best_e, best_pos


def construct(instance):
    """Parallel basin-hopping search over diverse icosahedral seeds."""
    size = instance["n"]
    potential = instance["potential"]
    
    # Generate diverse seeds
    seeds = _seeds(size)
    
    # Determine how many parallel workers to use (max 4 threads)
    n_workers = min(4, len(seeds))
    
    # Deadline: leave 10s margin for final processing
    deadline = _elapsed() + max(10.0, _remaining() - 10.0)
    
    # Run parallel basin hopping
    results = []
    best_e = np.inf
    best_pos = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            seed = seeds[i % len(seeds)]
            f = executor.submit(_basin_hopping_worker, seed, size, deadline, i)
            futures.append(f)
        
        for f in futures:
            try:
                e, p = f.result(timeout=max(5.0, _remaining()))
                if np.all(np.isfinite(p)):
                    results.append((e, p))
                    if e < best_e:
                        best_e = e
                        best_pos = p.copy()
            except Exception:
                continue
    
    # If no valid result, fall back to best seed relaxed
    if best_pos is None:
        for s in seeds:
            try:
                e, p = _relax(s, _elapsed() + max(5.0, _remaining()))
                if np.all(np.isfinite(p)) and e < best_e:
                    best_e = e
                    best_pos = p.copy()
            except Exception:
                continue
    
    if best_pos is None:
        best_pos = _build_icosahedral(size, scale=1.0)
        try:
            _, best_pos = relax(best_pos, potential)
        except Exception:
            pass
    
    best_pos = np.asarray(best_pos, dtype=float)
    return best_pos
# EVOLVE-BLOCK-END
