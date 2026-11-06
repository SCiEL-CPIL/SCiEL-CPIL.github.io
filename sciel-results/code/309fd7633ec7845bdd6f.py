"""Dzugutov cluster global minima: Mackay icosahedral seeds + parallel basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 285.0  # safety margin under 300s kill
_STOP = threading.Event()

# Dzugutov parameters (kept for reference / possible direct energy use)
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
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    rv = r[iu]
    if rv.size == 0:
        return 0.0
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
    # Try the provided relaxer first
    try:
        energy, out = relax(pos, "dzugutov")
        if out is not None:
            out = np.asarray(out, dtype=float)
            if out.shape == pos.shape and np.all(np.isfinite(out)):
                return float(energy), out
    except Exception:
        pass
    # Fallback: adaptive gradient descent
    p = pos.copy()
    best_e = _energy(p)
    best_p = p.copy()
    for it in range(400):
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
        s = min(0.3, 0.02 * gn)
        new_p = p - s * g / gn
        if not np.all(np.isfinite(new_p)):
            s *= 0.1
            new_p = p - s * g / gn
            if not np.all(np.isfinite(new_p)):
                break
        e_new = _energy(new_p)
        if e_new < best_e:
            best_e = e_new
            best_p = new_p.copy()
            p = new_p
        else:
            if s < 1e-6:
                break
    return best_e, best_p


def _icosahedron():
    """Standard icosahedron vertices (12), normalized to unit edge length."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize so edge length is 1
    edge = np.linalg.norm(v[0] - v[1])
    return v / edge


def _icosahedral_shells():
    """Mackay icosahedral shell radii and counts.
    
    Shell 0: 1 atom (center), radius 0
    Shell 1: 12 atoms (icosahedron vertices), radius 1.0
    Shell 2: 42 atoms (face centers + edge midpoints), radius 1 + 1/phi
    Shell 3: 92 atoms, radius 1 + 2/phi  
    Shell 4: 120 atoms, radius 1 + 3/phi
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    return [
        (0.0, 1),      # center
        (1.0, 12),     # first icosahedral shell
        (1.0 + 1.0/phi, 42),  # second shell
        (1.0 + 2.0/phi, 92),  # third shell
        (1.0 + 3.0/phi, 120), # fourth shell
    ]


def _build_icosahedral(n, scale=1.0, rng=None):
    """Build an n-atom Mackay icosahedral cluster by filling shells properly."""
    if rng is None:
        rng = np.random.default_rng(n)
    shells = _icosahedral_shells()
    pts = []
    for radius, count in shells:
        if len(pts) >= n:
            break
        need = min(count, n - len(pts))
        if radius == 0.0:
            # Center atom
            pts.append(np.zeros(3) * scale)
        else:
            # Get directions for this shell
            cand = _shell_directions(int(round(radius - 1.0)) if radius > 1.0 else 1)
            if cand is not None and len(cand) >= need:
                chosen = cand[:need]
            else:
                # Scatter on a sphere of radius radius
                chosen = _scatter(need, 1.0, rng)
            for d in chosen:
                pts.append(d * radius * scale)
    # If we still need more atoms, add them near the surface
    while len(pts) < n:
        # Add atoms at random positions on the outermost shell
        outer_r = shells[-1][0] * scale
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * outer_r * 1.05)
    return np.array(pts[:n])


def _shell_directions(shell_num):
    """Candidate unit directions for Mackay icosahedral shell.
    
    shell_num=1: 12 vertices (icosahedron)
    shell_num=2: 42 directions (20 face centers + 22 edge midpoints)
    shell_num=3: 92 directions (scatter)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = _icosahedron()
    
    if shell_num == 1:
        return ico
    
    if shell_num == 2:
        # 42 directions: 20 face centers + 22 edge midpoints
        dirs = []
        # Face centers (20 triangular faces)
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
        # Edge midpoints (30 edges, but we only need 22 more)
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
    
    # For higher shells, use a more uniform distribution
    rng = np.random.default_rng(shell_num * 1000)
    n_pts = 92 if shell_num == 3 else 120
    # Use Fibonacci sphere for even distribution
    pts = []
    for i in range(n_pts):
        z = 1 - 2 * (i + 0.5) / n_pts
        r = np.sqrt(1 - z*z)
        theta = np.pi * (1 + np.sqrt(5)) * i
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _scatter(need, radius, rng):
    """need unit vectors * radius, roughly evenly spread on a sphere using Fibonacci spiral."""
    pts = []
    for i in range(need):
        z = 1 - 2 * (i + 0.5) / need
        r = np.sqrt(1 - z*z)
        theta = np.pi * (1 + np.sqrt(5)) * i
        v = np.array([r * np.cos(theta), r * np.sin(theta), z])
        pts.append(v * radius)
    return np.array(pts)


def _fcc_cluster(n, scale=1.0):
    """Build an FCC-like cluster (cuboctahedral shells) as a polycrystalline seed."""
    pts = [np.zeros(3)]
    a = scale
    # first shell: 12 (cuboctahedron)
    s = a / np.sqrt(2.0)
    shell1 = []
    for (x, y, z) in [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                      (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]:
        shell1.append(np.array([x, y, z]) * s)
    for p in shell1:
        if len(pts) < n:
            pts.append(p)
    # second shell: 30 (octahedron edges)
    s2 = a
    shell2 = []
    for (x, y, z) in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -8),
                      (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0),
                      (0, 0, 2), (0, 0, -2)]:
        shell2.append(np.array([x, y, z]) * a / 2.0)
    rng = np.random.default_rng(0)
    for p in shell2:
        if len(pts) < n:
            pts.append(p)
    # fill remaining randomly near surface
    while len(pts) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * s2 * 1.6)
    return np.array(pts[:n])


def _random_compact(n, spacing=1.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(n)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    pts = []
    tries = 0
    while len(pts) < n and tries < 4000:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.75 * spacing for q in pts):
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
    # Slightly perturbed icosahedral structures at different scales
    for scale in (0.92, 0.97, 1.03, 1.08):
        try:
            seeds.append(_build_icosahedral(n, scale=scale, rng=rng))
        except Exception:
            pass
    # FCC / polycrystalline seed
    try:
        seeds.append(_fcc_cluster(n, scale=1.0))
    except Exception:
        pass
    # Random compact seeds
    for _ in range(3):
        seeds.append(_random_compact(n, spacing=1.1, rng=rng))
    # Additional perturbed icosahedral
    if seeds:
        base = seeds[0]
        for _ in range(2):
            perturbed = base + rng.normal(0, 0.12, size=base.shape)
            seeds.append(perturbed)
    return seeds


def _basin_hopping_worker(seed_pos, n, deadline, worker_id):
    """Run a single basin-hopping trajectory from seed_pos until deadline.
    
    Returns (best_energy, best_positions).
    """
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
    
    # Basin hopping: perturb + re-relax with Metropolis acceptance
    amp = 0.18  # perturbation amplitude
    temp = 0.5  # effective temperature for Metropolis
    rng = np.random.default_rng(42 + worker_id * 1000 + n)
    current_e = best_e
    current_pos = best_pos.copy()
    n_iter = 0
    
    while not _STOP.is_set() and _elapsed() < deadline:
        # Adaptive perturbation
        noise = rng.normal(0, amp, size=current_pos.shape)
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
        
        # Acceptance criterion
        if e_new < current_e:
            current_e = e_new
            current_pos = p_new.copy()
            amp = min(amp * 1.1, 0.4)  # expand search slightly
        else:
            # Metropolis: accept with probability exp(-(E_new - E_cur)/T)
            delta_e = e_new - current_e
            if delta_e > 0:
                p_accept = np.exp(-delta_e / max(temp, 1e-8))
                if rng.random() > p_accept:
                    amp = max(amp * 0.9, 0.05)  # shrink search
                    temp = max(temp * 0.95, 0.05)
                    continue
            current_e = e_new
            current_pos = p_new.copy()
        
        # Update global best
        if e_new < best_e:
            best_e = e_new
            best_pos = p_new.copy()
        
        n_iter += 1
        # Gradually cool
        temp = max(temp * 0.98, 0.02)
    
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
