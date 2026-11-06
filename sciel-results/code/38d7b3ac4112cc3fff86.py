"""Morse cluster global minima: parallel basin-hopping chains with diverse initialization and adaptive cooling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    mask = dist > 0.5
    dist_m = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_m)
    e = np.exp(x)
    pair_e = e * (e - 2.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sum(pair_e[iu[0], iu[1]]))


def _enforce_min_dist(pos, min_d=0.5):
    """Ensure all pairs are at least min_d apart."""
    n = len(pos)
    for a in range(n):
        for b in range(a + 1, n):
            d = np.linalg.norm(pos[a] - pos[b])
            if d < min_d:
                direction = pos[a] - pos[b]
                if np.linalg.norm(direction) < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                direction = direction / np.linalg.norm(direction)
                shift = (min_d - d) / 2.0 + 0.005
                pos[a] += direction * shift
                pos[b] -= direction * shift
    return pos


def _fcc_points(n, spacing=1.0, seed=0):
    """FCC lattice points."""
    rng = np.random.default_rng(seed)
    a = spacing * 0.7071
    r_est = spacing * n ** (1.0 / 3.0) * 1.3
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for offset in [np.array([0,0,0]), np.array([0.5,0.5,0]), np.array([0.5,0,0.5]), np.array([0,0.5,0.5])]:
                    p = (np.array([i, j, k]) + offset) * a * np.sqrt(2)
                    if np.linalg.norm(p) <= r_est:
                        points.append(p)
    if len(points) == 0:
        return rng.uniform(-r_est, r_est, (n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    if len(points) >= n:
        return points[:n].copy()
    extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
    return np.vstack([points, extra])


def _bcc_points(n, spacing=1.0, seed=0):
    """BCC lattice points."""
    rng = np.random.default_rng(seed)
    a = 2.0 * spacing / np.sqrt(3.0)
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.9
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p1 = np.array([i, j, k]) * a
                if np.linalg.norm(p1) <= r_est:
                    points.append(p1)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
    if len(points) == 0:
        return rng.uniform(-r_est, r_est, (n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    if len(points) >= n:
        return points[:n].copy()
    extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
    return np.vstack([points, extra])


def _sc_points(n, spacing=1.0, seed=0):
    """Simple cubic lattice points."""
    rng = np.random.default_rng(seed)
    a = spacing
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.9
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
    if len(points) == 0:
        return rng.uniform(-r_est, r_est, (n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    if len(points) >= n:
        return points[:n].copy()
    extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
    return np.vstack([points, extra])


def _icosahedral_shells(n, spacing=1.0, seed=0):
    """Icosahedral shell structure."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0]) * spacing
    pts = [np.zeros(3)]
    if n <= 1:
        return np.array(pts[:n])
    pts.extend(verts.tolist())
    for shell_r in [2.0, 2.6, 3.2, 3.8, 4.4, 5.0]:
        verts_s = verts / spacing * shell_r
        for v in verts_s:
            if len(pts) < n:
                pts.append(v.tolist())
    while len(pts) < n:
        p = rng.uniform(-5.0, 5.0, 3)
        pts.append(p.tolist())
    return np.array(pts[:n])


def _random_compact(n, spacing=1.0, seed=0):
    """Random compact configuration with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = n * 300
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _gradient_refine(pos, rho, max_steps=20, step_size=0.01):
    """Langevin-style gradient refinement between hops."""
    current = pos.copy()
    try:
        energy, grad = morse_energy_gradient(current, rho)
    except Exception:
        return current, _morse_energy(current, rho)
    for _ in range(max_steps):
        if _time_left() < 2.0:
            break
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-6:
            break
        step = -step_size * grad / norm_grad
        # Add small noise for Langevin
        noise = 0.005 * np.random.default_rng(hash(tuple(current.ravel())) % 2**32).uniform(-1, 1, current.shape)
        proposed = current + step + noise
        try:
            new_e, new_grad = morse_energy_gradient(proposed, rho)
        except Exception:
            break
        if new_e < energy:
            current = proposed
            energy = new_e
            grad = new_grad
            step_size *= 0.95
        else:
            step_size *= 0.9
    return current, energy


def _basin_hop_chain(init_pos, rho, chain_id, max_iter, kick_start, cool_rate, seed):
    """Single basin-hopping chain with specific parameters."""
    rng = np.random.default_rng(seed + chain_id * 1000)
    current_pos = init_pos.copy()
    try:
        current_energy, current_pos = relax(current_pos, 'morse', rho)
    except Exception:
        current_energy = _morse_energy(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    kick_scale = kick_start
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        
        # Generate kick
        kick = rng.uniform(-kick_scale, kick_scale, current_pos.shape)
        proposed_pos = current_pos + kick
        
        # Relax proposed
        try:
            proposed_energy, proposed_pos = relax(proposed_pos, 'morse', rho)
        except Exception:
            proposed_energy = _morse_energy(proposed_pos)
            proposed_pos = proposed_pos
        
        # Metropolis acceptance with adaptive temperature
        delta = proposed_energy - current_energy
        T = 0.15 * (cool_rate ** it) + 0.005
        accept = delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10))
        
        if accept:
            current_pos = proposed_pos
            current_energy = proposed_energy
            # Gradient refinement between hops
            if it % 3 == 0:
                refined_pos, refined_e = _gradient_refine(current_pos, rho, max_steps=10, step_size=0.008)
                if refined_e < current_energy:
                    current_pos = refined_pos
                    current_energy = refined_e
            
            if current_energy < best_energy:
                best_pos = current_pos.copy()
                best_energy = current_energy
        
        # Cool kick scale
        kick_scale *= cool_rate
        kick_scale = max(kick_scale, 0.02)
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster via parallel basin-hopping chains."""
    _instance_call_count[0] += 1
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget: ~45s per instance, 6 instances = 270s total
    per_instance_budget = 44.0
    instance_deadline = time.time() + per_instance_budget
    
    # Generate diverse initial configurations
    init_generators = [
        lambda: _fcc_points(n, 1.0, seed=1),
        lambda: _bcc_points(n, 1.0, seed=2),
        lambda: _sc_points(n, 1.0, seed=3),
        lambda: _icosahedral_shells(n, 1.0, seed=4),
        lambda: _random_compact(n, 1.0, seed=10),
        lambda: _random_compact(n, 1.0, seed=20),
        lambda: _random_compact(n, 1.0, seed=30),
        lambda: _fcc_points(n, 1.0, seed=40),
    ]
    
    # Chain parameters: (kick_start, cool_rate, max_iter)
    # Different chains explore different scales
    chain_params = [
        (0.15, 0.96, 35),  # Moderate kick, slow cool
        (0.25, 0.93, 30),  # Large kick, fast cool (explore)
        (0.10, 0.97, 40),  # Small kick, very slow cool (refine)
        (0.20, 0.95, 32),  # Balanced
    ]
    
    # Assign initial configs to chains (round-robin for diversity)
    n_chains = min(len(chain_params), 4)
    chain_inits = []
    for c in range(n_chains):
        gen_idx = (c * 2) % len(init_generators)  # Spread out
        try:
            init = init_generators[gen_idx]()
            init = _enforce_min_dist(init.copy())
            chain_inits.append(init)
        except Exception:
            chain_inits.append(_random_compact(n, 1.0, seed=c + 50))
    
    # Run parallel chains (sequential but independent)
    best_energy = np.inf
    best_pos = None
    chain_results = []
    
    for c in range(n_chains):
        if time.time() > instance_deadline or _time_left() < 5.0:
            break
        
        kick_start, cool_rate, max_iter = chain_params[c]
        seed = c * 100 + 7
        
        try:
            e, p = _basin_hop_chain(chain_inits[c], rho, c, max_iter, kick_start, cool_rate, seed)
            chain_results.append((e, p))
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue
    
    # If no chain succeeded, do a simple relaxation
    if best_pos is None:
        try:
            init = _random_compact(n, 1.0, seed=0)
            init = _enforce_min_dist(init)
            best_energy, best_pos = relax(init, 'morse', rho)
        except Exception:
            best_pos = _random_compact(n, 1.0, seed=0)
            best_energy = _morse_energy(best_pos)
    
    # If we have time, do a final refinement pass on the best
    if _time_left() > 5.0 and time.time() < instance_deadline:
        try:
            refined_pos, refined_e = _gradient_refine(best_pos, rho, max_steps=30, step_size=0.005)
            if refined_e < best_energy:
                best_energy = refined_e
                best_pos = refined_pos.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos = _enforce_min_dist(best_pos.copy())
    
    return best_pos
# EVOLVE-BLOCK-END
