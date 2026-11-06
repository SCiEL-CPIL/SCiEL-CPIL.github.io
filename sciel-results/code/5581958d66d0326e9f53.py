"""Morse cluster global minima: basin-hopping with diverse structural seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 290.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _fcc_points(n, spacing=1.0):
    """Generate n points in FCC-like arrangement centered at origin."""
    a = spacing
    r_max = a * (n ** (1.0 / 3.0) + 2)
    step = a * 0.5

    n_side = int(np.ceil(2 * r_max / step)) + 2
    coords = []
    for i in range(-n_side // 2, n_side // 2 + 1):
        for j in range(-n_side // 2, n_side // 2 + 1):
            for k in range(-n_side // 2, n_side // 2 + 1):
                x = i * step
                y = j * step
                z = k * step
                if x * x + y * y + z * z <= r_max * r_max:
                    coords.append([x, y, z])

    coords = np.array(coords)
    if len(coords) == 0:
        return np.zeros((n, 3))

    dists = np.sum(coords**2, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx].copy()
    points -= points.mean(axis=0)
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=42)
    return points


def _hcp_points(n, spacing=1.0):
    """Generate n points in HCP-like arrangement."""
    a = spacing
    # HCP: layers of triangular lattice, ABAB stacking
    a1 = np.array([a, 0, 0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2, 0])
    layer_spacing = a * np.sqrt(2.0 / 3.0)
    
    points = []
    n_per_layer_est = max(1, int(np.sqrt(n * 2 / 3)))
    n_layers = max(2, int(np.ceil(np.sqrt(n / n_per_layer_est))))
    
    for layer in range(n_layers):
        z = layer * layer_spacing
        if layer % 2 == 0:
            offset = np.array([0, 0, 0])
        else:
            offset = np.array([a / 2, a * np.sqrt(3) / 6, 0])
        
        n_side = int(np.ceil(np.sqrt(n_per_layer_est) * 2)) + 2
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                p = i * a1 + j * a2 + offset
                if np.linalg.norm(p[:2]) < a * (n ** (1.0/3.0) + 1):
                    points.append(np.array([p[0], p[1], z]))
    
    if len(points) == 0:
        return _fcc_points(n, spacing)
    
    points = np.array(points)
    centroid = points.mean(axis=0)
    dists = np.sum((points - centroid)**2, axis=1)
    idx = np.argsort(dists)[:n]
    points = points[idx].copy()
    points -= points.mean(axis=0)
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=44)
    return points


def _icosahedral_shells(n, spacing=1.0):
    """Generate points using icosahedral shells (good for small clusters)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    points.append(np.zeros(3))
    
    for v in verts:
        points.append(v * spacing)
    
    if n > 13:
        for v in verts:
            points.append(v * 2 * spacing)
    
    if n > 25:
        for v in verts:
            points.append(v * 3 * spacing)
    
    while len(points) < n:
        rng = np.random.default_rng(len(points))
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * (3.5 * spacing)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=45)
    return points


def _bcc_points(n, spacing=1.0):
    """Generate n points in BCC-like arrangement."""
    a = spacing
    r_max = a * (n ** (1.0 / 3.0) + 2)
    step = a * 0.5

    n_side = int(np.ceil(2 * r_max / step)) + 2
    coords = []
    for i in range(-n_side // 2, n_side // 2 + 1):
        for j in range(-n_side // 2, n_side // 2 + 1):
            for k in range(-n_side // 2, n_side // 2 + 1):
                x = i * step
                y = j * step
                z = k * step
                if x * x + y * y + z * z <= r_max * r_max:
                    coords.append([x, y, z])

    coords = np.array(coords)
    if len(coords) == 0:
        return np.zeros((n, 3))

    dists = np.sum(coords**2, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx].copy()
    points -= points.mean(axis=0)
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=43)
    return points


def _random_compact(n, spacing=1.0, seed=0):
    """Random compact cluster with minimum distance enforcement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 50
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
            attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    points = np.array(points)
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=seed)
    return points


def _enforce_min_dist(points, min_dist=0.5, rng_seed=0):
    """Push overlapping points apart to satisfy minimum distance constraint."""
    rng = np.random.default_rng(rng_seed)
    n = len(points)
    for iteration in range(80):
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        mask = dist < min_dist
        if not np.any(mask):
            break
        
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = points[i] - points[j]
                dist_ij = np.linalg.norm(d)
                if dist_ij < min_dist:
                    if dist_ij > 1e-10:
                        push = (min_dist - dist_ij) / dist_ij * 0.5
                        points[i] += d * push
                        points[j] -= d * push
                        moved = True
                    else:
                        points[i] += rng.uniform(-0.1, 0.1, 3)
                        points[j] += rng.uniform(-0.1, 0.1, 3)
                        moved = True
        if not moved:
            break
    return points


def _perturb(points, scale=0.3, seed=0):
    """Add random perturbation."""
    rng = np.random.default_rng(seed)
    perturbed = points + rng.uniform(-scale, scale, points.shape)
    return _enforce_min_dist(perturbed, min_dist=0.5, rng_seed=seed)


def _basin_hop(positions, rho, max_iter=20, time_budget=10.0):
    """
    Basin hopping: perturb, relax, accept/reject based on energy.
    """
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    current = positions.copy()
    current_energy, _ = relax(current.copy(), 'morse', rho)
    best = current.copy()
    best_energy = current_energy
    
    start = time.time()
    kick_scale = 0.3
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        if _time_left() < 2.0:
            break
        
        kick = rng.uniform(-kick_scale, kick_scale, current.shape)
        perturbed = current + kick
        perturbed = _enforce_min_dist(perturbed, min_dist=0.5, rng_seed=it)
        
        try:
            new_energy, new_pos = relax(perturbed.copy(), 'morse', rho)
        except Exception:
            continue
        
        if new_energy < current_energy:
            current = new_pos
            current_energy = new_energy
            kick_scale *= 0.95
        else:
            delta = new_energy - current_energy
            kT = max(0.01, abs(current_energy) * 0.01)
            if rng.random() < np.exp(-delta / kT):
                current = new_pos
                current_energy = new_energy
                kick_scale *= 1.05
            else:
                kick_scale *= 1.02
        
        if new_energy < best_energy:
            best = new_pos.copy()
            best_energy = new_energy
    
    return best, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations using basin hopping with diverse seeds."""
    _init_timer()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    best_energy = np.inf
    best_positions = None
    
    seeds = []
    
    fcc = _fcc_points(n, spacing=1.0)
    seeds.append(("fcc", fcc))
    
    hcp = _hcp_points(n, spacing=1.0)
    seeds.append(("hcp", hcp))
    
    bcc = _bcc_points(n, spacing=1.0)
    seeds.append(("bcc", bcc))
    
    ico = _icosahedral_shells(n, spacing=1.0)
    seeds.append(("ico", ico))
    
    for seed in [0, 1, 2, 3, 4, 5]:
        rc = _random_compact(n, spacing=1.0, seed=seed)
        seeds.append((f"rand{seed}", rc))
    
    for name, base in [("fcc", fcc), ("hcp", hcp), ("bcc", bcc), ("ico", ico)]:
        for s in [10, 11]:
            p = _perturb(base, scale=0.2, seed=s)
            seeds.append((f"p{name}{s}", p))
    
    # Phase 1: Relax each seed to get local minima
    local_minima = []
    for name, config in seeds:
        if _time_left() < 5.0:
            break
        try:
            energy, positions = relax(config.copy(), 'morse', rho)
            if energy < np.inf:
                local_minima.append((energy, positions))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
        except Exception:
            continue
    
    # Phase 2: Basin hopping from the best local minima
    if local_minima and _time_left() > 10.0:
        local_minima.sort(key=lambda x: x[0])
        n_bh = min(5, len(local_minima))
        
        for i in range(n_bh):
            if _time_left() < 3.0:
                break
            
            e0, pos0 = local_minima[i]
            remaining = _time_left() - 2.0
            per_bh = remaining / (n_bh - i)
            
            try:
                bh_pos, bh_energy = _basin_hop(
                    pos0, rho, 
                    max_iter=30, 
                    time_budget=min(per_bh, 15.0)
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos.copy()
            except Exception:
                continue
    
    # Phase 3: If time remains, do more basin hopping from best
    if best_positions is not None and _time_left() > 5.0:
        remaining = _time_left() - 2.0
        n_extra = int(remaining / 8.0)
        n_extra = max(1, min(n_extra, 10))
        
        for i in range(n_extra):
            if _time_left() < 3.0:
                break
            try:
                bh_pos, bh_energy = _basin_hop(
                    best_positions, rho,
                    max_iter=20,
                    time_budget=min(8.0, (_time_left() - 2.0) / (n_extra - i))
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos.copy()
            except Exception:
                continue
    
    if best_positions is None:
        best_positions = fcc
    
    return best_positions
# EVOLVE-BLOCK-END
