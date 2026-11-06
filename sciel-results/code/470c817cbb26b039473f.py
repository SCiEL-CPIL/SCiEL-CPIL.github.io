"""Morse cluster global minima: diverse lattice seeds + perturb-refine cycles."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

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
    r_max = a * (n ** (1.0/3.0) + 2)
    step = a * 0.5
    n_side = int(np.ceil(2 * r_max / step)) + 2
    coords = []
    for i in range(-n_side//2, n_side//2 + 1):
        for j in range(-n_side//2, n_side//2 + 1):
            for k in range(-n_side//2, n_side//2 + 1):
                x = i * step
                y = j * step
                z = k * step
                if x*x + y*y + z*z <= r_max*r_max:
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
    """Generate n points in HCP-like arrangement centered at origin."""
    a = spacing
    r_max = a * (n ** (1.0/3.0) + 2)
    step = a * 0.5
    n_side = int(np.ceil(2 * r_max / step)) + 2
    coords = []
    # HCP: ABAB stacking. Layer A: (i*step, j*step, k*2*step)
    # Layer B: offset by (step/2, step*sqrt(3)/6, step*sqrt(3)/3) approximately
    for i in range(-n_side//2, n_side//2 + 1):
        for j in range(-n_side//2, n_side//2 + 1):
            for k in range(-n_side//2, n_side//2 + 1):
                # A layer
                x = i * step
                y = j * step
                z = k * 2 * step
                if x*x + y*y + z*z <= r_max*r_max:
                    coords.append([x, y, z])
                # B layer (offset)
                x2 = (i + 0.5) * step
                y2 = (j + 0.2887) * step  # sqrt(3)/6
                z2 = (k + 0.5774) * 2 * step  # sqrt(3)/3
                if x2*x2 + y2*y2 + z2*z2 <= r_max*r_max:
                    coords.append([x2, y2, z2])
    coords = np.array(coords)
    if len(coords) == 0:
        return np.zeros((n, 3))
    dists = np.sum(coords**2, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx].copy()
    points -= points.mean(axis=0)
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=44)
    return points


def _sc_points(n, spacing=1.0):
    """Generate n points in simple cubic arrangement."""
    a = spacing
    r_max = a * (n ** (1.0/3.0) + 2)
    step = a * 0.7
    n_side = int(np.ceil(2 * r_max / step)) + 2
    coords = []
    for i in range(-n_side//2, n_side//2 + 1):
        for j in range(-n_side//2, n_side//2 + 1):
            for k in range(-n_side//2, n_side//2 + 1):
                x = i * step
                y = j * step
                z = k * step
                if x*x + y*y + z*z <= r_max*r_max:
                    coords.append([x, y, z])
    coords = np.array(coords)
    if len(coords) == 0:
        return np.zeros((n, 3))
    dists = np.sum(coords**2, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx].copy()
    points -= points.mean(axis=0)
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=45)
    return points


def _bcc_points(n, spacing=1.0):
    """Generate n points in BCC-like arrangement."""
    a = spacing
    r_max = a * (n ** (1.0/3.0) + 2)
    step = a * 0.5
    
    n_side = int(np.ceil(2 * r_max / step)) + 2
    coords = []
    for i in range(-n_side//2, n_side//2 + 1):
        for j in range(-n_side//2, n_side//2 + 1):
            for k in range(-n_side//2, n_side//2 + 1):
                x = i * step
                y = j * step
                z = k * step
                if x*x + y*y + z*z <= r_max*r_max:
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
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = n * 50
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
            attempts += 1
    # If we couldn't place all, fill with random
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
    for iteration in range(30):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-10:
                    push = (min_dist - dist) / dist * 0.5
                    points[i] += d * push
                    points[j] -= d * push
                    moved = True
                elif dist <= 1e-10:
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


def construct(instance):
    """Find low-energy Morse cluster configurations using diverse lattice seeds + perturb-refine."""
    _init_timer()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    initial_configs = []
    
    # FCC lattice
    fcc = _fcc_points(n, spacing=1.0)
    initial_configs.append(fcc)
    
    # HCP lattice
    hcp = _hcp_points(n, spacing=1.0)
    initial_configs.append(hcp)
    
    # BCC lattice
    bcc = _bcc_points(n, spacing=1.0)
    initial_configs.append(bcc)
    
    # Simple cubic
    sc = _sc_points(n, spacing=1.0)
    initial_configs.append(sc)
    
    # Random compact (multiple seeds)
    for seed in [0, 1, 2, 3, 4, 5]:
        rc = _random_compact(n, spacing=1.0, seed=seed)
        initial_configs.append(rc)
    
    # Perturbed FCC (larger scale for exploration)
    for seed in [10, 11, 12, 13]:
        pf = _perturb(fcc, scale=0.2, seed=seed)
        initial_configs.append(pf)
    
    # Perturbed HCP
    for seed in [14, 15]:
        ph = _perturb(hcp, scale=0.2, seed=seed)
        initial_configs.append(ph)
    
    # Perturbed BCC
    for seed in [20, 21]:
        pb = _perturb(bcc, scale=0.2, seed=seed)
        initial_configs.append(pb)
    
    # Relax each configuration
    for config in initial_configs:
        if _time_left() < 5.0:
            break
        try:
            energy, positions = relax(config.copy(), 'morse', rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Iterative refinement: perturb best and re-relax
    if best_positions is not None:
        seed_offset = 30
        while _time_left() > 3.0:
            pert = _perturb(best_positions, scale=0.15, seed=seed_offset)
            try:
                energy, positions = relax(pert.copy(), 'morse', rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            seed_offset += 1
    
    if best_positions is None:
        best_positions = fcc
    
    return best_positions
# EVOLVE-BLOCK-END
