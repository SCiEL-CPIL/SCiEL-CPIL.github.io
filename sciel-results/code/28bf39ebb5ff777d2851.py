"""Morse cluster global minima: lattice-initialized multi-restart with time budgeting."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 295.0  # Safety margin before 300s kill
_instance_start = None
_instance_deadline = None


def _fcc_points(n, spacing=1.0):
    """Generate n points in FCC-like arrangement centered at origin."""
    # FCC lattice vectors
    a = spacing
    # Generate points from FCC: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0) in cubic cells
    # Simpler: generate a dense grid and pick closest to origin
    r_max = a * (n ** (1.0/3.0) + 2)
    step = a * 0.5  # half-spacing for FCC-like density
    
    # Generate candidate points
    n_side = int(np.ceil(2 * r_max / step)) + 2
    coords = []
    for i in range(-n_side//2, n_side//2 + 1):
        for j in range(-n_side//2, n_side//2 + 1):
            for k in range(-n_side//2, n_side//2 + 1):
                x = i * step
                y = j * step
                z = k * step
                # FCC: only include if (i+j+k) is even for one sublattice, or use all
                # Actually just use a simple cubic grid with spacing ~ a/sqrt(2) for FCC-like density
                if x*x + y*y + z*z <= r_max*r_max:
                    coords.append([x, y, z])
    
    coords = np.array(coords)
    if len(coords) == 0:
        return np.zeros((n, 3))
    
    # Sort by distance from origin, take closest n
    dists = np.sum(coords**2, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx].copy()
    
    # Center
    points -= points.mean(axis=0)
    
    # Ensure minimum distance of 0.5
    points = _enforce_min_dist(points, min_dist=0.5, rng_seed=42)
    
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
    # Simple iterative push-apart
    for iteration in range(50):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    push = (min_dist - dist) / dist * 0.5
                    points[i] += d * push
                    points[j] -= d * push
                    moved = True
                elif dist <= 1e-10:
                    # Random nudge
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
    """Find low-energy Morse cluster configurations using lattice init + multi-restart relaxation."""
    global _instance_start, _instance_deadline
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    _instance_start = time.time()
    # Per-instance budget: leave 5s margin, split remaining among instances
    # We don't know how many instances remain, so use a fixed per-instance budget
    # Total 300s for 6 instances = 50s each, use 45s to be safe
    _instance_deadline = _instance_start + 45.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    initial_configs = []
    
    # FCC lattice
    fcc = _fcc_points(n, spacing=1.0)
    initial_configs.append(("fcc", fcc))
    
    # BCC lattice
    bcc = _bcc_points(n, spacing=1.0)
    initial_configs.append(("bcc", bcc))
    
    # Random compact (multiple seeds)
    for seed in [0, 1, 2, 3]:
        rc = _random_compact(n, spacing=1.0, seed=seed)
        initial_configs.append((f"rand{seed}", rc))
    
    # Perturbed FCC
    for seed in [10, 11, 12]:
        pf = _perturb(fcc, scale=0.3, seed=seed)
        initial_configs.append((f"pfcc{seed}", pf))
    
    # Perturbed BCC
    for seed in [20, 21]:
        pb = _perturb(bcc, scale=0.3, seed=seed)
        initial_configs.append((f"pbcc{seed}", pb))
    
    # Relax each configuration
    for name, config in initial_configs:
        # Check time
        if time.time() > _instance_deadline:
            break
        
        try:
            energy, positions = relax(config.copy(), 'morse', rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time, do more perturbations of the best
    if best_positions is not None and time.time() < _instance_deadline - 2.0:
        for seed in range(30, 60):
            if time.time() > _instance_deadline - 1.0:
                break
            pert = _perturb(best_positions, scale=0.15, seed=seed)
            try:
                energy, positions = relax(pert, 'morse', rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return the FCC config
        best_positions = fcc
    
    return best_positions
# EVOLVE-BLOCK-END
