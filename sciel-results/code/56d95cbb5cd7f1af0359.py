"""Dzugutov cluster global minima: icosahedral seeds + parallel basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_TOTAL_BUDGET = 300.0
_PER_INSTANCE_BUDGET = 300.0 / 6.0  # ~50s each, with safety
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining_for_instance():
    return max(1.0, _PER_INSTANCE_BUDGET - (_elapsed() % _PER_INSTANCE_BUDGET) * 0.5)


def _dzugutov_energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    E = 0.0
    mask_a = dist < a
    mask_b = (dist < b) & ~mask_a
    
    if np.any(mask_a):
        r = dist[mask_a]
        E += np.sum(A * (r**(-m) - B) * np.exp(c / (r - a)))
    if np.any(mask_b):
        r = dist[mask_b]
        E += np.sum(B * np.exp(d / (r - b)))
    return E


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # normalize to unit radius
    return verts


def _make_icosahedral_seed(n, rng, radius_scale=1.0):
    """Generate an icosahedral-like seed with n atoms.
    
    Layer 1: center atom
    Layer 2: 12 icosahedron vertices
    Layer 3+: additional shells with icosahedral symmetry
    """
    ico = _icosahedron_vertices()
    base_spacing = 1.1  # Dzugutov equilibrium spacing
    
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms at icosahedron vertices
    r1 = base_spacing * radius_scale
    for v in ico:
        if len(points) < n:
            points.append(r1 * v)
    
    # Shell 2: 30 atoms at edge midpoints of icosahedron (approx)
    # Use a slightly larger radius with icosahedral directions
    if len(points) < n:
        # Generate more directions by combining icosahedron vertices
        r2 = base_spacing * 1.55 * radius_scale
        # icosahedron has 6 vertices per face, 20 faces -> but we need unique directions
        # Use scaled icosahedron + rotated versions
        angles = [0, np.pi/5, 2*np.pi/5, 3*np.pi/5, 4*np.pi/5]
        for ax in range(3):
            for ang in angles:
                if len(points) >= n:
                    break
                # Rotate icosahedron vertex by angle around each axis
                v = ico[0].copy()
                if ax == 0:
                    v[1], v[2] = v[1]*np.cos(ang) - v[2]*np.sin(ang), v[1]*np.sin(ang) + v[2]*np.cos(ang)
                elif ax == 1:
                    v[0], v[2] = v[0]*np.cos(ang) - v[2]*np.sin(ang), v[0]*np.sin(ang) + v[2]*np.cos(ang)
                else:
                    v[0], v[1] = v[0]*np.cos(ang) - v[1]*np.sin(ang), v[0]*np.sin(ang) + v[1]*np.cos(ang)
                v = v / np.linalg.norm(v)
                points.append(r2 * v)
    
    # Fill remaining with random compact positions
    while len(points) < n:
        r_max = base_spacing * (n ** (1/3)) * 0.8
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            # Check minimum distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * base_spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                # Just add with small random perturbation
                base = points[rng.integers(len(points))]
                p = base + rng.normal(0, 0.3, 3)
                points.append(p)
    
    return np.array(points[:n])


def _make_deformed_icosahedral_seed(n, rng, radius_scale=1.0, deform=0.15):
    """Create a deformed icosahedral seed for diversity."""
    seed = _make_icosahedral_seed(n, rng, radius_scale)
    # Apply random deformation
    noise = rng.normal(0, deform, seed.shape)
    seed = seed + noise
    # Re-center
    seed -= seed.mean(axis=0)
    return seed


def _make_random_compact_seed(n, rng, spacing=1.1):
    """Random compact seed with minimum distance constraint."""
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all, add remaining with perturbation
    while len(points) < n:
        if points:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.4, 3)
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _relax_and_kick(pos, potential, parameter, rng, max_kicks=3, kick_scale=0.4):
    """Relax, then do kick-and-relax cycles to escape local minima."""
    best_energy = np.inf
    best_pos = pos.copy()
    
    for kick_iter in range(max_kicks + 1):
        try:
            energy, relaxed = relax(pos, potential, parameter)
        except Exception:
            break
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
        
        if kick_iter == max_kicks:
            break
        
        # Kick: add random displacement
        kick = rng.normal(0, kick_scale, relaxed.shape)
        pos = relaxed + kick
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster using icosahedral seeds + parallel basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 1000 + 42)
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Standard icosahedral seed
    seeds.append(_make_icosahedral_seed(n, rng, radius_scale=1.0))
    
    # 2. Slightly compressed icosahedral
    seeds.append(_make_icosahedral_seed(n, rng, radius_scale=0.92))
    
    # 3. Slightly expanded icosahedral
    seeds.append(_make_icosahedral_seed(n, rng, radius_scale=1.08))
    
    # 4. Deformed icosahedral
    seeds.append(_make_deformed_icosahedral_seed(n, rng, radius_scale=1.0, deform=0.12))
    
    # 5. Deformed icosahedral (different deformation)
    seeds.append(_make_deformed_icosahedral_seed(n, rng, radius_scale=0.95, deform=0.2))
    
    # 6. Random compact
    seeds.append(_make_random_compact_seed(n, rng))
    
    # 7. Another random compact with different seed
    rng2 = np.random.default_rng(n * 777 + 13)
    seeds.append(_make_random_compact_seed(n, rng2))
    
    # 8. Icosahedral with different rotation
    ico = _icosahedron_vertices()
    # Apply a random rotation
    theta = rng.uniform(0, 2*np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    rotated_ico = ico @ R.T
    # Rebuild seed with rotated icosahedron
    base_spacing = 1.1
    points = [np.zeros(3)]
    r1 = base_spacing
    for v in rotated_ico:
        if len(points) < n:
            points.append(r1 * v)
    # Fill rest
    while len(points) < n:
        r_max = base_spacing * (n ** (1/3)) * 0.8
        p = rng.uniform(-r_max, r_max, 3)
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55 * base_spacing:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.3, 3)
            points.append(p)
    seeds.append(np.array(points[:n]))
    
    # Check time budget
    elapsed = _elapsed()
    remaining = _TOTAL_BUDGET - elapsed
    
    # Determine how many workers we can afford
    # Each relaxation takes roughly 0.5-2s for these sizes
    # We have ~50s per instance, so we can do many relaxations
    n_workers = min(4, len(seeds))
    
    best_energy = np.inf
    best_pos = seeds[0].copy()
    
    # Parallel relaxation with kick-and-relax
    def _process_seed(seed_idx):
        nonlocal best_energy, best_pos
        seed = seeds[seed_idx].copy()
        
        # Multiple kick-and-relax rounds with different kick scales
        local_best_energy = np.inf
        local_best_pos = seed.copy()
        
        kick_scales = [0.3, 0.5, 0.8]
        for ks in kick_scales:
            try:
                e, p = _relax_and_kick(seed, potential, parameter, 
                                       np.random.default_rng(seed_idx * 1000 + int(ks*100)),
                                       max_kicks=2, kick_scale=ks)
            except Exception:
                continue
            
            if e < local_best_energy:
                local_best_energy = e
                local_best_pos = p
        
        with _THREAD_LOCK:
            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_pos = local_best_pos.copy()
        
        return local_best_energy, local_best_pos
    
    # Use ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            # Check time
            if _elapsed() > _TOTAL_BUDGET - 5:
                break
            futures.append(executor.submit(_process_seed, i))
        
        for fut in as_completed(futures):
            try:
                fut.result(timeout=30)
            except Exception:
                pass
    
    # If we have time, do additional basin hopping on the best result
    if _elapsed() < _TOTAL_BUDGET - 10 and best_energy < np.inf:
        rng_bh = np.random.default_rng(n * 31337)
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        # Basin hopping: kick, relax, accept if lower
        for bh_iter in range(15):
            if _elapsed() > _TOTAL_BUDGET - 5:
                break
            
            kick = rng_bh.normal(0, 0.5, current_pos.shape)
            kicked = current_pos + kick
            
            try:
                e, relaxed = relax(kicked, potential, parameter)
            except Exception:
                continue
            
            if e < current_energy:
                current_energy = e
                current_pos = relaxed.copy()
                with _THREAD_LOCK:
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()
    
    # Final validation: ensure minimum distance constraint
    # If any atoms are too close, nudge them apart
    pos = best_pos.copy()
    for _ in range(50):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= 0.5:
            break
        # Nudge apart
        i, j = min_pair
        direction = pos[i] - pos[j]
        if np.linalg.norm(direction) < 1e-10:
            direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        pos[i] += 0.01 * direction
        pos[j] -= 0.01 * direction
    
    return pos
# EVOLVE-BLOCK-END
