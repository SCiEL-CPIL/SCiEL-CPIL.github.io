"""Sutton-Chen transition-metal cluster global minima: structure prediction via rigid cores + adaptive relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 15.0

def _time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

# --- Structural initialization helpers ---

def _generate_lattice_points(lattice_type, n, spacing=0.72):
    """Generate points from a lattice structure filled into a sphere."""
    if n == 0:
        return np.zeros((0, 3))
    
    a = spacing
    if lattice_type == 'fcc':
        atoms_per_vol = 4.0 / (a**3)
    elif lattice_type == 'bcc':
        atoms_per_vol = 2.0 / (a**3)
    else:
        atoms_per_vol = 3.0 / (a**3)
    
    volume_needed = n / atoms_per_vol
    R = (3.0 * volume_needed / (4.0 * np.pi)) ** (1.0/3.0)
    
    max_coord = int(np.ceil(R / a)) + 2
    candidates = []
    
    if lattice_type == 'fcc':
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                for iz in range(-max_coord, max_coord + 1):
                    candidates.append(np.array([ix * a, iy * a, iz * a]))
                    candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a]))
                    candidates.append(np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a]))
                    candidates.append(np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    elif lattice_type == 'bcc':
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                for iz in range(-max_coord, max_coord + 1):
                    candidates.append(np.array([ix * a, iy * a, iz * a]))
                    candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    else:
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                for iz in range(-max_coord, max_coord + 1):
                    candidates.append(np.array([ix * a, iy * a, iz * a]))
    
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return np.zeros((n, 3))
        
    dists = np.linalg.norm(candidates, axis=1)
    inside = candidates[dists <= R]
    
    if len(inside) >= n:
        dists_inside = np.linalg.norm(inside, axis=1)
        order = np.argsort(dists_inside)
        return inside[order[:n]]
    else:
        points = list(inside)
        while len(points) < n:
            p = np.random.uniform(-R, R, 3)
            points.append(p)
        return np.array(points)

def _random_compact(n, spacing=0.72, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    max_attempts = n * 1000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.4 * spacing:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    
    return np.array(points)

def _icosahedron(n, spacing=0.72):
    """Generate an icosahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v)
    
    if len(positions) < n:
        octa = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        octa = octa / np.linalg.norm(octa, axis=1, keepdims=True) * spacing * 1.6
        for v in octa:
            if len(positions) >= n:
                break
            positions.append(v)
            
    if len(positions) < n:
        cube = np.array(list(product([-1, 1], repeat=3)))
        cube = cube / np.linalg.norm(cube, axis=1, keepdims=True) * spacing * 2.2
        for v in cube:
            if len(positions) >= n:
                break
            positions.append(v)

    while len(positions) < n:
        p = np.random.uniform(-spacing * 2.5, spacing * 2.5, 3)
        positions.append(p)
    
    return np.array(positions[:n])

def _validate(positions):
    """Check if positions are valid."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _relax_one(start, kind):
    """Relax a single starting configuration. Returns (energy, positions) or (inf, None)."""
    try:
        if not _validate(start):
            return (np.inf, None)
        energy, positions = relax(start, 'sutton_chen', kind)
        if not np.all(np.isfinite(positions)):
            return (np.inf, None)
        if not _validate(positions):
            return (np.inf, None)
        return (energy, positions)
    except Exception:
        return (np.inf, None)

def _build_from_core(core, n, kind, rng):
    """
    Structure prediction strategy:
    1. Take a small known/solved core structure (or generate one).
    2. Add atoms one by one at low-energy sites (surface/interstitial).
    3. Relax the whole thing.
    This is a heuristic for hierarchical growth.
    """
    if len(core) >= n:
        # If core is already big enough, just relax a subset or return as is (shouldn't happen if we pick smaller cores)
        return core[:n]
    
    current = core.copy()
    spacing = 0.72
    R_core = np.max(np.linalg.norm(current, axis=1)) if len(current) > 0 else 0
    
    # Generate candidate sites for new atoms
    # We try to place new atoms on the "surface" of the core
    # A simple heuristic: place atoms on a slightly larger sphere and relax
    
    attempts = 0
    max_attempts = (n - len(core)) * 20
    
    while len(current) < n and attempts < max_attempts:
        # Generate a random point on a sphere slightly larger than the core
        r = R_core + spacing * rng.uniform(0.5, 1.5)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        # Check distance to existing atoms
        if len(current) > 0:
            dists = np.linalg.norm(current - p, axis=1)
            if np.min(dists) >= 0.4 * spacing:
                current = np.vstack([current, p])
        else:
            current = np.vstack([current, p])
        
        attempts += 1
    
    # If we couldn't fill it, add random points
    while len(current) < n:
        p = rng.uniform(-R_core - spacing, R_core + spacing, 3)
        current = np.vstack([current, p])
    
    return current

def _get_small_core(n, kind, rng):
    """
    Get a small core structure for seeding.
    For small n, use known structures (icosahedron for 13, etc.)
    For larger n, we can't easily get the exact global minimum, but we can use
    a relaxed small cluster as a core.
    """
    # We don't have the global minima for all sizes, but we can approximate
    # by relaxing a small random cluster or using a lattice truncation.
    # For this strategy, we assume that small clusters (n=13-20) are often icosahedral.
    core_size = min(13, n)  # Use 13-atom icosahedron as a common core
    
    if core_size <= 1:
        return np.zeros((core_size, 3))
    
    # Generate an icosahedron for the core
    core = _icosahedron(core_size, 0.72)
    
    # Relax the core to get a better starting point
    try:
        energy, core_relaxed = relax(core, 'sutton_chen', kind)
        if np.all(np.isfinite(core_relaxed)) and _validate(core_relaxed):
            return core_relaxed
    except:
        pass
    
    return core

def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 67890)
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. FCC-like bulk
    try:
        fcc = _generate_lattice_points('fcc', n, 0.72)
        if len(fcc) == n and _validate(fcc):
            starts.append(fcc)
    except:
        pass
        
    # 2. BCC-like bulk
    try:
        bcc = _generate_lattice_points('bcc', n, 0.72)
        if len(bcc) == n and _validate(bcc):
            starts.append(bcc)
    except:
        pass
        
    # 3. Icosahedral
    try:
        icosa = _icosahedron(n, 0.72)
        if len(icosa) == n and _validate(icosa):
            starts.append(icosa)
    except:
        pass
        
    # 4. Structure prediction: Build from small core
    try:
        core = _get_small_core(n, kind, rng)
        if len(core) > 0 and len(core) < n:
            grown = _build_from_core(core, n, kind, rng)
            if len(grown) == n and _validate(grown):
                starts.append(grown)
    except:
        pass
        
    # 5. Random compact (multiple)
    for _ in range(10):
        try:
            rand = _random_compact(n, 0.72, rng)
            if len(rand) == n and _validate(rand):
                starts.append(rand)
        except:
            pass
            
    # 6. Deformed FCC/BCC
    if len(starts) > 0:
        for base_idx in [0, 1]:
            if base_idx < len(starts):
                for scale in [0.05, 0.1, 0.2]:
                    deformed = starts[base_idx] + rng.normal(0, scale, starts[base_idx].shape)
                    if _validate(deformed):
                        starts.append(deformed)

    if not starts:
        starts = [_random_compact(n, 0.72, rng)]
    
    # Parallel relaxation using up to 4 threads
    best_energy = np.inf
    best_positions = starts[0]
    
    # First, try the structural starts in parallel
    max_workers = min(4, len(starts))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_relax_one, start, kind): i for i, start in enumerate(starts)}
        for future in as_completed(futures):
            if _time_left() < 5.0:
                break
            energy, positions = future.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    # If we have time, do more random restarts in parallel
    attempts = 0
    max_attempts = 40
    while _time_left() > 15.0 and attempts < max_attempts:
        # Generate batch of random starts
        batch_size = min(4, max_attempts - attempts)
        batch = []
        for _ in range(batch_size):
            start = _random_compact(n, 0.72, rng)
            if _validate(start):
                batch.append(start)
        
        if not batch:
            attempts += batch_size
            continue
        
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            futures = {executor.submit(_relax_one, start, kind): i for i, start in enumerate(batch)}
            for future in as_completed(futures):
                if _time_left() < 5.0:
                    break
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
        
        attempts += batch_size
    
    # Final polish: re-relax the best position with potentially tighter settings
    if _time_left() > 8.0 and np.isfinite(best_energy):
        try:
            energy, positions = relax(best_positions, 'sutton_chen', kind)
            if np.all(np.isfinite(positions)) and _validate(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
        except Exception:
            pass
            
    # Final validation
    if not _validate(best_positions):
        best_positions = _random_compact(n, 0.72, rng)
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
    
    return best_positions
# EVOLVE-BLOCK-END
