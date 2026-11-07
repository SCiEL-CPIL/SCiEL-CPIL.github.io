"""Sutton-Chen transition-metal cluster global minima: diverse structural starts + adaptive relaxation."""
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
        
    # 4. Random compact (multiple)
    for _ in range(10):
        try:
            rand = _random_compact(n, 0.72, rng)
            if len(rand) == n and _validate(rand):
                starts.append(rand)
        except:
            pass
            
    # 5. Deformed FCC/BCC
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
