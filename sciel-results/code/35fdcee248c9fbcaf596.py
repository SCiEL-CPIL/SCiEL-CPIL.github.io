import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0
_LOCK = threading.Lock()
_best_energy = float('inf')
_best_pos = None

def _time_left():
    return _DEADLINE - time.time()

def _get_icosahedron_verts():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _generate_mackay(n, r1=1.0, r2=1.6):
    """Generate n atoms for a truncated Mackay icosahedron."""
    verts = _get_icosahedron_verts()
    atoms = [np.zeros(3)]
    atoms.extend((verts * r1).tolist())
    
    if n <= 13:
        return np.array(atoms[:n])
    
    # Find edges and faces of icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i,j] = np.linalg.norm(verts[i] - verts[j])
    min_dist = np.min(dists[dists > 1e-6])
    
    edges = []
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist, rtol=1e-3):
                edges.append((i, j))
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (np.isclose(dists[i,j], min_dist, rtol=1e-3) and
                    np.isclose(dists[j,k], min_dist, rtol=1e-3) and
                    np.isclose(dists[i,k], min_dist, rtol=1e-3)):
                    faces.append((i, j, k))
    
    # Edge midpoints (30) and face centers (12)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid_dir = mid / mid_norm
            atoms.append(mid_dir * r2)
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-10:
            center_dir = center / center_norm
            atoms.append(center_dir * r2)
    
    return np.array(atoms[:n])

def _random_seed(n, scale, seed):
    """Generate random seed with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = scale
    points = []
    min_dist = 0.55
    max_attempts = n * 300
    for _ in range(max_attempts):
        if len(points) >= n: break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) < radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _relax_and_evaluate(pos, n, potential, parameter):
    """Relax structure and evaluate energy with validity check."""
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        # Check validity
        min_dist = 0.5
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(relaxed_pos[i] - relaxed_pos[j]) < min_dist:
                    # Fix by moving apart
                    fixed_pos = relaxed_pos.copy()
                    for i2 in range(n):
                        for j2 in range(i2+1, n):
                            dist = np.linalg.norm(fixed_pos[i2] - fixed_pos[j2])
                            if dist < min_dist:
                                direction = (fixed_pos[i2] - fixed_pos[j2]) / max(dist, 1e-10)
                                shift = (min_dist - dist) / 2 + 0.01
                                fixed_pos[i2] += direction * shift
                                fixed_pos[j2] -= direction * shift
                    try:
                        energy, fixed_pos = relax(fixed_pos, potential, parameter)
                        return energy, fixed_pos
                    except:
                        return float('inf'), pos
                    break
            else:
                continue
            break
        return energy, relaxed_pos
    except Exception:
        return float('inf'), pos

def _kick_relax_worker(seed_pos, n, potential, parameter, worker_id, kick_scales, n_kicks):
    """Kick-and-relax worker that cycles through different kick scales."""
    global _best_energy, _best_pos
    
    rng = np.random.default_rng(worker_id * 1000 + 42)
    current_pos = seed_pos.copy()
    current_energy, current_pos = _relax_and_evaluate(current_pos, n, potential, parameter)
    local_best_energy = current_energy
    local_best_pos = current_pos.copy()
    
    for kick_idx in range(n_kicks):
        if _time_left() < 3.0:
            break
        
        # Cycle through kick scales
        scale = kick_scales[kick_idx % len(kick_scales)]
        
        # Occasionally adopt the global best
        if kick_idx > 0 and kick_idx % 5 == 0:
            with _LOCK:
                if _best_pos is not None and _best_energy < current_energy * 0.95:
                    current_pos = _best_pos.copy()
                    current_energy, current_pos = _relax_and_evaluate(
                        current_pos, n, potential, parameter)
                    if current_energy < local_best_energy:
                        local_best_energy = current_energy
                        local_best_pos = current_pos.copy()
        
        # Multi-atom kick for diversity
        if rng.random() < 0.3:
            # Kick multiple atoms
            n_kick = max(1, n // 4)
            kick_atoms = rng.choice(n, size=n_kick, replace=False)
            perturbed = current_pos.copy()
            perturbed[kick_atoms] += rng.normal(0, scale, (n_kick, 3))
        else:
            # Single atom kick
            kick_atom = rng.integers(n)
            perturbed = current_pos.copy()
            perturbed[kick_atom] += rng.normal(0, scale * 1.5, 3)
        
        new_energy, new_pos = _relax_and_evaluate(perturbed, n, potential, parameter)
        
        # Accept if better or with Metropolis criterion
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < local_best_energy:
                local_best_energy = new_energy
                local_best_pos = new_pos.copy()
                with _LOCK:
                    if new_energy < _best_energy:
                        _best_energy = new_energy
                        _best_pos = new_pos.copy()
        else:
            delta = new_energy - current_energy
            if delta < 1.0:
                temp = 0.1 * (1 - kick_idx / max(1, n_kicks))
                if rng.random() < np.exp(-delta / max(temp, 0.01)):
                    current_energy = new_energy
                    current_pos = new_pos
    
    with _LOCK:
        if local_best_energy < _best_energy:
            _best_energy = local_best_energy
            _best_pos = local_best_pos.copy()
    
    return local_best_energy, local_best_pos

def construct(instance):
    global _best_energy, _best_pos
    _best_energy = float('inf')
    _best_pos = None
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Standard Mackay icosahedron
    mackay_pos = _generate_mackay(n, r1=1.0, r2=1.6)
    seeds.append(mackay_pos)
    
    # 2. Compressed icosahedron
    seeds.append(_generate_mackay(n, r1=0.85, r2=1.35))
    
    # 3. Expanded icosahedron
    seeds.append(_generate_mackay(n, r1=1.15, r2=1.85))
    
    # 4. Deformed icosahedron
    rng_def = np.random.default_rng(123)
    deformed = mackay_pos + rng_def.normal(0, 0.1, mackay_pos.shape)
    seeds.append(deformed)
    
    # 5. Random seed
    est_r = (3 * n / (4 * np.pi)) ** (1/3) * 0.8
    seeds.append(_random_seed(n, est_r, 44))
    
    # 6. Another random with different scale
    seeds.append(_random_seed(n, est_r * 1.3, 46))
    
    # Kick scales: small, medium, large
    kick_scales = [0.1, 0.2, 0.35, 0.5]
    
    # Determine how many kicks each worker can do based on time
    # Estimate: each kick-relax takes ~0.1-0.3s for n~50
    # With 290s total and 4 workers, each gets ~70s
    # Conservative estimate: 150 kicks per worker
    n_kicks_per_worker = 150
    
    # Launch 4 parallel workers with different seeds
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            seed = seeds[i % len(seeds)]
            futures.append(executor.submit(
                _kick_relax_worker, seed, n, potential, parameter, i, 
                kick_scales, n_kicks_per_worker))
        
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
    
    # Final polish with small kicks if time remains
    if _best_pos is not None and _time_left() > 10.0:
        polish_scales = [0.05, 0.08, 0.1]
        _kick_relax_worker(_best_pos, n, potential, parameter, 99, 
                          polish_scales, 50)
    
    if _best_pos is None:
        _best_pos = seeds[0]
    
    return _best_pos
