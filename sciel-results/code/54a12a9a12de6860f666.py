# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s safety margin before 330s kill

def _remaining():
    return _DEADLINE - time.time()

def _get_params(kind):
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    return 12, 6, 144.41

def _get_spacing(kind):
    n, m, c = _get_params(kind)
    return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))

def _fcc_positions(n, a=1.0):
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    dist = np.linalg.norm(pos)
                    sites.append((dist, pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _hcp_positions(n, a=1.0):
    c = a * np.sqrt(8/3)
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c / 2
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([i * a, j * a * np.sqrt(3)/2, z])
                sites.append((np.linalg.norm(pos), pos))
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([(i + 0.5) * a, (j + 1/6) * a * np.sqrt(3), z + c/2])
                sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _bcc_positions(n, a=1.0):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    dist = np.linalg.norm(pos)
                    sites.append((dist, pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _icosahedral_positions(n, a=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * a / np.sqrt(1 + phi**2)
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v)
    if n > 13:
        faces = [
            [0, 1, 8], [0, 2, 9], [1, 3, 8], [2, 4, 9],
            [1, 0, 5], [3, 2, 6], [0, 4, 5], [2, 6, 7],
            [5, 8, 10], [6, 9, 11], [8, 1, 10], [9, 3, 11],
            [10, 5, 7], [11, 6, 4], [8, 10, 0], [9, 11, 2],
            [5, 7, 1], [6, 4, 3], [10, 8, 0], [11, 9, 2]
        ]
        for f in faces:
            if len(positions) >= n:
                break
            fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            positions.append(fc)
        if n > 33:
            edges = []
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(verts[i] - verts[j])
                    if d < 2.1 * a / np.sqrt(1+phi**2):
                        edges.append((i, j))
            for e in edges:
                if len(positions) >= n:
                    break
                em = (verts[e[0]] + verts[e[1]]) / 2.0
                positions.append(em)
    while len(positions) < n:
        idx = np.random.randint(0, 12)
        pos = verts[idx] + np.random.normal(0, 0.1 * a, 3)
        positions.append(pos)
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    if min_dist is None:
        min_dist = spacing * 0.9
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _molten_cluster(n, spacing, seed, temp_scale=1.0):
    """Molten-like cluster: random positions with soft repulsion."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.2
    points = []
    min_d = spacing * 0.5 * temp_scale
    max_attempts = 10000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_d for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _deformed_fcc_positions(n, a=1.0, seed=0):
    """FCC with random per-atom deformation."""
    rng = np.random.default_rng(seed)
    base = _fcc_positions(n, a)
    noise = rng.normal(0, a * 0.15, base.shape)
    return base + noise

def _incremental_energy(pos, new_atom, kind):
    """Compute the incremental energy contribution of adding new_atom to cluster pos."""
    n_exp, m_exp, c = _get_params(kind)
    if len(pos) == 0:
        return 0.0
    d = pos - new_atom
    r2 = np.sum(d * d, axis=1)
    r2 = np.maximum(r2, 1e-12)
    r_inv_n = r2 ** (-n_exp / 2.0)
    r_inv_m = r2 ** (-m_exp / 2.0)
    pair_sum = np.sum(r_inv_n)
    coh_sum = np.sum(r_inv_m)
    # New atom's contribution: 1/2 * pair_sum - c * sqrt(coh_sum)
    # Plus the change in existing atoms' cohesion (they gain one more neighbor)
    # Existing atoms' pair part unchanged, but cohesion increases
    # Old cohesion for atom i: sqrt(sum_{j!=i} r_ij^-m)
    # New cohesion for atom i: sqrt(sum_{j!=i} r_ij^-m + r_i,new^-m)
    # This is complex; approximate by just the new atom's direct contribution
    return 0.5 * pair_sum - c * np.sqrt(coh_sum)

def _gradient_guided_insertion(n, kind, spacing, seed=0):
    """
    Build cluster atom-by-atom, placing each new atom to minimize
    incremental energy. This produces thermodynamically consistent seeds.
    """
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    
    for count in range(1, n):
        if _remaining() < 5.0:
            break
        # Generate candidate positions for the new atom
        # Sample on a shell around existing cluster + random positions
        radius = spacing * (count + 1) ** (1.0/3.0) * 1.2
        candidates = []
        
        # Candidates near existing atoms (surface growth)
        for p in positions:
            for _ in range(3):
                direction = rng.normal(0, 1, 3)
                direction /= (np.linalg.norm(direction) + 1e-10)
                new_p = p + direction * spacing * (0.8 + 0.4 * rng.random())
                candidates.append(new_p)
        
        # Random candidates
        for _ in range(5):
            new_p = rng.uniform(-radius, radius, 3)
            candidates.append(new_p)
        
        # Evaluate incremental energy for each candidate
        best_e = np.inf
        best_p = candidates[0]
        for cand in candidates:
            # Enforce minimum distance
            if len(positions) > 0:
                d = np.linalg.norm(np.array(positions) - cand, axis=1)
                if np.min(d) < 0.35:
                    continue
            e = _incremental_energy(np.array(positions), cand, kind)
            if e < best_e:
                best_e = e
                best_p = cand
        
        positions.append(best_p)
    
    # Pad if we ran out of time
    while len(positions) < n:
        p = rng.uniform(-spacing * n**(1/3), spacing * n**(1/3), 3)
        positions.append(p)
    
    return np.array(positions[:n])

def _min_distance_check(pos):
    n = len(pos)
    if n < 2:
        return np.inf
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _valid_result(energy, pos):
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance_check(pos) < 0.35:
        return False
    return True

def _relax_task(args):
    pos, potential, kind = args
    try:
        energy, relaxed_pos = relax(pos, potential, kind)
        if _valid_result(energy, relaxed_pos):
            return energy, relaxed_pos
        else:
            return np.inf, relaxed_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    global _START_TIME, _DEADLINE
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    spacing = _get_spacing(kind)
    
    # Generate diverse starting points using multiple structural families
    candidates = []
    
    # Geometric seeds
    for gen_func in [_fcc_positions, _hcp_positions, _bcc_positions, _icosahedral_positions]:
        try:
            pos = gen_func(n, spacing)
            if len(pos) == n:
                candidates.append(pos)
        except Exception:
            pass
    
    # Deformed geometric seeds
    for seed in [10, 20]:
        try:
            pos = _deformed_fcc_positions(n, spacing, seed=seed)
            candidates.append(pos)
        except Exception:
            pass
    
    # Molten seeds
    for seed, ts in [(50, 0.8), (60, 1.2)]:
        try:
            pos = _molten_cluster(n, spacing, seed=seed, temp_scale=ts)
            candidates.append(pos)
        except Exception:
            pass
    
    # Gradient-guided insertion seeds (key structural innovation)
    for seed in [100, 200, 300]:
        try:
            pos = _gradient_guided_insertion(n, kind, spacing, seed=seed)
            if len(pos) == n:
                candidates.append(pos)
        except Exception:
            pass
    
    # Random clusters
    for seed, md in [(42, None), (123, None), (456, spacing * 0.7), (789, spacing * 1.2)]:
        try:
            pos = _random_cluster(n, spacing, seed=seed, min_dist=md)
            candidates.append(pos)
        except Exception:
            pass
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Parallel relaxation of all initial candidates
    if _remaining() > 20.0:
        tasks = [(pos, potential, kind) for pos in candidates]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_task, task): task for task in tasks}
            for future in as_completed(futures):
                if _remaining() < 5.0:
                    break
                try:
                    energy, pos = future.result()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos
                except Exception:
                    pass

    # Fallback sequential if parallel failed
    if best_positions is None:
        for pos in candidates:
            if _remaining() < 10.0:
                break
            try:
                energy, relaxed_pos = relax(pos, potential, kind)
                if _valid_result(energy, relaxed_pos) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass

    # Phase 2: Basin hopping with simulated annealing acceptance
    if best_positions is not None and _remaining() > 30.0:
        rng = np.random.default_rng(999)
        iteration = 0
        no_improve = 0
        # Simulated annealing: accept worse solutions early with probability exp(-dE/T)
        T0 = spacing * 0.5  # initial "temperature" in energy units (approximate)
        
        while _remaining() > 40.0:
            iteration += 1
            # Decay kick size and temperature
            kick = spacing * 0.5 * (0.85 ** iteration)
            T = T0 * (0.9 ** iteration)
            
            # Generate multiple perturbed candidates
            n_kicks = 4
            tasks = []
            for _ in range(n_kicks):
                noise = rng.normal(0, kick, best_positions.shape)
                perturbed = best_positions + noise
                tasks.append((perturbed, potential, kind))
            
            best_in_round = best_energy
            best_pos_round = best_positions
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_relax_task, task): task for task in tasks}
                for future in as_completed(futures):
                    if _remaining() < 10.0:
                        break
                    try:
                        energy, pos = future.result()
                        if energy < best_in_round:
                            best_in_round = energy
                            best_pos_round = pos
                    except Exception:
                        pass
            
            # Simulated annealing acceptance
            if best_in_round < best_energy:
                best_energy = best_in_round
                best_positions = best_pos_round
                no_improve = 0
            else:
                dE = best_in_round - best_energy
                if T > 1e-6 and rng.random() < np.exp(-dE / max(T, 1e-10)):
                    best_positions = best_pos_round
                no_improve += 1
                if no_improve >= 5:
                    break
    
    # Phase 3: Final polish
    if best_positions is not None and _remaining() > 10.0:
        try:
            energy, relaxed_pos = relax(best_positions, potential, kind)
            if _valid_result(energy, relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        try:
            pos = _random_cluster(n, spacing, 12345, min_dist=0.4)
            energy, relaxed_pos = relax(pos, potential, kind)
            if _valid_result(energy, relaxed_pos):
                best_positions = relaxed_pos
            else:
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]
        except Exception:
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i * 0.8, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
