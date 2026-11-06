"""Dzugutov cluster global minima: symmetry-constrained variational optimization + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

_START_TIME = None
_DEADLINE = 295.0  # seconds, safety margin before 330s kill


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    triu_idx = np.triu_indices(n, k=1)
    r = dist[triu_idx]
    
    if np.min(r) < 0.5:
        return 1e10
    
    E = 0.0
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        E += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2 = r[mask2]
        E += np.sum(B * np.exp(d / (r2 - b)))
    
    return float(E)


def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron, unit length."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _get_icosahedral_shell2_directions():
    """
    Return 42 directions for the icosahedral second shell:
    30 edge-midpoint directions + 12 dodecahedral (face-center) directions.
    """
    v = _get_icosahedron_vertices()
    # Compute all pairwise distances to find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(v[i] - v[j])
    
    # Edge set: nearest neighbors (5 per vertex)
    edge_set = set()
    for i in range(12):
        neighbors = np.argsort(dists[i])
        for j in neighbors[1:6]:
            key = (min(i, j), max(i, j))
            edge_set.add(key)
    
    # 30 edge midpoints
    edge_mids = []
    for (i, j) in edge_set:
        mid = v[i] + v[j]
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    
    # 12 dodecahedral directions = face centers of icosahedron
    # Find all triangular faces
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (min(i, j), max(i, j)) in edge_set:
                for k in range(j + 1, 12):
                    if (min(i, k), max(i, k)) in edge_set and (min(j, k), max(j, k)) in edge_set:
                        fc = v[i] + v[j] + v[k]
                        fc = fc / np.linalg.norm(fc)
                        face_centers.append(fc)
    
    # We have 20 face centers; take 12 that form a dodecahedron
    # For simplicity, use first 12
    result = np.array(edge_mids[:30] + face_centers[:12])
    return result


def _build_mackay_icosahedral(n, r1, r2):
    """
    Build a 2-shell icosahedral cluster with n atoms.
    Shell 0: 1 atom at center
    Shell 1: 12 atoms at radius r1 (icosahedral directions)
    Shell 2: remaining atoms at radius r2 (42-icosahedral directions)
    
    Parameters:
        n: total number of atoms
        r1: radius of inner shell (12 atoms)
        r2: radius of outer shell (remaining atoms)
    """
    if n < 1:
        return np.zeros((0, 3))
    
    v1 = _get_icosahedron_vertices()  # 12 directions
    positions = [np.zeros(3)]  # center
    count = 1
    
    # Inner shell: 12 atoms
    n_inner = min(12, n - 1)
    for i in range(n_inner):
        positions.append(r1 * v1[i])
    count += n_inner
    
    if count >= n:
        return np.array(positions[:n])
    
    # Outer shell: remaining atoms
    remaining = n - count
    if remaining > 0:
        v2 = _get_icosahedral_shell2_directions()  # 42 directions
        for i in range(min(remaining, len(v2))):
            positions.append(r2 * v2[i])
        count += min(remaining, len(v2))
        
        # If we still need more atoms (n > 55), add more
        if count < n:
            rng = np.random.default_rng(42)
            while count < n:
                theta = np.arccos(2 * rng.random() - 1)
                phi_a = 2 * np.pi * rng.random()
                d = np.array([np.sin(theta) * np.cos(phi_a),
                              np.sin(theta) * np.sin(phi_a),
                              np.cos(theta)])
                positions.append(r2 * 1.1 * d)
                count += 1
    
    return np.array(positions[:n])


def _symmetric_energy(params, n):
    """Energy of symmetric icosahedral cluster given [r1, r2]."""
    r1, r2 = params
    if r1 < 0.3 or r2 < 0.5 or r1 > 5.0 or r2 > 8.0:
        return 1e10
    pos = _build_mackay_icosahedral(n, r1, r2)
    if len(pos) != n:
        return 1e10
    return _energy(pos)


def _optimize_symmetric(n, time_limit=30.0):
    """
    Optimize the 2-parameter symmetric icosahedral model using L-BFGS-B.
    Returns (best_energy, best_positions).
    """
    start_time = time.time()
    
    # Initial guesses: Mackay radii
    r5 = np.sqrt(5 + 2 * np.sqrt(5))
    initial_guesses = [
        [1.0, r5 / 2.0],      # Mackay
        [0.95, r5 / 2.0 * 0.95],
        [1.05, r5 / 2.0 * 1.05],
        [1.0, r5 / 2.0 * 0.9],
        [1.0, r5 / 2.0 * 1.1],
    ]
    
    best_energy = 1e10
    best_pos = None
    
    for guess in initial_guesses:
        if time.time() - start_time > time_limit:
            break
        if _remaining() < 5:
            break
        
        try:
            result = minimize(
                _symmetric_energy,
                x0=guess,
                args=(n,),
                method='L-BFGS-B',
                bounds=[(0.3, 5.0), (0.5, 8.0)],
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                r1_opt, r2_opt = result.x
                best_pos = _build_mackay_icosahedral(n, r1_opt, r2_opt)
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback: use Mackay
        best_pos = _build_mackay_icosahedral(n, 1.0, r5 / 2.0)
        best_energy = _energy(best_pos)
    
    return best_energy, best_pos


def _basin_hopping(pos, n, max_iter=40, time_limit=20, rng_seed=None):
    """Basin hopping with annealing step size."""
    start_time = time.time()
    current_energy = _energy(pos)
    best_energy = current_energy
    best_pos = pos.copy()
    
    if rng_seed is None:
        rng_seed = hash(tuple(pos.flatten()[:3])) % (2**32)
    rng = np.random.default_rng(rng_seed)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if _remaining() < 5:
            break
        
        # Annealing: large steps early, small steps late
        progress = it / max_iter
        base_step = 0.5 * (1.0 - 0.7 * progress)
        
        if rng.random() < 0.15:
            step_size = base_step * 2.0
        else:
            step_size = base_step
        
        perturbed = pos + rng.normal(0, step_size, size=pos.shape)
        
        try:
            e, relaxed = relax(perturbed, 'dzugutov', None)
        except:
            continue
        
        if not np.all(np.isfinite(e)) or e >= 1e9:
            continue
        
        if e < current_energy:
            current_energy = e
            pos = relaxed
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        else:
            # Metropolis acceptance with cooling
            T = 0.3 * (1.0 - progress)
            if T > 0.01 and rng.random() < np.exp(-(e - current_energy) / T):
                pos = relaxed
    
    return best_energy, best_pos


def _scatter(n, radius=3.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-0.3, 0.3, 3)
        points.append(p)
    
    return np.array(points[:n])


def _search_symmetric(n, time_limit=40.0):
    """
    Search strategy 1: Optimize symmetric icosahedral model, then basin hop.
    """
    start_time = time.time()
    
    # Step 1: Optimize symmetric parameters
    sym_energy, sym_pos = _optimize_symmetric(n, time_limit=min(25.0, time_limit * 0.5))
    
    if sym_pos is None or not np.all(np.isfinite(sym_pos)):
        r5 = np.sqrt(5 + 2 * np.sqrt(5))
        sym_pos = _build_mackay_icosahedral(n, 1.0, r5 / 2.0)
        sym_energy = _energy(sym_pos)
    
    # Relax the symmetric structure
    try:
        e, relaxed = relax(sym_pos, 'dzugutov', None)
        if np.all(np.isfinite(e)) and e < sym_energy:
            sym_energy = e
            sym_pos = relaxed
    except:
        pass
    
    best_energy = sym_energy
    best_pos = sym_pos.copy()
    
    # Step 2: Basin hopping from the optimized symmetric structure
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        try:
            bh_energy, bh_pos = _basin_hopping(
                sym_pos, n,
                max_iter=30,
                time_limit=min(remaining_time - 2, 25.0),
                rng_seed=12345
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except:
            pass
    
    return best_energy, best_pos


def _search_random(n, time_limit=40.0, seed_base=0):
    """
    Search strategy 2: Random compact seed + basin hopping.
    """
    start_time = time.time()
    
    pos = _scatter(n, radius=2.5, seed=seed_base)
    
    try:
        e, pos = relax(pos, 'dzugutov', None)
    except:
        return 1e10, None
    
    if not np.all(np.isfinite(e)) or e >= 1e9:
        return 1e10, None
    
    best_energy = e
    best_pos = pos.copy()
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        try:
            bh_energy, bh_pos = _basin_hopping(
                pos, n,
                max_iter=30,
                time_limit=min(remaining_time - 2, 25.0),
                rng_seed=seed_base * 1000 + 7
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except:
            pass
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster structure."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    
    best_energy = 1e10
    best_pos = None
    
    # Time budget
    total_budget = min(280.0, _remaining())
    
    # Run symmetric optimization (primary strategy)
    sym_time = min(total_budget * 0.5, 60.0)
    try:
        e_sym, pos_sym = _search_symmetric(n, time_limit=sym_time)
        if pos_sym is not None and np.all(np.isfinite(pos_sym)) and e_sym < best_energy:
            best_energy = e_sym
            best_pos = pos_sym
    except:
        pass
    
    # Run random searches in parallel (secondary strategy)
    remaining_budget = total_budget - (time.time() - _START_TIME if _START_TIME else 0)
    if remaining_budget > 10:
        n_random = min(3, max(1, int(remaining_budget / 30)))
        per_search_time = min(remaining_budget / n_random, 35.0)
        
        with ThreadPoolExecutor(max_workers=min(3, n_random)) as executor:
            futures = []
            for i in range(n_random):
                if _remaining() < 10:
                    break
                futures.append(executor.submit(_search_random, n, per_search_time, seed_base=100 + i))
            
            for future in as_completed(futures):
                if _remaining() < 5:
                    break
                try:
                    e, pos = future.result(timeout=20)
                    if pos is not None and np.all(np.isfinite(pos)) and e < best_energy:
                        best_energy = e
                        best_pos = pos
                except:
                    pass
    
    # Fallback
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        r5 = np.sqrt(5 + 2 * np.sqrt(5))
        best_pos = _build_mackay_icosahedral(n, 1.0, r5 / 2.0)
        try:
            _, best_pos = relax(best_pos, 'dzugutov', None)
        except:
            pass
    
    # Final check: ensure minimum distance
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        if min_dist < 0.5:
            for _ in range(200):
                if min_dist >= 0.5:
                    break
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                if i > j:
                    i, j = j, i
                dir_vec = best_pos[i] - best_pos[j]
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-10:
                    dir_vec = np.array([1.0, 0, 0])
                else:
                    dir_vec = dir_vec / norm
                best_pos[i] += 0.01 * dir_vec
                best_pos[j] -= 0.01 * dir_vec
                diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
            
            try:
                _, best_pos = relax(best_pos, 'dzugutov', None)
            except:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
