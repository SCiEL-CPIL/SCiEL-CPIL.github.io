# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 295.0  # Safety margin: 330s limit, keep 35s buffer
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_INDEX = {n: i for i, n in enumerate(_INSTANCE_ORDER)}

# Dzugutov potential parameters
A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij < 1e-10:
                continue
            if rij < a_DZ:
                v = A_DZ * (rij ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (rij - a_DZ))
            elif rij < b_DZ:
                v = B_DZ * np.exp(d_DZ / (rij - b_DZ))
            else:
                v = 0.0
            total += v
    return total


def _get_mackay_13():
    """Return the 13-atom Mackay icosahedron (known global minimum for n=13)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    # Optimal bond length for Dzugutov ~1.0
    r0 = 1.0
    positions = np.vstack([np.zeros(3), verts * r0])
    return positions


def _get_mackay_55():
    """
    Return the 55-atom Mackay icosahedral cluster (known global minimum for n=55).
    Structure: 1 center + 12 (r=1) + 30 (r=sqrt(2)) + 12 (r=sqrt(3))
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    positions = [np.zeros(3)]
    
    # Shell 1: 12 vertices at r=1
    r1 = 1.0
    for v in verts:
        positions.append(v * r1)
    
    # Shell 2: 30 edge midpoints at r=sqrt(2)
    r2 = np.sqrt(2)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.05:  # closest neighbors
                edges.append((i, j))
    
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid = mid / np.linalg.norm(mid) * r2
        positions.append(mid)
    
    # Shell 3: 12 face centers at r=sqrt(3)
    r3 = np.sqrt(3)
    # Find triangular faces
    edge_set = set()
    for i, j in edges:
        edge_set.add((i, j))
        edge_set.add((j, i))
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in edge_set:
                for k in range(j + 1, 12):
                    if (j, k) in edge_set and (i, k) in edge_set:
                        faces.append((i, j, k))
    
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3
        center = center / np.linalg.norm(center) * r3
        positions.append(center)
    
    return np.array(positions[:55])


def _insert_atom_optimal(positions, rng, min_dist=0.5):
    """
    Insert a new atom at the optimal interstitial site:
    maximize the minimum distance to existing atoms (subject to min_dist constraint).
    Uses a grid-based search for efficiency.
    """
    centroid = positions.mean(axis=0)
    max_r = np.max(np.linalg.norm(positions - centroid, axis=1))
    search_radius = max_r + 0.5
    
    best_pos = None
    best_min_dist = -1
    
    # Generate candidate positions on a fine grid around the cluster
    n_grid = 12
    candidates = []
    for dx in np.linspace(-search_radius, search_radius, n_grid):
        for dy in np.linspace(-search_radius, search_radius, n_grid):
            for dz in np.linspace(-search_radius, search_radius, n_grid):
                p = centroid + np.array([dx, dy, dz])
                candidates.append(p)
    
    # Also add random candidates
    for _ in range(200):
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        r = rng.uniform(0.5, search_radius)
        p = centroid + r * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        candidates.append(p)
    
    candidates = np.array(candidates)
    
    # Vectorized distance computation
    for idx in range(len(candidates)):
        p = candidates[idx]
        dists = np.linalg.norm(positions - p, axis=1)
        min_d = np.min(dists)
        if min_d >= min_dist and min_d > best_min_dist:
            best_min_dist = min_d
            best_pos = p
    
    if best_pos is None:
        # Fallback: random placement with larger offset
        best_pos = centroid + rng.normal(0, 1.0, 3)
    
    return best_pos


def _build_from_core(core_positions, n_atoms, rng, seed_id):
    """
    Build a cluster by starting from a known core and inserting remaining atoms
    at optimal interstitial sites.
    """
    positions = list(core_positions)
    
    while len(positions) < n_atoms:
        new_pos = _insert_atom_optimal(np.array(positions), rng, min_dist=0.5)
        positions.append(new_pos)
    
    return np.array(positions[:n_atoms])


def _generate_mackay_icosahedron(n_atoms):
    """
    Generate an icosahedral cluster structure for n_atoms atoms.
    Uses concentric icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = []
    positions.append(np.zeros(3))
    
    r1 = 1.0
    positions.extend([v * r1 for v in ico_verts])
    
    remaining = n_atoms - len(positions)
    
    if remaining > 0:
        # Generate second shell points (edges and faces)
        ico_edges = []
        dists = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                dists.append((d, i, j))
        dists.sort()
        min_dist = dists[0][0]
        for d, i, j in dists:
            if d < min_dist * 1.01:
                ico_edges.append((i, j))
        
        edge_mids = []
        for i, j in ico_edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2
            edge_mids.append(mid)
        edge_mids = np.array(edge_mids)
        
        connected = set()
        for i, j in ico_edges:
            connected.add((i, j))
            connected.add((j, i))
        
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                if (i, j) in connected:
                    for k in range(j+1, 12):
                        if (j, k) in connected and (i, k) in connected:
                            center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                            face_centers.append(center)
        face_centers = np.array(face_centers)
        
        r2 = r1 * np.sqrt(2)
        for mid in edge_mids:
            if remaining <= 0:
                break
            positions.append(mid * r2)
            remaining -= 1
        
        if remaining > 0:
            for fc in face_centers:
                if remaining <= 0:
                    break
                positions.append(fc * r2)
                remaining -= 1
    
    if remaining > 0:
        r3 = r1 * np.sqrt(3)
        rng = np.random.default_rng(42)
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = r3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _generate_fcc_cluster(n_atoms):
    """Generate an FCC-like cluster."""
    positions = []
    a = 1.0
    max_coord = int(np.ceil(n_atoms ** 0.5)) + 2
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    candidates.append(np.array([i, j, k]) * a / np.sqrt(2))
                candidates.append(np.array([i, j, k + 0.5]) * a / np.sqrt(2))
                candidates.append(np.array([i, j + 0.5, k]) * a / np.sqrt(2))
                candidates.append(np.array([i + 0.5, j, k]) * a / np.sqrt(2))
    
    candidates = np.array(candidates)
    centers = np.zeros(3)
    dists = np.linalg.norm(candidates - centers, axis=1)
    order = np.argsort(dists)
    
    selected = []
    for idx in order:
        p = candidates[idx]
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
        if len(selected) == n_atoms:
            break
    
    rng = np.random.default_rng(n_atoms)
    while len(selected) < n_atoms:
        p = rng.uniform(-3, 3, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
    
    return np.array(selected)


def _generate_random_compact(n_atoms, seed):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    spacing = 1.1
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.75
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        rng2 = np.random.default_rng(seed + 1)
        radius2 = spacing * n_atoms ** (1.0 / 3.0) * 0.6
        points2 = []
        attempts = 0
        while len(points2) < n_atoms and attempts < max_attempts:
            p = rng2.uniform(-radius2, radius2, 3)
            if np.linalg.norm(p) <= radius2:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points2):
                    points2.append(p)
            attempts += 1
        if len(points2) >= n_atoms:
            points = points2
    
    while len(points) < n_atoms:
        p = np.array([rng.uniform(-radius, radius) for _ in range(3)])
        points.append(p)
    
    return np.array(points[:n_atoms])


def _generate_langevin_seed(n_atoms, seed):
    """
    Generate a seed via physics-driven self-assembly:
    sequentially add atoms at random positions near the growing cluster,
    allowing the system to naturally evolve toward low-energy configurations.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a small nucleus
    positions = [np.zeros(3)]
    
    # Add first shell atoms around the center using icosahedral geometry
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Add up to 12 atoms from icosahedral vertices with small random perturbation
    for v in ico_verts:
        if len(positions) >= n_atoms:
            break
        p = v * 1.0 + rng.normal(0, 0.05, 3)
        # Ensure minimum distance from existing atoms
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)
    
    # Now sequentially add remaining atoms via self-assembly
    while len(positions) < n_atoms:
        # Compute current cluster centroid and radius
        pos_arr = np.array(positions)
        centroid = pos_arr.mean(axis=0)
        if len(pos_arr) > 1:
            radius = np.max(np.linalg.norm(pos_arr - centroid, axis=1))
        else:
            radius = 1.0
        
        # Try to add an atom at a random position on the surface
        added = False
        for attempt in range(100):
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            
            # Place at radius + small offset
            r_new = radius + rng.uniform(-0.1, 0.2)
            p_new = centroid + direction * r_new
            
            # Check minimum distance
            valid = True
            for q in positions:
                if np.linalg.norm(p_new - q) < 0.5:
                    valid = False
                    break
            if valid:
                positions.append(p_new)
                added = True
                break
        
        # If we couldn't add an atom after 100 attempts, place it with larger offset
        if not added:
            p_new = centroid + rng.normal(0, 0.8, 3)
            positions.append(p_new)
    
    positions = np.array(positions[:n_atoms])
    
    return positions


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_scale_start, kick_scale_end, deadline):
    """
    Basin hopping: random kick + local relaxation.
    Returns (best_energy, best_positions).
    """
    best_energy, best_positions = relax(positions.copy(), potential, parameter)
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        t = it / max(1, max_iters - 1)
        kick_scale = kick_scale_start + (kick_scale_end - kick_scale_start) * t
        
        kick = rng.normal(0, kick_scale, size=current_positions.shape)
        new_positions = current_positions + kick
        
        try:
            new_energy, new_relaxed = relax(new_positions, potential, parameter)
        except Exception:
            continue
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0.1:
                accept_prob = np.exp(-delta_e / 0.05)
                if rng.random() < accept_prob:
                    current_energy = new_energy
                    current_positions = new_relaxed
    
    return best_energy, best_positions


def _search_chain(seed_positions, potential, parameter, n_atoms, deadline, chain_id):
    """
    Run a basin hopping search chain starting from seed_positions.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(chain_id * 1000 + n_atoms)
    
    # Adaptive iterations based on time budget
    elapsed = time.time() - _START_TIME if _START_TIME else 0
    remaining = max(0, _HARD_LIMIT - elapsed)
    
    # Estimate how many iterations we can afford
    est_relax_time = 0.2 + 0.01 * n_atoms
    max_iters = max(5, int(remaining / (est_relax_time * 2)))
    max_iters = min(max_iters, 30)
    
    kick_scale_start = 0.35
    kick_scale_end = 0.08
    
    best_energy, best_positions = _basin_hopping(
        seed_positions, potential, parameter, rng,
        max_iters, kick_scale_start, kick_scale_end, deadline
    )
    
    return best_energy, best_positions


def _generate_core_seeds(n_atoms, deadline):
    """
    Generate seeds by building from known global minimum cores.
    For n=55, use the exact 55-atom Mackay icosahedron.
    For other sizes, build from the 13-atom core or 55-atom core (if smaller).
    """
    seeds = []
    rng_base = np.random.default_rng(12345)
    
    # For n=55, use the exact known global minimum
    if n_atoms == 55:
        try:
            mackay55 = _get_mackay_55()
            if len(mackay55) == 55:
                seeds.append(('mackay55', mackay55))
        except Exception:
            pass
    
    # Build from 13-atom core
    try:
        mackay13 = _get_mackay_13()
        if n_atoms >= 13:
            rng = np.random.default_rng(1000 + n_atoms)
            built = _build_from_core(mackay13, n_atoms, rng, 0)
            seeds.append(('core13', built))
    except Exception:
        pass
    
    # Build from 55-atom core for n > 55
    if n_atoms > 55:
        try:
            mackay55 = _get_mackay_55()
            rng = np.random.default_rng(2000 + n_atoms)
            built = _build_from_core(mackay55, n_atoms, rng, 1)
            seeds.append(('core55', built))
        except Exception:
            pass
    
    # Also build from 13-atom core with different random seeds
    for seed_id in range(1, 4):
        if time.time() > deadline:
            break
        try:
            mackay13 = _get_mackay_13()
            rng = np.random.default_rng(3000 + n_atoms + seed_id)
            built = _build_from_core(mackay13, n_atoms, rng, seed_id)
            seeds.append((f'core13_{seed_id}', built))
        except Exception:
            pass
    
    return seeds


def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using basin hopping
    with diverse polytetrahedral seeds, including known global minimum cores.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Calculate time budget
    elapsed = time.time() - _START_TIME
    remaining_total = max(0, _HARD_LIMIT - elapsed)
    
    # Determine how many instances are left, including this one
    idx = _INSTANCE_INDEX.get(size, 0)
    n_rem = len(_INSTANCE_ORDER) - idx
    
    # Allocate time for this call: share remaining time among remaining instances
    if n_rem > 0:
        call_budget = remaining_total / n_rem
    else:
        call_budget = remaining_total
    
    # Ensure we don't allocate negative or zero time
    call_budget = max(call_budget, 1.0)
    
    call_deadline = time.time() + call_budget
    
    # Generate diverse seed structures
    seeds = []
    
    # Core-based seeds (highest priority)
    core_deadline = time.time() + min(2.0, call_budget * 0.2)
    core_seeds = _generate_core_seeds(size, core_deadline)
    for name, pos in core_seeds:
        seeds.append((name, pos))
    
    # Traditional seeds
    try:
        ico = _generate_mackay_icosahedron(size)
        seeds.append(('ico', ico))
    except Exception:
        pass
    
    try:
        fcc = _generate_fcc_cluster(size)
        seeds.append(('fcc', fcc))
    except Exception:
        pass
    
    try:
        lang1 = _generate_langevin_seed(size, 99)
        seeds.append(('lang1', lang1))
    except Exception:
        pass
    
    try:
        lang2 = _generate_langevin_seed(size, 199)
        seeds.append(('lang2', lang2))
    except Exception:
        pass
    
    # Add random seeds
    n_random_seeds = 2
    for seed_val in range(n_random_seeds):
        try:
            rand = _generate_random_compact(size, seed_val + 1)
            seeds.append((f'rand{seed_val}', rand))
        except Exception:
            pass
    
    if not seeds:
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        seeds.append(('fallback', np.array(points)))
    
    best_energy = float('inf')
    best_positions = None
    
    # If time is very tight, just relax the best seed
    if call_budget < 5.0:
        try:
            _, best_positions = relax(seeds[0][1].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0][1]
        return best_positions
    
    # Prioritize seeds: core-based first, then icosahedron, then others
    seed_priority = []
    for i, (name, s) in enumerate(seeds):
        if name.startswith('core') or name == 'mackay55':
            seed_priority.append((0, i))
        elif name == 'ico':
            seed_priority.append((1, i))
        elif name == 'fcc':
            seed_priority.append((2, i))
        elif name.startswith('lang'):
            seed_priority.append((3, i))
        else:
            seed_priority.append((4, i))
    seed_priority.sort()
    prioritized_seeds = [seeds[idx] for _, idx in seed_priority]
    
    n_threads = min(4, len(prioritized_seeds))
    
    if n_threads == 1:
        energy, positions = _search_chain(prioritized_seeds[0][1], potential, parameter, size, call_deadline, 0)
        best_energy = energy
        best_positions = positions
    else:
        results = [None] * len(prioritized_seeds)
        
        def run_chain(i):
            if time.time() > call_deadline:
                return (float('inf'), None)
            return _search_chain(prioritized_seeds[i][1], potential, parameter, size, call_deadline, i)
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(run_chain, i): i for i in range(len(prioritized_seeds))}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    timeout = max(0.1, call_deadline - time.time())
                    energy, positions = future.result(timeout=timeout)
                    if positions is not None and energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    continue
    
    if best_positions is None:
        try:
            _, best_positions = relax(seeds[0][1].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0][1]
    
    return best_positions
# EVOLVE-BLOCK-END
