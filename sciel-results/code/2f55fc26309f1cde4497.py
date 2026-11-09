# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 292.0
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


def _get_icosahedron_13():
    """Return the 13-atom icosahedral cluster (known global minimum for Dzugutov)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return np.vstack([np.zeros(3), verts])


def _build_hierarchical_seed(n_atoms, seed_idx):
    """
    Build a seed by starting with the 13-atom icosahedron core and
    adding atoms at geometrically favorable positions (edge midpoints,
    face centers, and random surface positions) scaled to fill the cluster.
    """
    core = _get_icosahedron_13()
    phi = (1 + np.sqrt(5)) / 2
    verts = core[1:]  # 12 vertices
    
    # Identify edges and faces of the icosahedron
    edges = []
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    min_dist = dists[0][0]
    for d, i, j in dists:
        if d < min_dist * 1.01:
            edges.append((i, j))
    
    connected = set()
    for i, j in edges:
        connected.add((i, j))
        connected.add((j, i))
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in connected:
                for k in range(j + 1, 12):
                    if (j, k) in connected and (i, k) in connected:
                        faces.append((i, j, k))
    
    # Generate candidate positions for additional atoms
    rng = np.random.default_rng(seed_idx * 7919 + n_atoms)
    
    # Edge midpoints (scaled outward)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers (scaled outward)
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Build position list: core + edge mids + face centers + random surface
    positions = [core.copy()]
    
    remaining = n_atoms - 13
    if remaining <= 0:
        return core[:n_atoms]
    
    # Scale factor for second shell
    r2 = np.sqrt(2.0)  # typical second shell radius ratio
    
    # Add edge midpoints
    for mid in edge_mids:
        if remaining <= 0:
            break
        p = mid * r2
        # Check minimum distance
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)
            remaining -= 1
    
    # Add face centers
    for fc in face_centers:
        if remaining <= 0:
            break
        p = fc * r2
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)
            remaining -= 1
    
    # Add random surface atoms for remaining
    if remaining > 0:
        r3 = r2 * 1.3
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            p = r3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            # Try to find a valid position
            for attempt in range(20):
                valid = True
                for q in positions:
                    if np.linalg.norm(p - q) < 0.5:
                        valid = False
                        break
                if valid:
                    break
                p = p + rng.normal(0, 0.3, 3)
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _build_icosahedral_shell_seed(n_atoms, seed_idx):
    """
    Alternative seed: pure concentric icosahedral shells with perturbation.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    rng = np.random.default_rng(seed_idx * 104729 + n_atoms)
    
    positions = [np.zeros(3)]
    r1 = 1.0
    for v in ico_verts:
        if len(positions) >= n_atoms:
            break
        p = v * r1 + rng.normal(0, 0.03, 3)
        positions.append(p)
    
    remaining = n_atoms - len(positions)
    if remaining > 0:
        # Second shell
        edges = []
        dists = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                dists.append((d, i, j))
        dists.sort()
        min_dist = dists[0][0]
        for d, i, j in dists:
            if d < min_dist * 1.01:
                edges.append((i, j))
        
        connected = set()
        for i, j in edges:
            connected.add((i, j))
            connected.add((j, i))
        
        edge_mids = [(ico_verts[i] + ico_verts[j]) / 2 for i, j in edges]
        face_centers = []
        for i in range(12):
            for j in range(i + 1, 12):
                if (i, j) in connected:
                    for k in range(j + 1, 12):
                        if (j, k) in connected and (i, k) in connected:
                            fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                            face_centers.append(fc)
        
        r2 = r1 * np.sqrt(2)
        for mid in edge_mids:
            if remaining <= 0:
                break
            p = mid * r2 + rng.normal(0, 0.03, 3)
            positions.append(p)
            remaining -= 1
        
        for fc in face_centers:
            if remaining <= 0:
                break
            p = fc * r2 + rng.normal(0, 0.03, 3)
            positions.append(p)
            remaining -= 1
    
    if remaining > 0:
        r3 = r1 * np.sqrt(3)
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            p = r3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _build_random_compact_seed(n_atoms, seed_idx):
    """Random compact cluster as diversity seed."""
    rng = np.random.default_rng(seed_idx * 31 + n_atoms)
    spacing = 1.1
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.75
    points = []
    max_attempts = 5000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n_atoms])


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_scale_start, kick_scale_end, deadline):
    """
    Basin hopping: random kick + local relaxation.
    Returns (best_energy, best_positions).
    """
    try:
        best_energy, best_positions = relax(positions.copy(), potential, parameter)
    except Exception:
        return float('inf'), positions
    
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
            if delta_e < 0.2:
                accept_prob = np.exp(-delta_e / 0.06)
                if rng.random() < accept_prob:
                    current_energy = new_energy
                    current_positions = new_relaxed
    
    return best_energy, best_positions


def _search_chain(seed_positions, potential, parameter, n_atoms, deadline, chain_id):
    """
    Run a basin hopping search chain starting from seed_positions.
    """
    rng = np.random.default_rng(chain_id * 1000 + n_atoms)
    
    max_iters = 25
    kick_scale_start = 0.35
    kick_scale_end = 0.10
    
    best_energy, best_positions = _basin_hopping(
        seed_positions, potential, parameter, rng,
        max_iters, kick_scale_start, kick_scale_end, deadline
    )
    
    return best_energy, best_positions


def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using hierarchical
    polytetrahedral seeds with basin hopping refinement.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    elapsed = time.time() - _START_TIME
    remaining_total = max(0, _HARD_LIMIT - elapsed)
    
    idx = _INSTANCE_INDEX.get(size, 0)
    n_rem = len(_INSTANCE_ORDER) - idx
    
    if n_rem > 0:
        call_budget = remaining_total / n_rem
    else:
        call_budget = remaining_total
    
    call_budget = max(call_budget, 2.0)
    call_deadline = time.time() + call_budget
    
    # Generate diverse seeds emphasizing polytetrahedral hierarchy
    seeds = []
    
    # Primary: hierarchical icosahedron-based seed
    try:
        hier = _build_hierarchical_seed(size, seed_idx=0)
        seeds.append(hier)
    except Exception:
        pass
    
    # Alternative hierarchical with different seed
    try:
        hier2 = _build_hierarchical_seed(size, seed_idx=1)
        seeds.append(hier2)
    except Exception:
        pass
    
    # Icosahedral shell seeds
    for s in range(3):
        try:
            ico = _build_icosahedral_shell_seed(size, seed_idx=s)
            seeds.append(ico)
        except Exception:
            pass
    
    # Random compact for diversity
    for s in range(2):
        try:
            rand = _build_random_compact_seed(size, seed_idx=s)
            seeds.append(rand)
        except Exception:
            pass
    
    if not seeds:
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        seeds.append(np.array(points))
    
    best_energy = float('inf')
    best_positions = None
    
    # If time is very tight, just relax the best seed
    if call_budget < 5.0:
        try:
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
        return best_positions
    
    n_threads = min(4, len(seeds))
    
    if n_threads == 1:
        energy, positions = _search_chain(seeds[0], potential, parameter, size, call_deadline, 0)
        best_energy = energy
        best_positions = positions
    else:
        results = [None] * len(seeds)
        
        def run_chain(i):
            if time.time() > call_deadline:
                return (float('inf'), None)
            return _search_chain(seeds[i], potential, parameter, size, call_deadline, i)
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(run_chain, i): i for i in range(len(seeds))}
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
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
    
    return best_positions
# EVOLVE-BLOCK-END
