# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START_TIME = None
_HARD_LIMIT = 310.0
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_INDEX = {n: i for i, n in enumerate(_INSTANCE_ORDER)}

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask_a = dists < a_DZ
        mask_b = (dists >= a_DZ) & (dists < b_DZ)
        
        if np.any(mask_a):
            da = dists[mask_a]
            da = np.maximum(da, 1e-10)
            v_a = A_DZ * (da ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (da - a_DZ))
            total += np.sum(v_a)
        
        if np.any(mask_b):
            db = dists[mask_b]
            db = np.maximum(db, 1e-10)
            v_b = B_DZ * np.exp(d_DZ / (db - b_DZ))
            total += np.sum(v_b)
    
    return total


def _generate_icosahedron(n_atoms):
    """Generate an icosahedral cluster with concentric shells."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = [np.zeros(3)]
    
    # First shell: 12 vertices of icosahedron
    r1 = 1.0
    for v in ico_verts:
        if len(positions) >= n_atoms:
            break
        positions.append(v * r1)
    
    if len(positions) >= n_atoms:
        return np.array(positions[:n_atoms])
    
    remaining = n_atoms - len(positions)
    
    # Second shell: 30 edge midpoints (scaled)
    ico_edges = []
    min_edge_dist = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < min_edge_dist * 1.01:
                ico_edges.append((i, j))
    
    edge_mids = []
    for i, j in ico_edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Scale second shell to maintain reasonable spacing
    r2 = r1 * 1.7
    for mid in edge_mids:
        if len(positions) >= n_atoms:
            break
        positions.append(mid * r2)
    
    if len(positions) >= n_atoms:
        return np.array(positions[:n_atoms])
    
    remaining = n_atoms - len(positions)
    
    # Third shell: 20 face centers (scaled)
    face_centers = []
    edge_set = set()
    for i, j in ico_edges:
        edge_set.add((i, j))
        edge_set.add((j, i))
    
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) in edge_set:
                for k in range(j+1, 12):
                    if (j, k) in edge_set and (i, k) in edge_set:
                        center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    r3 = r1 * 2.2
    for fc in face_centers:
        if len(positions) >= n_atoms:
            break
        positions.append(fc * r3)
    
    if len(positions) >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Fourth shell and beyond: use random points on a sphere
    r4 = r1 * 2.8
    rng = np.random.default_rng(42)
    while len(positions) < n_atoms:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        p = r4 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        positions.append(p)
    
    return np.array(positions[:n_atoms])


def _generate_mackay_like(n_atoms):
    """Generate a Mackay-like icosahedral cluster with specific shell radii."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = [np.zeros(3)]
    
    # Shell 1: 12 vertices at r=1
    r1 = 1.0
    for v in ico_verts:
        if len(positions) >= n_atoms:
            break
        positions.append(v * r1)
    
    if len(positions) >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Shell 2: 30 edge midpoints at r=1.7
    r2 = 1.7
    ico_edges = []
    min_edge_dist = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < min_edge_dist * 1.01:
                ico_edges.append((i, j))
    
    edge_mids = []
    for i, j in ico_edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        edge_mids.append(mid)
    
    for mid in edge_mids:
        if len(positions) >= n_atoms:
            break
        positions.append(mid * r2)
    
    if len(positions) >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Shell 3: 20 face centers at r=2.2
    r3 = 2.2
    edge_set = set()
    for i, j in ico_edges:
        edge_set.add((i, j))
        edge_set.add((j, i))
    
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) in edge_set:
                for k in range(j+1, 12):
                    if (j, k) in edge_set and (i, k) in edge_set:
                        center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                        face_centers.append(center)
    
    for fc in face_centers:
        if len(positions) >= n_atoms:
            break
        positions.append(fc * r3)
    
    if len(positions) >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Additional shells: random points
    r4 = 2.8
    rng = np.random.default_rng(n_atoms)
    while len(positions) < n_atoms:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        p = r4 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        positions.append(p)
    
    return np.array(positions[:n_atoms])


def _generate_decahedral(n_atoms):
    """Generate a decahedral cluster (5 icosahedral fragments fused)."""
    positions = [np.zeros(3)]
    
    # 5-fold symmetry axes
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # 5 main axes (vertices of a regular icosahedron, pick 5)
    main_axes = ico_verts[:5]
    main_axes = main_axes / np.linalg.norm(main_axes, axis=1, keepdims=True)
    
    # Place atoms along these 5 directions
    r1 = 1.0
    for axis in main_axes:
        if len(positions) >= n_atoms:
            break
        positions.append(axis * r1)
        # Add 4 more atoms around this axis (forming a small icosahedron fragment)
        for k in range(1, 5):
            if len(positions) >= n_atoms:
                break
            # Rotate around the axis by golden angle
            angle = 2 * np.pi * k / 5
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # Simple rotation approximation
            v = axis.copy()
            # Create perpendicular vectors
            perp1 = np.cross(axis, np.array([1, 0, 0]))
            if np.linalg.norm(perp1) < 1e-6:
                perp1 = np.cross(axis, np.array([0, 1, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(axis, perp1)
            new_v = v * cos_a + perp1 * sin_a * 0.5
            new_v = new_v / np.linalg.norm(new_v)
            positions.append(new_v * r1)
    
    if len(positions) >= n_atoms:
        return np.array(positions[:n_atoms])
    
    # Add more atoms in a spherical shell
    r2 = 1.8
    rng = np.random.default_rng(n_atoms + 100)
    while len(positions) < n_atoms:
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        p = r2 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        positions.append(p)
    
    return np.array(positions[:n_atoms])


def _generate_compact_random(n_atoms, seed):
    """Generate a compact random cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    spacing = 1.0
    radius = spacing * (n_atoms / 4.18879) ** (1.0 / 3.0) * 1.2  # ~40% packing
    
    points = []
    max_attempts = 5000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        # Fallback: use a grid
        grid_size = int(np.ceil(n_atoms ** 0.5)) + 2
        candidates = []
        for i in range(-grid_size, grid_size + 1):
            for j in range(-grid_size, grid_size + 1):
                for k in range(-grid_size, grid_size + 1):
                    candidates.append(np.array([i, j, k]) * spacing / np.sqrt(2))
        
        candidates = np.array(candidates)
        dists = np.linalg.norm(candidates, axis=1)
        order = np.argsort(dists)
        
        for idx in order:
            if len(points) >= n_atoms:
                break
            p = candidates[idx]
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
    
    # If still not enough, add random points far away
    while len(points) < n_atoms:
        p = rng.uniform(-radius * 2, radius * 2, 3)
        points.append(p)
    
    return np.array(points[:n_atoms])


def _self_assemble(n_atoms, seed, deadline):
    """
    Physics-driven self-assembly: sequentially add atoms to a growing cluster,
    allowing the system to relax after each addition. This lets the structure
    naturally evolve into low-energy polytetrahedral configurations.
    """
    rng = np.random.default_rng(seed)
    potential = 'dzugutov'
    parameter = None
    
    # Start with a small icosahedral core (13 atoms: 1 center + 12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Initial core: center + up to 12 vertices
    core_size = min(13, n_atoms)
    positions = [np.zeros(3)]
    for v in ico_verts[:core_size - 1]:
        positions.append(v * 1.0)
    positions = np.array(positions)
    
    # Relax the core
    if len(positions) > 1 and time.time() < deadline:
        try:
            _, positions = relax(positions.copy(), potential, parameter)
        except Exception:
            pass
    
    # Sequentially add atoms one by one
    while len(positions) < n_atoms:
        if time.time() > deadline:
            break
        
        n_current = len(positions)
        
        # Determine where to add the new atom: near existing atoms but not too close
        # Try several candidate positions and pick the one that gives lowest energy
        best_candidate = None
        best_candidate_energy = float('inf')
        
        for attempt in range(20):
            # Pick a random existing atom
            idx = rng.integers(0, n_current)
            base_atom = positions[idx]
            
            # Add new atom at a random offset of ~1.0-1.5 from base_atom
            offset = rng.normal(0, 1.0, 3)
            offset_norm = np.linalg.norm(offset)
            if offset_norm < 1e-6:
                offset = np.array([1.0, 0, 0])
            else:
                offset = offset / offset_norm
            
            new_pos = base_atom + offset * rng.uniform(0.8, 1.3)
            
            # Check minimum distance constraint
            dists = np.linalg.norm(positions - new_pos, axis=1)
            if np.min(dists) < 0.5:
                continue
            
            candidate = np.vstack([positions, new_pos.reshape(1, 3)])
            
            # Quick energy check (only for nearby pairs to save time)
            try:
                e = _dzugutov_energy(candidate)
                if e < best_candidate_energy:
                    best_candidate_energy = e
                    best_candidate = candidate
            except Exception:
                continue
        
        if best_candidate is None:
            # Fallback: add atom on a sphere around the center
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            r = 1.5 * (n_current / 13) ** (1/3)
            new_pos = r * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            positions = np.vstack([positions, new_pos.reshape(1, 3)])
        else:
            positions = best_candidate
        
        # Relax after adding the new atom
        if time.time() < deadline:
            try:
                _, positions = relax(positions.copy(), potential, parameter)
            except Exception:
                pass
    
    return positions[:n_atoms]


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_start, kick_end, deadline):
    """
    Basin hopping with local relaxation.
    Returns (best_energy, best_positions).
    """
    # First, relax the initial structure
    try:
        best_energy, best_positions = relax(positions.copy(), potential, parameter)
    except Exception:
        best_energy = _dzugutov_energy(positions)
        best_positions = positions.copy()
    
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Linearly decrease kick size
        t = it / max(1, max_iters - 1)
        kick_scale = kick_start + (kick_end - kick_start) * t
        
        # Generate random kick
        kick = rng.normal(0, kick_scale, size=current_positions.shape)
        new_positions = current_positions + kick
        
        # Relax
        try:
            new_energy, new_relaxed = relax(new_positions, potential, parameter)
        except Exception:
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
        else:
            delta_e = new_energy - current_energy
            # Accept with some probability for exploration
            if delta_e < 0.5:
                accept_prob = np.exp(-delta_e / 0.1)
                if rng.random() < accept_prob:
                    current_energy = new_energy
                    current_positions = new_relaxed
    
    return best_energy, best_positions


def _search_from_seed(seed, potential, parameter, n_atoms, deadline, seed_id):
    """
    Run basin hopping from a single seed.
    """
    rng = np.random.default_rng(seed_id * 10000 + n_atoms)
    max_iters = 25
    kick_start = 0.5
    kick_end = 0.15
    
    return _basin_hopping(seed, potential, parameter, rng, max_iters, kick_start, kick_end, deadline)


def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using diverse seeds including
    self-assembly and basin hopping optimization.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time management
    elapsed = time.time() - _START_TIME
    remaining_total = max(0, _HARD_LIMIT - elapsed)
    
    idx = _INSTANCE_INDEX.get(size, 0)
    n_rem = len(_INSTANCE_ORDER) - idx
    
    if n_rem > 0:
        call_budget = remaining_total / n_rem
    else:
        call_budget = remaining_total
    
    call_budget = max(call_budget, 3.0)
    call_deadline = time.time() + call_budget
    
    # Generate diverse seeds
    seeds = []
    
    # Seed 0: icosahedral
    try:
        seeds.append(_generate_icosahedron(size))
    except Exception:
        pass
    
    # Seed 1: Mackay-like
    try:
        seeds.append(_generate_mackay_like(size))
    except Exception:
        pass
    
    # Seed 2: decahedral
    try:
        seeds.append(_generate_decahedral(size))
    except Exception:
        pass
    
    # Seed 3: self-assembly (physics-driven growth)
    try:
        seeds.append(_self_assemble(size, seed=size * 7, deadline=call_deadline - 0.5))
    except Exception:
        pass
    
    # Seeds 4-6: random compact
    for s in range(3):
        try:
            seeds.append(_generate_compact_random(size, 100 + s))
        except Exception:
            pass
    
    if not seeds:
        # Fallback
        rng = np.random.default_rng(size)
        radius = 1.5 * size ** (1.0 / 3.0)
        points = [rng.uniform(-radius, radius, 3) for _ in range(size)]
        seeds.append(np.array(points))
    
    best_energy = float('inf')
    best_positions = None
    
    # If very little time, just relax the first seed
    if call_budget < 5.0:
        try:
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
        return best_positions
    
    # Run parallel basin hopping chains
    n_threads = min(4, len(seeds))
    results = [None] * len(seeds)
    
    def run_chain(i):
        if time.time() > call_deadline:
            return (float('inf'), None)
        try:
            return _search_from_seed(seeds[i], potential, parameter, size, call_deadline, i)
        except Exception:
            return (float('inf'), None)
    
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
