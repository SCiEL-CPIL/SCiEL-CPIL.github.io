"""
Strategy: Basin hopping with diverse polytetrahedral seeds for Dzugutov clusters.
The Dzugutov potential favours icosahedral/polytetrahedral order. We generate
multiple diverse initial structures (Mackay supericosahedron-based, FCC-based,
random compact) and run basin hopping (random kick + local relaxation) on each.
We use up to 4 threads to parallelize the search chains.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 300.0  # Total budget for all 6 calls combined
# We'll allocate time proportionally. Since construct is called sequentially
# (or we can't control order), we track remaining time.

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


def _generate_mackay_icosahedron(n_atoms):
    """
    Generate an icosahedral cluster structure for n_atoms atoms.
    Uses concentric icosahedral shells.
    """
    # Icosahedral shell atom counts: 1, 12, 42, 92, ...
    # For n=38: 1+12+25 (partial second shell)
    # For n=45: 1+12+32 (partial)
    # For n=50: 1+12+37 (partial)
    # For n=55: 1+12+42 (Mackay)
    # For n=57: 1+12+42+2 (partial third)
    # For n=60: 1+12+42+5 (partial third)
    
    # Icosahedral vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # Normalize to unit icosahedron
    
    # Second shell: 42 atoms (edges of icosahedron)
    # The 30 edges of an icosahedron + 12 face centers
    ico_edges = []
    ico_faces = []
    # Find edges (pairs of vertices at minimum distance)
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
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in ico_edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers (12 faces)
    # Find faces (triples of mutually connected vertices)
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
    
    # Build shell positions
    # Shell 0: center (1 atom)
    # Shell 1: 12 vertices at radius r1
    # Shell 2: 30 edge mids + 12 face centers at radius r2
    # Shell 3: 92 atoms at radius r3
    
    positions = []
    positions.append(np.zeros(3))  # center
    
    r1 = 1.0
    positions.extend([v * r1 for v in ico_verts])
    
    # We need to fill up to n_atoms
    remaining = n_atoms - len(positions)
    
    if remaining > 0:
        r2 = r1 * np.sqrt(2)  # Approximate second shell radius
        # Add edge midpoints
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
        # Third shell: generate more positions
        # Use a larger set of icosahedral directions
        r3 = r1 * np.sqrt(3)
        # Generate additional positions on a sphere
        rng = np.random.default_rng(42)
        for _ in range(remaining):
            # Random point on sphere
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = r3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _generate_fcc_cluster(n_atoms):
    """Generate an FCC-like cluster."""
    positions = []
    # FCC lattice vectors
    a = 1.0  # lattice constant
    # Generate FCC points
    max_coord = int(np.ceil(n_atoms ** 0.5)) + 2
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (i,j,k) where i+j+k is even, plus face centers
                if (i + j + k) % 2 == 0:
                    candidates.append(np.array([i, j, k]) * a / np.sqrt(2))
                # Face centers
                candidates.append(np.array([i, j, k + 0.5]) * a / np.sqrt(2))
                candidates.append(np.array([i, j + 0.5, k]) * a / np.sqrt(2))
                candidates.append(np.array([i + 0.5, j, k]) * a / np.sqrt(2))
    
    # Sort by distance from center
    candidates = np.array(candidates)
    centers = np.zeros(3)
    dists = np.linalg.norm(candidates - centers, axis=1)
    order = np.argsort(dists)
    
    # Select n_atoms closest, ensuring minimum distance
    selected = []
    for idx in order:
        p = candidates[idx]
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
        if len(selected) == n_atoms:
            break
    
    # If we didn't get enough, add random points
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
    
    # If we couldn't place all atoms, relax spacing
    if len(points) < n_atoms:
        # Use a denser packing
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
    
    # Ensure we have exactly n_atoms
    while len(points) < n_atoms:
        p = np.array([rng.uniform(-radius, radius) for _ in range(3)])
        points.append(p)
    
    return np.array(points[:n_atoms])


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_scale_start, kick_scale_end, deadline):
    """
    Basin hopping: random kick + local relaxation.
    Returns (best_energy, best_positions).
    """
    best_energy, best_positions = relax(positions.copy(), potential, parameter)
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    n_kicks = max_iters
    for it in range(n_kicks):
        if time.time() > deadline:
            break
        
        # Linearly anneal kick scale
        t = it / max(1, n_kicks - 1)
        kick_scale = kick_scale_start + (kick_scale_end - kick_scale_start) * t
        
        # Random kick
        kick = rng.normal(0, kick_scale, size=current_positions.shape)
        new_positions = current_positions + kick
        
        # Relax
        try:
            new_energy, new_relaxed = relax(new_positions, potential, parameter)
        except Exception:
            continue
        
        # Accept if lower energy (or with Metropolis criterion for exploration)
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
        else:
            # Metropolis acceptance for exploration
            delta_e = new_energy - current_energy
            if delta_e < 0.1:  # Small energy increase
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
    
    # Determine max iterations based on time budget
    # Rough estimate: each relax call takes ~5-20ms for n~50
    # We'll cap at a reasonable number
    max_iters = 50
    kick_scale_start = 0.3
    kick_scale_end = 0.1
    
    best_energy, best_positions = _basin_hopping(
        seed_positions, potential, parameter, rng,
        max_iters, kick_scale_start, kick_scale_end, deadline
    )
    
    return best_energy, best_positions


def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using basin hopping
    with diverse polytetrahedral seeds.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this call
    elapsed = time.time() - _START_TIME
    # Assume 6 calls total, each gets roughly 300/6 = 50s
    # But we should use remaining time wisely
    remaining_total = max(0, _HARD_LIMIT - elapsed)
    # Allocate this call a fraction of remaining time
    # Be conservative: use 80% of remaining for this call
    call_deadline = time.time() + remaining_total * 0.8
    
    # Generate diverse seed structures
    seeds = []
    
    # 1. Mackay icosahedron
    try:
        ico = _generate_mackay_icosahedron(size)
        seeds.append(ico)
    except Exception:
        pass
    
    # 2. FCC cluster
    try:
        fcc = _generate_fcc_cluster(size)
        seeds.append(fcc)
    except Exception:
        pass
    
    # 3-6. Random compact clusters with different seeds
    for seed_val in [1, 2, 3, 4]:
        try:
            rand = _generate_random_compact(size, seed_val)
            seeds.append(rand)
        except Exception:
            pass
    
    # Ensure we have at least one seed
    if not seeds:
        # Fallback: simple random
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        seeds.append(np.array(points))
    
    # Run basin hopping on each seed in parallel (up to 4 threads)
    n_threads = min(4, len(seeds))
    
    best_energy = float('inf')
    best_positions = None
    
    if n_threads == 1:
        # Single chain
        energy, positions = _search_chain(seeds[0], potential, parameter, size, call_deadline, 0)
        best_energy = energy
        best_positions = positions
    else:
        # Parallel execution
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
                    energy, positions = future.result(timeout=5)
                    if positions is not None and energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    continue
    
    # If no valid result, do a simple relax of the first seed
    if best_positions is None:
        try:
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
    
    # Final check: ensure minimum distance constraint
    # If any pair is too close, we could try to fix, but relax should handle this
    return best_positions
# EVOLVE-BLOCK-END
