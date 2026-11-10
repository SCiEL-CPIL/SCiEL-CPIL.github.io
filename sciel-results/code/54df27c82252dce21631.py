# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 295s safety margin before 330s kill

def _remaining():
    return _DEADLINE - time.time()

def _get_spacing(kind):
    """
    Calculate approximate equilibrium bond length for the pair part of Sutton-Chen.
    """
    if kind == '12-6':
        n, m, c = 12, 6, 144.41
    elif kind == '9-6':
        n, m, c = 9, 6, 39.432
    elif kind == '10-8':
        n, m, c = 10, 8, 34.408
    else:
        return 1.0
    return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))

def _fcc_positions(n, a=1.0):
    """Generate n atoms in FCC lattice positions (truncated)."""
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
    """Generate n atoms in HCP lattice positions (truncated)."""
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
    """Generate n atoms in BCC lattice positions (truncated)."""
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
    """
    Generate n atoms in an icosahedral cluster (Mackay icosahedron truncated).
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * a / np.sqrt(1 + phi**2)
    
    positions = [np.zeros(3)]  # center
    
    # Add vertices
    for v in verts:
        positions.append(v)
    
    # If we need more, add face centers and edge midpoints
    if n > 13:
        # Face centers of the 20 faces
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
        
        # Edge midpoints
        if n > 33:
            edges = []
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(verts[i] - verts[j])
                    if d < 2.1 * a / np.sqrt(1+phi**2):  # approximate edge length
                        edges.append((i, j))
            for e in edges:
                if len(positions) >= n:
                    break
                em = (verts[e[0]] + verts[e[1]]) / 2.0
                positions.append(em)
    
    # Fill remaining with random points on the surface
    while len(positions) < n:
        idx = np.random.randint(0, 12)
        pos = verts[idx] + np.random.normal(0, 0.1 * a, 3)
        positions.append(pos)
    
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    """Random rejection-sampling cluster."""
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

def _molten_cluster(n, spacing, seed):
    """Generate a 'molten' cluster with some structure but high entropy."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = [np.zeros(3)]
    min_dist = spacing * 0.7
    max_attempts = 10000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        # Sample from a Gaussian distribution centered at origin
        p = rng.normal(0, radius * 0.5, 3)
        if np.linalg.norm(p) > radius * 1.2:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    # Fill remaining with more random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _deformed_fcc_positions(n, spacing, seed, deformation=0.15):
    """Generate FCC positions with random deformation to break symmetry."""
    rng = np.random.default_rng(seed)
    fcc = _fcc_positions(n, spacing)
    # Apply random rotation
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, np.pi)
    R = np.array([
        [np.cos(theta)*np.cos(phi), -np.sin(theta), np.cos(theta)*np.sin(phi)],
        [np.sin(theta)*np.cos(phi), np.cos(theta), np.sin(theta)*np.sin(phi)],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    fcc = fcc @ R.T
    # Add random noise
    noise = rng.normal(0, deformation * spacing, fcc.shape)
    return fcc + noise

def _incremental_energy(pos, new_pos, kind):
    """
    Calculate the incremental energy when adding new_pos to existing cluster pos.
    This is the interaction energy between new_pos and all atoms in pos.
    """
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        return 0.0
    
    # Vectorized distance calculation
    diffs = pos - new_pos  # (n, 3)
    dists = np.linalg.norm(diffs, axis=1)  # (n,)
    
    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)
    
    pair_term = np.sum(dists ** (-n_exp))
    many_body_term = np.sqrt(np.sum(dists ** (-m_exp)))
    
    # The incremental energy contribution from adding one atom
    # In Sutton-Chen, each pair contributes half to each atom, and the many-body
    # term is shared. For incremental calculation, we approximate:
    # E_incremental ≈ 0.5 * pair_term - c * many_body_term
    # But this isn't exact because the many-body term for existing atoms also changes.
    # For a good heuristic, we use:
    return 0.5 * pair_term - c * many_body_term

def _gradient_guided_insertion(initial_pos, kind, n_target, max_time_per_step=2.0):
    """
    Grow a cluster by sequentially inserting atoms at positions that minimize
    the incremental energy, using gradient information to guide placement.
    """
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        return initial_pos
    
    spacing = _get_spacing(kind)
    cluster = list(initial_pos)
    
    while len(cluster) < n_target:
        if _remaining() < 10.0:
            break
        
        # Try multiple candidate positions for the next atom
        best_energy = np.inf
        best_pos = None
        
        # Generate candidate positions around existing atoms
        candidates = []
        
        # 1. Positions near each existing atom (first shell)
        for atom in cluster:
            # 12 FCC neighbors, 6 BCC neighbors, 8 icosahedral directions
            directions = np.array([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
            ]).astype(float)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            
            for d in directions:
                for r in [spacing * 0.95, spacing * 1.0, spacing * 1.05, spacing * 1.2]:
                    cand = atom + d * r
                    # Check minimum distance to all existing atoms
                    if all(np.linalg.norm(cand - a) >= 0.36 for a in cluster):
                        candidates.append(cand)
        
        # 2. Some random positions
        rng = np.random.default_rng(len(cluster) * 1000 + 42)
        for _ in range(20):
            cand = rng.uniform(-spacing * n_target**(1/3), spacing * n_target**(1/3), 3)
            if all(np.linalg.norm(cand - a) >= 0.36 for a in cluster):
                candidates.append(cand)
        
        # Evaluate candidates
        for cand in candidates:
            if _remaining() < 5.0:
                break
            try:
                e = _incremental_energy(np.array(cluster), cand, kind)
                if e < best_energy:
                    best_energy = e
                    best_pos = cand
            except Exception:
                pass
        
        if best_pos is None:
            # Fallback: place at a random position far from existing atoms
            rng = np.random.default_rng(len(cluster) * 777 + 13)
            best_pos = rng.uniform(-spacing * 2, spacing * 2, 3)
            while any(np.linalg.norm(best_pos - a) < 0.36 for a in cluster):
                best_pos = rng.uniform(-spacing * 2, spacing * 2, 3)
        
        cluster.append(best_pos)
    
    return np.array(cluster)

def _min_distance_check(pos):
    """Check minimum distance between any two atoms."""
    n = len(pos)
    if n < 2:
        return np.inf
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _valid_result(pos):
    """Check if positions are valid (finite, min distance >= 0.35)."""
    if pos is None:
        return False
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance_check(pos) < 0.35:
        return False
    return True

def _relax_and_validate(pos, potential, kind):
    """Relax positions and validate the result."""
    try:
        energy, relaxed_pos = relax(pos, potential, kind)
        if _valid_result(relaxed_pos):
            return energy, relaxed_pos
    except Exception:
        pass
    return None, None

def _parallel_relax(candidates, potential, kind, max_workers=4):
    """
    Relax multiple candidates in parallel using a thread pool.
    Returns the best (energy, positions) pair.
    """
    best_energy = np.inf
    best_positions = None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, pos in candidates:
            if _remaining() < 10.0:
                break
            future = executor.submit(_relax_and_validate, pos, potential, kind)
            futures[future] = name
        
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                energy, relaxed_pos = future.result()
                if energy is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass
    
    return best_energy, best_positions

def construct(instance):
    global _START_TIME, _DEADLINE
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Use physically motivated spacing based on pair potential minimum
    spacing = _get_spacing(kind)
    
    # Generate diverse starting points
    candidates = []
    
    # 1. FCC
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n and _valid_result(fcc):
            candidates.append(("fcc", fcc))
    except Exception:
        pass
    
    # 2. HCP
    try:
        hcp = _hcp_positions(n, spacing)
        if len(hcp) == n and _valid_result(hcp):
            candidates.append(("hcp", hcp))
    except Exception:
        pass
    
    # 3. BCC
    try:
        bcc = _bcc_positions(n, spacing)
        if len(bcc) == n and _valid_result(bcc):
            candidates.append(("bcc", bcc))
    except Exception:
        pass
    
    # 4. Icosahedral
    try:
        ico = _icosahedral_positions(n, spacing)
        if len(ico) == n and _valid_result(ico):
            candidates.append(("ico", ico))
    except Exception:
        pass
    
    # 5. Perturbed FCC
    try:
        fcc = _fcc_positions(n, spacing)
        rng = np.random.default_rng(1)
        noise = rng.normal(0, spacing * 0.1, fcc.shape)
        perturbed_fcc = fcc + noise
        if _valid_result(perturbed_fcc):
            candidates.append(("pert_fcc", perturbed_fcc))
    except Exception:
        pass
    
    # 6. Perturbed HCP
    try:
        hcp = _hcp_positions(n, spacing)
        rng = np.random.default_rng(2)
        noise = rng.normal(0, spacing * 0.1, hcp.shape)
        perturbed_hcp = hcp + noise
        if _valid_result(perturbed_hcp):
            candidates.append(("pert_hcp", perturbed_hcp))
    except Exception:
        pass
    
    # 7. Molten cluster
    try:
        molten = _molten_cluster(n, spacing, seed=42)
        if _valid_result(molten):
            candidates.append(("molten", molten))
    except Exception:
        pass
    
    # 8. Deformed FCC
    try:
        def_fcc = _deformed_fcc_positions(n, spacing, seed=123)
        if _valid_result(def_fcc):
            candidates.append(("def_fcc", def_fcc))
    except Exception:
        pass
    
    # 9-12. Random clusters with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            rand = _random_cluster(n, spacing, seed)
            if _valid_result(rand):
                candidates.append((f"rand_{seed}", rand))
        except Exception:
            pass
    
    # 13-15. Gradient-guided insertion from different starting points
    for start_seed in [0, 1, 2]:
        if _remaining() < 30.0:
            break
        try:
            # Start from a small seed cluster
            if start_seed == 0:
                seed_cluster = _fcc_positions(5, spacing)
            elif start_seed == 1:
                seed_cluster = _hcp_positions(5, spacing)
            else:
                seed_cluster = np.array([
                    [0, 0, 0],
                    [spacing, 0, 0],
                    [0, spacing, 0],
                    [0, 0, spacing],
                    [spacing * 0.5, spacing * 0.5, spacing * 0.5]
                ])
            
            if len(seed_cluster) == 5 and _valid_result(seed_cluster):
                grown = _gradient_guided_insertion(seed_cluster, kind, n)
                if len(grown) == n and _valid_result(grown):
                    candidates.append((f"grad_insert_{start_seed}", grown))
        except Exception:
            pass
    
    # Relax all candidates in parallel
    best_energy, best_positions = _parallel_relax(candidates, potential, kind, max_workers=4)
    
    # If we have a valid best position and time remains, try to improve it via perturbation
    if best_positions is not None and _remaining() > 40.0:
        # Perturb the best solution
        rng = np.random.default_rng(999)
        # Try a few perturbation levels
        perturbation_candidates = []
        for kick_level in [0.05, 0.1, 0.2, 0.5]:
            try:
                noise = rng.normal(0, kick_level * spacing, best_positions.shape)
                perturbed = best_positions + noise
                if _valid_result(perturbed):
                    perturbation_candidates.append((f"perturb_{kick_level}", perturbed))
            except Exception:
                pass
        
        if perturbation_candidates:
            pert_energy, pert_positions = _parallel_relax(perturbation_candidates, potential, kind, max_workers=4)
            if pert_energy is not None and pert_energy < best_energy:
                best_energy = pert_energy
                best_positions = pert_positions
    
    # If no valid result, try one more random attempt
    if best_positions is None:
        if _remaining() > 10.0:
            try:
                pos = _random_cluster(n, spacing, 12345, min_dist=0.4)
                energy, relaxed_pos = _relax_and_validate(pos, potential, kind)
                if relaxed_pos is not None:
                    best_positions = relaxed_pos
            except Exception:
                # Last resort: simple grid
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
