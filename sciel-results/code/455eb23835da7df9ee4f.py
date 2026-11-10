# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # 285s safety margin before 330s kill

def _remaining():
    return _DEADLINE - time.time()

def _get_params(kind):
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41

def _get_spacing(kind):
    n_exp, m_exp, c = _get_params(kind)
    # Equilibrium distance for the pair potential part
    # dE/dr = -n/r^(n+1) + c*m/2/r^(m/2+1) = 0
    # n/r^n = c*m/2/r^(m/2)
    # r^(n - m/2) = 2n/(c*m)
    return (2 * n_exp / (c * m_exp)) ** (1.0 / (n_exp - m_exp / 2.0))

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
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * a / np.sqrt(1 + phi**2)
    
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v)
    
    if n > 13:
        # Add face centers
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
            # Add edge midpoints
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
    """Random rejection-sampling cluster."""
    if min_dist is None:
        min_dist = spacing * 0.85
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.0
    points = []
    max_attempts = 10000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        # Fallback: place remaining atoms randomly with larger radius
        while len(points) < n:
            p = rng.uniform(-radius*1.5, radius*1.5, 3)
            points.append(p)
    return np.array(points)

def _min_distance_check(pos):
    """Check minimum distance between any two atoms."""
    n = len(pos)
    if n < 2:
        return np.inf
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _valid_result(energy, pos):
    """Check if a relaxation result is valid."""
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance_check(pos) < 0.35:
        return False
    return True

def _relax_task(args):
    """Task for parallel execution: relax a single candidate."""
    pos, potential, kind = args
    try:
        energy, relaxed_pos = relax(pos, potential, kind)
        if _valid_result(energy, relaxed_pos):
            return energy, relaxed_pos
        else:
            return np.inf, relaxed_pos
    except Exception:
        return np.inf, pos

def _gradient_guided_growth(n, spacing, kind, seed=0):
    """
    Grow the cluster atom by atom using gradient information.
    Start with a small core, then add atoms at positions that minimize
    the energy contribution of the new atom given the existing cluster.
    """
    rng = np.random.default_rng(seed)
    n_exp, m_exp, c = _get_params(kind)
    
    # Start with a small icosahedral core (up to 13 atoms)
    core_size = min(13, n)
    if core_size >= 13:
        core = _icosahedral_positions(core_size, spacing)
    elif core_size >= 2:
        # Simple linear or small cluster
        if core_size == 2:
            core = np.array([[0,0,0], [spacing, 0, 0]])
        else:
            # Small random cluster
            core = _random_cluster(core_size, spacing, seed, min_dist=spacing*0.8)
    else:
        core = np.zeros((1, 3))
    
    positions = list(core)
    
    # Add remaining atoms one by one
    for atom_idx in range(len(positions), n):
        if _remaining() < 30.0:
            # Not enough time, fill rest randomly
            while len(positions) < n:
                p = rng.uniform(-spacing*2, spacing*2, 3)
                positions.append(p)
            return np.array(positions)
        
        # Try several candidate positions for the new atom
        best_pos = None
        best_energy = np.inf
        
        # Candidate 1: Near an existing atom (surface growth)
        existing = np.array(positions)
        # Pick a random existing atom and place new one nearby
        for _ in range(8):
            anchor_idx = rng.integers(0, len(existing))
            anchor = existing[anchor_idx]
            # Random direction
            direction = rng.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)
            candidate = anchor + direction * spacing * rng.uniform(0.9, 1.1)
            
            # Check min distance
            if len(existing) > 0:
                dists = np.linalg.norm(existing - candidate, axis=1)
                if np.min(dists) < 0.35:
                    continue
            
            # Evaluate energy of full cluster with this candidate
            test_pos = np.vstack([existing, candidate.reshape(1, 3)])
            try:
                e, _ = sutton_chen_energy_gradient(test_pos, kind)
                if e < best_energy:
                    best_energy = e
                    best_pos = candidate
            except Exception:
                pass
        
        # Candidate 2: Random position in a shell
        for _ in range(4):
            r = spacing * rng.uniform(1.0, 2.5)
            theta = rng.uniform(0, 2*np.pi)
            phi = rng.uniform(0, np.pi)
            candidate = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            
            if len(existing) > 0:
                dists = np.linalg.norm(existing - candidate, axis=1)
                if np.min(dists) < 0.35:
                    continue
            
            test_pos = np.vstack([existing, candidate.reshape(1, 3)])
            try:
                e, _ = sutton_chen_energy_gradient(test_pos, kind)
                if e < best_energy:
                    best_energy = e
                    best_pos = candidate
            except Exception:
                pass
        
        # If no valid candidate found, use a random one
        if best_pos is None:
            best_pos = rng.uniform(-spacing*2, spacing*2, 3)
        
        positions.append(best_pos)
    
    return np.array(positions)

def _molten_cluster(n, spacing, seed):
    """Generate a molten-like cluster with higher density fluctuations."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.2
    points = []
    for _ in range(n):
        # Sample from a Gaussian-like distribution
        p = rng.normal(0, radius/2.5, 3)
        # Ensure min distance
        attempts = 0
        while any(np.linalg.norm(p - q) < 0.35 for q in points) and attempts < 50:
            p = rng.normal(0, radius/2.5, 3)
            attempts += 1
        points.append(p)
    return np.array(points)

def construct(instance):
    global _START_TIME, _DEADLINE
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    spacing = _get_spacing(kind)
    
    # Generate diverse starting points
    candidates = []
    
    # 1. FCC
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n:
            candidates.append(fcc)
    except Exception:
        pass
    
    # 2. HCP
    try:
        hcp = _hcp_positions(n, spacing)
        if len(hcp) == n:
            candidates.append(hcp)
    except Exception:
        pass
    
    # 3. BCC
    try:
        bcc = _bcc_positions(n, spacing)
        if len(bcc) == n:
            candidates.append(bcc)
    except Exception:
        pass
    
    # 4. Icosahedral
    try:
        ico = _icosahedral_positions(n, spacing)
        if len(ico) == n:
            candidates.append(ico)
    except Exception:
        pass
    
    # 5. Perturbed FCC (small)
    try:
        fcc = _fcc_positions(n, spacing)
        rng = np.random.default_rng(1)
        noise = rng.normal(0, spacing * 0.08, fcc.shape)
        candidates.append(fcc + noise)
    except Exception:
        pass
        
    # 6. Perturbed HCP (small)
    try:
        hcp = _hcp_positions(n, spacing)
        rng = np.random.default_rng(2)
        noise = rng.normal(0, spacing * 0.08, hcp.shape)
        candidates.append(hcp + noise)
    except Exception:
        pass
    
    # 7. Perturbed FCC (medium)
    try:
        fcc = _fcc_positions(n, spacing)
        rng = np.random.default_rng(3)
        noise = rng.normal(0, spacing * 0.2, fcc.shape)
        candidates.append(fcc + noise)
    except Exception:
        pass
    
    # 8. Perturbed HCP (medium)
    try:
        hcp = _hcp_positions(n, spacing)
        rng = np.random.default_rng(4)
        noise = rng.normal(0, spacing * 0.2, hcp.shape)
        candidates.append(hcp + noise)
    except Exception:
        pass
    
    # 9-12. Random clusters
    for seed in [42, 123, 456, 789]:
        try:
            rand = _random_cluster(n, spacing, seed)
            candidates.append(rand)
        except Exception:
            pass
    
    # 13-14. Molten clusters
    for seed in [111, 222]:
        try:
            molten = _molten_cluster(n, spacing, seed)
            candidates.append(molten)
        except Exception:
            pass
    
    # 15-17. Gradient-guided growth (if time permits)
    if _remaining() > 60.0:
        for seed in [7, 8, 9]:
            try:
                grown = _gradient_guided_growth(n, spacing, kind, seed)
                if len(grown) == n:
                    candidates.append(grown)
            except Exception:
                pass
    
    # Ensure we have at least a few candidates
    while len(candidates) < 8:
        try:
            rand = _random_cluster(n, spacing, np.random.randint(1000, 9999))
            candidates.append(rand)
        except Exception:
            break

    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Parallel Relaxation of Initial Candidates
    if _remaining() > 25.0:
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

    # If no valid result yet, try sequential fallback
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

    # Phase 2: Iterative Perturbation Refinement with simulated annealing-like schedule
    if best_positions is not None and _remaining() > 40.0:
        rng = np.random.default_rng(999)
        iteration = 0
        no_improve_count = 0
        while _remaining() > 50.0 and no_improve_count < 5:
            iteration += 1
            # Generate perturbed versions with annealing-like kick size
            base_kick = spacing * 0.6 * (0.7 ** iteration)
            kicks = [base_kick * 0.3, base_kick * 0.6, base_kick, base_kick * 1.5]
            
            tasks = []
            for kick in kicks:
                noise = rng.normal(0, kick, best_positions.shape)
                perturbed = best_positions + noise
                tasks.append((perturbed, potential, kind))
                
            # Relax in parallel
            improved = False
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_relax_task, task): task for task in tasks}
                for future in as_completed(futures):
                    if _remaining() < 15.0:
                        break
                    try:
                        energy, pos = future.result()
                        if energy < best_energy - 1e-8:
                            best_energy = energy
                            best_positions = pos
                            improved = True
                    except Exception:
                        pass
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
    
    # Phase 3: Final High-Precision Polish
    if best_positions is not None and _remaining() > 15.0:
        try:
            energy, relaxed_pos = relax(best_positions, potential, kind)
            if _valid_result(energy, relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            pass

    # Fallback if everything failed
    if best_positions is None:
        try:
            pos = _random_cluster(n, spacing, 12345, min_dist=0.4)
            energy, relaxed_pos = relax(pos, potential, kind)
            if _valid_result(energy, relaxed_pos):
                best_positions = relaxed_pos
            else:
                # Last resort: linear chain
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]
        except Exception:
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i * 0.8, 0, 0]
            
    return best_positions
# EVOLVE-BLOCK-END
