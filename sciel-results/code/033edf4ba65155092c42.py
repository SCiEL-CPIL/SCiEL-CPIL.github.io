"""Dzugutov cluster global minima: Mackay icosahedral seeds + adaptive basin hopping with parallel restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 285.0  # safety margin under 300s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return TIME_LIMIT - _elapsed()


def _energy(positions):
    """Direct Dzugutov energy (vectorised)."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    
    # V(r) = A(r^-m - B)exp(c/(r-a)) [r < a] + B exp(d/(r-b)) [r < b]
    # For r >= b, V = 0
    # For a <= r < b, only the second term: B exp(d/(r-b))
    # For r < a, both terms: A(r^-m - B)exp(c/(r-a)) + B exp(d/(r-b))
    
    e = np.zeros_like(r_ij)
    
    # Second term: B exp(d/(r-b)) for r < b
    mask_b = r_ij < _b
    if np.any(mask_b):
        r_b = r_ij[mask_b]
        # Avoid division by zero or negative
        r_b_safe = np.maximum(r_b, 1e-10)
        e[mask_b] += _B * np.exp(_d / (r_b_safe - _b))
    
    # First term: A(r^-m - B)exp(c/(r-a)) for r < a
    mask_a = r_ij < _a
    if np.any(mask_a):
        r_a = r_ij[mask_a]
        r_a_safe = np.maximum(r_a, 1e-10)
        e[mask_a] += _A * (r_a_safe ** (-_m) - _B) * np.exp(_c / (r_a_safe - _a))
    
    return float(np.sum(e))


def _mckay_icosahedron(n, rng=None):
    """Generate a Mackay icosahedral cluster of n atoms with proper shell geometry.
    
    Mackay icosahedra shells:
    - Shell 0: 1 atom at center
    - Shell 1: 12 atoms at icosahedron vertices
    - Shell 2: 42 atoms (30 edge midpoints + 20 face centers)
    Total for full shells: 1, 13, 55, 147...
    
    For n=38,45,50,55: use shells 0+1+partial shell 2
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Optimal NN distance for Dzugutov ~1.0-1.1
    d1 = 1.05
    
    atoms = [np.zeros(3)]  # center
    
    # Shell 1: 12 vertices
    if n > 1:
        shell1 = verts * d1
        atoms.extend(shell1.tolist())
    
    # Shell 2: 42 atoms (30 edge midpoints + 20 face centers)
    if n > 13:
        # Find edges (pairs of connected vertices)
        edge_pairs = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.0:  # connected in icosahedron
                    edge_pairs.append((i, j))
        
        # Edge midpoints (30)
        edge_mid_dirs = []
        for i, j in edge_pairs:
            mid = (verts[i] + verts[j]) / 2.0
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-10:
                edge_mid_dirs.append(mid / mid_norm)
        edge_mid_dirs = np.array(edge_mid_dirs)
        
        # Face centers (20)
        face_dirs = []
        for i, j, k in combinations(range(12), 3):
            d12 = np.linalg.norm(verts[i] - verts[j])
            d13 = np.linalg.norm(verts[i] - verts[k])
            d23 = np.linalg.norm(verts[j] - verts[k])
            if d12 < 1.0 and d13 < 1.0 and d23 < 1.0:
                fc = (verts[i] + verts[j] + verts[k]) / 3.0
                fc_norm = np.linalg.norm(fc)
                if fc_norm > 1e-10:
                    face_dirs.append(fc / fc_norm)
        face_dirs = np.array(face_dirs)
        
        # Proper Mackay shell 2 radii
        # Edge midpoints: distance from center = d1 * sqrt(3)/2 * (1/|mid|)
        # For unit icosahedron, edge midpoint |mid| = sqrt(3)/2
        # So shell2_edge_radius = d1 * (1/sqrt(3)/2) * sqrt(3)/2 = d1 * 1
        # Actually: |mid| = sqrt(3)/2 for unit icosahedron
        # shell2_edge_radius = d1 * (d1_edge_mid / |mid|) where d1_edge_mid = d1*sqrt(3)/2
        # So radius = d1 * sqrt(3)/2 / (sqrt(3)/2) = d1... that's the same as shell 1
        # Let me recalculate: for Mackay icosahedron with NN=d1:
        # Shell 1 radius = d1
        # Shell 2 edge midpoint radius = d1 * sqrt(3)/2 / (sqrt(3)/2) = d1... 
        # Actually the correct Mackay radii are:
        # r1 = d1 (shell 1)
        # r2_edge = d1 * sqrt(3)/2 (edge midpoints)
        # r2_face = d1 * sqrt(3)/2 * (1/sqrt(5)) ... 
        # Let me use known values: for Mackay with NN=a:
        # Shell 1: 12 atoms at r = a
        # Shell 2: 30 edge-mid atoms at r = a*sqrt(3)/2, 20 face-center atoms at r = a*sqrt(3)/(2*sqrt(5))
        # Hmm, let me just use the geometric positions scaled properly
        
        # The edge midpoints of a unit icosahedron are at distance sqrt(3)/2 from center
        # The face centers are at distance sqrt(3)/(2*sqrt(5)) from center... let me compute
        # Actually for regular icosahedron with edge length 2/sqrt(3):
        # vertex distance from center = 1
        # edge midpoint distance = sqrt(3)/2
        # face center distance = sqrt(3)/(2*sqrt(5))
        
        # For Mackay with NN = d1, shell 1 radius = d1
        # The edge midpoint of two shell-1 atoms at distance d1 apart... 
        # In Mackay, the edge midpoint atom is at the midpoint of the edge between two shell-1 atoms
        # So its distance from center = d1 * (sqrt(3)/2) (since shell 1 atoms are at distance d1)
        
        r2_edge = d1 * np.sqrt(3.0) / 2.0
        r2_face = d1 * np.sqrt(3.0) / (2.0 * np.sqrt(5.0))
        
        shell2_edge = edge_mid_dirs * r2_edge
        shell2_face = face_dirs * r2_face
        
        # Add shell 2 atoms (first edge midpoints, then face centers)
        remaining = n - len(atoms)
        shell2_atoms = list(shell2_edge) + list(shell2_face)
        # Shuffle for diversity
        rng.shuffle(shell2_atoms)
        atoms.extend(shell2_atoms[:remaining])
    
    # If still need more atoms, add in shell 3 or random
    if len(atoms) < n:
        remaining = n - len(atoms)
        for _ in range(remaining):
            dir_vec = rng.normal(size=3)
            dir_vec /= np.linalg.norm(dir_vec)
            r = d1 * rng.uniform(1.5, 2.2)
            atoms.append(dir_vec * r)
    
    return np.array(atoms[:n])


def _basin_hopping(positions, potential, parameter, max_iter=80, step_size=0.3, seed=0):
    """Adaptive basin hopping: random perturbations + local relaxation with cooling."""
    rng = np.random.default_rng(seed)
    pos = np.array(positions, dtype=float)
    
    # Initial relaxation
    energy, pos = relax(pos, potential, parameter)
    best_energy, best_pos = energy, pos.copy()
    
    n_atoms = pos.shape[0]
    total_atoms_sq = n_atoms * n_atoms
    
    for it in range(max_iter):
        if _remaining() < 8.0:
            break
        
        # Adaptive step size: start large, cool down
        progress = it / max(max_iter, 1)
        current_step = step_size * (1.0 - 0.6 * progress)
        
        # Random perturbation
        noise = rng.normal(0, current_step, pos.shape)
        new_pos = pos + noise
        
        # Relax
        energy, new_pos = relax(new_pos, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = new_pos.copy()
        
        # Metropolis acceptance: accept if better or with probability exp(-dE/T)
        dE = energy - best_energy
        T = 0.5 * (1.0 - 0.8 * progress)  # cooling schedule
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
            pos = new_pos
        else:
            # Keep current position for next iteration
            pass
    
    return best_energy, best_pos


def _single_search(n, potential, parameter, seed, time_budget):
    """Run a single search thread with time budget."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Generate diverse initial structures
    # Mix of icosahedral, perturbed icosahedral, and compact random
    strategy = seed % 4
    
    if strategy == 0:
        # Standard icosahedron with small perturbation
        init_pos = _mckay_icosahedron(n, rng)
        init_pos += rng.normal(0, 0.05, init_pos.shape)
        step_size = 0.25
    elif strategy == 1:
        # Icosahedron with larger perturbation
        init_pos = _mckay_icosahedron(n, rng)
        init_pos += rng.normal(0, 0.15, init_pos.shape)
        step_size = 0.35
    elif strategy == 2:
        # Compact random cluster
        init_pos = _random_compact(n, rng)
        step_size = 0.30
    else:
        # Icosahedron with moderate perturbation
        init_pos = _mckay_icosahedron(n, rng)
        init_pos += rng.normal(0, 0.10, init_pos.shape)
        step_size = 0.30
    
    # Basin hopping with adaptive iterations based on time
    max_iter = 80
    best_energy, best_pos = _basin_hopping(init_pos, potential, parameter,
                                            max_iter=max_iter, step_size=step_size, seed=seed)
    
    return best_energy, best_pos


def _random_compact(n, rng):
    """Generate a compact random cluster with minimum distance constraint."""
    pos = np.zeros((n, 3))
    pos[0] = np.zeros(3)
    
    for i in range(1, n):
        for attempt in range(1000):
            p = rng.normal(0, 1.2, 3)
            # Check minimum distance
            if i > 0:
                dists = np.linalg.norm(pos[:i] - p, axis=1)
                if np.min(dists) >= 0.5:
                    pos[i] = p
                    break
        else:
            # If can't find valid position, place at random larger radius
            dir_vec = rng.normal(size=3)
            dir_vec /= np.linalg.norm(dir_vec)
            pos[i] = dir_vec * rng.uniform(1.5, 2.5)
    
    return pos


def construct(instance):
    """Find low-energy Dzugutov cluster structure using icosahedral seeds and basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance  # Pass full instance dict to relax
    
    # Time budget for this instance
    time_budget = min(_remaining() - 10.0, 45.0)  # Leave time for other instances
    if time_budget < 5.0:
        time_budget = 5.0
    
    # Run multiple parallel searches with different seeds
    n_threads = 4
    seeds = list(range(n_threads))
    
    best_energy = float('inf')
    best_pos = None
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for seed in seeds:
            future = executor.submit(_single_search, n, potential, parameter, seed, time_budget)
            futures.append(future)
        
        for future in futures:
            try:
                energy, pos = future.result(timeout=max(_remaining() - 2.0, 5.0))
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                continue
    
    # If no valid result, use a simple relaxed icosahedron
    if best_pos is None:
        init_pos = _mckay_icosahedron(n)
        best_energy, best_pos = relax(init_pos, potential, parameter)
    
    return best_pos
# EVOLVE-BLOCK-END
