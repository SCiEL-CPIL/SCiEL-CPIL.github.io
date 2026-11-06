"""Dzugutov cluster global minima: icosahedral subunit assembly + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16

# Global deadline tracking
_START_TIME = None
_DEADLINE = 290.0  # safety margin before 300s


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy."""
    A, B, a, b, c, d, m = A_POT, B_POT, a_POT, b_POT, c_POT, d_POT, m_POT
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    # Mask out self-interactions
    np.fill_diagonal(r, np.inf)
    r = np.triu(r, 1)
    r_flat = r.flatten()
    # Only pairs with r < b contribute
    mask = r_flat < b
    r_valid = r_flat[mask]
    if len(r_valid) == 0:
        return 0.0
    
    # Two regions: r < a and a <= r < b
    mask_a = r_valid < a
    r_a = r_valid[mask_a]
    r_b = r_valid[~mask_a]
    
    energy = 0.0
    if len(r_a) > 0:
        # V = A * (r^-m - B) * exp(c/(r - a))
        energy += np.sum(A_POT * (r_a ** (-m_POT) - B_POT) * np.exp(c_POT / (r_a - a_POT)))
    if len(r_b) > 0:
        # V = B * exp(d/(r - b))
        energy += np.sum(B_POT * np.exp(d_POT / (r_b - b_POT)))
    return float(energy)


def _icosahedron_vertices(center, radius, rotation=None):
    """Generate vertices of a regular icosahedron centered at 'center' with given radius."""
    t = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    # Standard icosahedron vertices
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale to desired radius
    verts *= radius
    # Apply rotation if provided
    if rotation is not None:
        verts = verts @ rotation.T
    # Translate to center
    verts += center
    return verts


def _random_rotation(rng):
    """Generate a random 3D rotation matrix."""
    # Use QR decomposition of a random Gaussian matrix
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    d = np.sign(np.diag(R))
    Q *= d
    return Q


def _generate_icosahedral_seed(n, rng, spacing=1.1):
    """Generate a seed structure based on icosahedral subunits."""
    # For icosahedral clusters, the Mackay icosahedron has shells of 1, 12, 42, ...
    # For n between 38-60, we use a core icosahedron (13 atoms) plus partial outer shell
    
    # Core: 1 atom at center + 12 at icosahedron vertices
    core_radius = spacing * 0.55
    center = np.zeros(3)
    
    if n <= 13:
        # Small cluster: just use icosahedron vertices
        if n == 1:
            return np.array([center])
        verts = _icosahedron_vertices(center, core_radius, _random_rotation(rng))
        return verts[:n]
    
    # Start with core icosahedron (13 atoms)
    core_verts = _icosahedron_vertices(center, core_radius, _random_rotation(rng))
    points = [center] + [v for v in core_verts]
    
    # Add outer shell atoms
    outer_radius = spacing * 0.95
    # Icosahedral directions for outer shell
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_dirs = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        # Additional directions for fuller outer shell
        [0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [t, 1, 0], [-t, 1, 0], [t, -1, 0], [-t, -1, 0],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [0, t, 1], [0, -t, 1], [0, t, -1], [0, -t, -1],
        [1, 0, t], [-1, 0, t], [1, 0, -t], [-1, 0, -t],
        [t, 1, 1], [-t, 1, 1], [t, -1, 1], [-t, -1, 1],
        [t, 1, -1], [-t, 1, -1], [t, -1, -1], [-t, -1, -1],
    ], dtype=float)
    ico_dirs /= np.linalg.norm(ico_dirs, axis=1, keepdims=True)
    
    # Shuffle directions
    idx = rng.permutation(len(ico_dirs))
    added = 0
    for i in idx:
        if len(points) >= n:
            break
        p = ico_dirs[i] * outer_radius
        # Check minimum distance
        min_dist = min(np.linalg.norm(p - q) for q in points)
        if min_dist >= 0.55:
            points.append(p)
            added += 1
    
    # If we still need more atoms, add random positions
    while len(points) < n:
        p = rng.uniform(-outer_radius, outer_radius, 3)
        min_dist = min(np.linalg.norm(p - q) for q in points)
        if min_dist >= 0.55:
            points.append(p)
    
    return np.array(points[:n])


def _perturb(positions, rng, sigma=0.15):
    """Perturb positions for basin hopping."""
    perturbed = positions + rng.normal(0, sigma, positions.shape)
    return perturbed


def _basin_hopping(positions, rng, n_iter=30, sigma_start=0.2, sigma_end=0.05):
    """Basin hopping algorithm."""
    current_energy = _dzugutov_energy(positions)
    current_pos = positions.copy()
    best_energy = current_energy
    best_pos = positions.copy()
    
    for i in range(n_iter):
        # Annealing schedule
        progress = i / max(1, n_iter - 1)
        sigma = sigma_start * (1 - progress) + sigma_end * progress
        
        # Perturb
        perturbed = _perturb(current_pos, rng, sigma)
        
        # Relax
        try:
            e_relaxed, relaxed = relax(perturbed, 'dzugutov')
        except Exception:
            continue
        
        # Metropolis acceptance
        delta_e = e_relaxed - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.5):
            current_pos = relaxed
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = relaxed.copy()
    
    return best_energy, best_pos


def _thread_worker(instance, seed_idx, rng_seed):
    """Worker function for parallel basin hopping."""
    n = instance['n']
    rng = np.random.default_rng(rng_seed)
    
    # Generate icosahedral seed
    positions = _generate_icosahedral_seed(n, rng)
    
    # Basin hopping
    energy, best_pos = _basin_hopping(positions, rng, n_iter=25, sigma_start=0.2, sigma_end=0.05)
    
    return energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster using icosahedral seeds + basin hopping."""
    _init_time()
    n = instance['n']
    
    # Determine number of parallel workers and iterations based on time
    time_left = _remaining()
    n_workers = min(4, max(1, int(time_left / 15)))
    
    results = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            rng_seed = n * 1000 + i
            future = executor.submit(_thread_worker, instance, i, rng_seed)
            futures.append(future)
        
        for future in as_completed(futures, timeout=max(1, _remaining() - 5)):
            if _remaining() < 5:
                break
            try:
                energy, positions = future.result(timeout=1)
                results.append((energy, positions))
            except Exception:
                continue
    
    if not results:
        # Fallback: simple random seed
        rng = np.random.default_rng(n)
        radius = 1.1 * n ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), 'dzugutov')
        return positions
    
    # Return the best result
    results.sort(key=lambda x: x[0])
    return results[0][1]
# EVOLVE-BLOCK-END
