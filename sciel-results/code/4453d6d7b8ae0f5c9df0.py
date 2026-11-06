"""Dzugutov cluster global minima: icosahedral shell seeds + basin hopping with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16

# Global deadline tracking
_START_TIME = [None]
_DEADLINE = 290.0  # safety margin before 300s
_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME[0] is None:
        return 0.0
    return time.time() - _START_TIME[0]


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy vectorized."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask for upper triangle
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    e = np.zeros_like(r)
    mask_a = r < a_POT
    mask_b = (r < b_POT) & (~mask_a)
    e[mask_a] = A_POT * (r[mask_a] ** (-m_POT) - B_POT) * np.exp(c_POT / (r[mask_a] - a_POT))
    e[mask_b] = B_POT * np.exp(d_POT / (r[mask_b] - b_POT))
    return float(np.sum(e))


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # normalize to unit sphere
    return verts


def _dodecahedron_vertices():
    """Return 20 vertices of a regular dodecahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = []
    # 8 vertices from cube
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append([sx, sy, sz])
    # 12 vertices from rectangle faces
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            verts.append([0, sx / phi, sx * phi * sz])
            verts.append([sx / phi, sx * phi * sz, 0])
            verts.append([sx * phi * sz, 0, sx / phi])
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _generate_shell_positions(center_pos, shell_radius, n_atoms_on_shell, seed=42):
    """Generate positions on a spherical shell using Fibonacci sphere distribution."""
    rng = np.random.default_rng(seed)
    if n_atoms_on_shell <= 0:
        return np.array([])
    
    # Fibonacci sphere for even distribution
    n = n_atoms_on_shell
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = np.arange(n) * golden_angle
    z = np.linspace(1.0, -1.0, n)
    radius_z = np.sqrt(1.0 - z ** 2)
    
    x = radius_z * np.cos(theta)
    y = radius_z * np.sin(theta)
    
    # Add small random perturbation to break perfect symmetry
    perturb = 0.02 * shell_radius
    x += rng.uniform(-perturb, perturb, n)
    y += rng.uniform(-perturb, perturb, n)
    z += rng.uniform(-perturb, perturb, n)
    
    pos = np.column_stack([x, y, z]) * shell_radius
    pos += center_pos
    return pos


def _build_icosahedral_cluster(n, seed=0):
    """Build an icosahedrally-symmetric cluster for n atoms."""
    rng = np.random.default_rng(seed)
    ico_verts = _icosahedron_vertices()
    
    # Mackay icosahedron layers: 1, 12, 42, 120, 220, ...
    # For clusters of 38-60 atoms, we use combinations of inner shells
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at icosahedron vertices
    # Shell 2: 42 atoms (but we may not need all)
    
    positions = []
    remaining = n
    
    # Center atom
    if remaining > 0:
        positions.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # First shell: 12 icosahedron vertices at radius ~1.1
    shell1_radius = 1.1
    n_shell1 = min(12, remaining)
    if n_shell1 > 0:
        shell1_pos = ico_verts[:n_shell1] * shell1_radius
        positions.extend(shell1_pos)
        remaining -= n_shell1
    
    # Second shell: distribute remaining atoms on a larger shell
    if remaining > 0:
        shell2_radius = 1.1 * 1.6  # ~1.76
        # Use Fibonacci sphere for even distribution
        n_shell2 = remaining
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        theta = np.arange(n_shell2) * golden_angle
        z = np.linspace(1.0, -1.0, n_shell2)
        radius_z = np.sqrt(1.0 - z ** 2)
        x = radius_z * np.cos(theta)
        y = radius_z * np.sin(theta)
        shell2_pos = np.column_stack([x, y, z]) * shell2_radius
        # Add small perturbation
        perturb = 0.03 * shell2_radius
        shell2_pos += rng.uniform(-perturb, perturb, (n_shell2, 3))
        positions.extend(shell2_pos)
    
    return np.array(positions)


def _build_dodecahedral_cluster(n, seed=0):
    """Build a dodecahedrally-symmetric cluster."""
    rng = np.random.default_rng(seed)
    dod_verts = _dodecahedron_vertices()
    
    positions = []
    remaining = n
    
    # Center
    if remaining > 0:
        positions.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # First shell: 20 dodecahedron vertices
    shell1_radius = 1.05
    n_shell1 = min(20, remaining)
    if n_shell1 > 0:
        shell1_pos = dod_verts[:n_shell1] * shell1_radius
        positions.extend(shell1_pos)
        remaining -= n_shell1
    
    # Second shell
    if remaining > 0:
        shell2_radius = 1.05 * 1.7
        n_shell2 = remaining
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        theta = np.arange(n_shell2) * golden_angle
        z = np.linspace(1.0, -1.0, n_shell2)
        radius_z = np.sqrt(1.0 - z ** 2)
        x = radius_z * np.cos(theta)
        y = radius_z * np.sin(theta)
        shell2_pos = np.column_stack([x, y, z]) * shell2_radius
        perturb = 0.03 * shell2_radius
        shell2_pos += rng.uniform(-perturb, perturb, (n_shell2, 3))
        positions.extend(shell2_pos)
    
    return np.array(positions)


def _build_random_cluster(n, seed=0):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min((np.linalg.norm(p - q) for q in points), default=np.inf)
            if min_dist >= 0.85 * spacing:
                points.append(p)
        attempts += 1
    
    # If we didn't get enough, fill in with tighter packing
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _basin_hopping(pos, potential, parameter, max_iter=50, temp=0.5, seed=0):
    """Perform basin hopping optimization."""
    rng = np.random.default_rng(seed)
    best_pos = pos.copy()
    best_energy = _dzugutov_energy(pos)
    current_energy = best_energy
    current_pos = pos.copy()
    
    for i in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Random perturbation
        step_size = 0.15 * (1.0 + 0.5 * np.exp(-i / 20.0))
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax the perturbed structure
        try:
            energy, relaxed = relax(perturbed, potential, parameter)
        except Exception:
            continue
        
        # Metropolis acceptance
        delta_e = energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temp):
            current_pos = relaxed
            current_energy = energy
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    return best_energy, best_pos


def _parallel_search(n, potential, parameter, time_budget):
    """Run parallel basin hopping from multiple seeds."""
    results = []
    lock = threading.Lock()
    
    def worker(seed):
        if _remaining() < 10.0:
            return None
        try:
            # Choose seed type based on seed number
            if seed % 3 == 0:
                pos = _build_icosahedral_cluster(n, seed)
            elif seed % 3 == 1:
                pos = _build_dodecahedral_cluster(n, seed)
            else:
                pos = _build_random_cluster(n, seed)
            
            # Initial relaxation
            try:
                energy, relaxed = relax(pos, potential, parameter)
            except Exception:
                return None
            
            # Basin hopping
            max_iter = max(10, int(time_budget / 50.0))
            bh_energy, bh_pos = _basin_hopping(relaxed, potential, parameter, 
                                                  max_iter=max_iter, temp=0.4, seed=seed)
            return (bh_energy, bh_pos)
        except Exception:
            return None
    
    # Use up to 4 threads
    n_workers = min(4, max(2, int(time_budget / 10.0)))
    seeds = list(range(n_workers * 3))
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, s): s for s in seeds}
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                result = future.result(timeout=2.0)
                if result is not None:
                    with lock:
                        results.append(result)
            except Exception:
                continue
    
    if not results:
        # Fallback: just do a simple relaxation
        pos = _build_icosahedral_cluster(n, 0)
        try:
            energy, relaxed = relax(pos, potential, parameter)
            return relaxed
        except Exception:
            return pos
    
    # Return the best result
    results.sort(key=lambda x: x[0])
    return results[0][1]


def construct(instance):
    """Find low-energy Dzugutov clusters using icosahedral seeds and basin hopping."""
    if _START_TIME[0] is None:
        _START_TIME[0] = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance (roughly equal split among 6 instances)
    # Total deadline is 290s, 6 instances, so ~48s per instance
    # But we track globally
    time_budget = min(45.0, _remaining() - 5.0)
    
    if time_budget < 5.0:
        # Very little time left, just do a quick relaxation
        pos = _build_icosahedral_cluster(size, 0)
        try:
            _, relaxed = relax(pos, potential, parameter)
            return relaxed
        except Exception:
            return pos
    
    return _parallel_search(size, potential, parameter, time_budget)
# EVOLVE-BLOCK-END
